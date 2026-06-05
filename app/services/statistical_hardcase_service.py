from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from textwrap import fill
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from app.services.export_service import create_export_bundle_dir, export_dataframe_to_dir, export_figure_to_dir, export_json_to_dir
from app.services.paths import REPO_ROOT
from app.services.signal_service import PURE_MODE, SignalConfig, generate_signal
from app.services.test_signal_diagnostics import (
    FailureThresholds,
    build_explanation_text,
    build_signal_difficulty_text,
    compare_model_results,
    compute_component_diagnostics,
    compute_signal_difficulty_descriptors,
)


STATISTICAL_HARDCASE_EXPORT_SUBDIR = "statistical_hardcase_testing"
MODEL_TYPE_COLORS = {"SNN": "#1f77b4", "DNN": "#d62728", "Classical": "#2ca02c", "Unknown": "#7f7f7f"}

DIFFICULTY_PRESETS: dict[str, dict[str, Any]] = {
    "easy": {"noise": (0.0, 0.02), "components": (2, 3), "amplitude_imbalance": (1.0, 2.0), "weak_prob": 0.05, "transient_bias": 0.05, "chirp_bias": 0.10, "candidate_pool": 2},
    "medium": {"noise": (0.01, 0.05), "components": (2, 4), "amplitude_imbalance": (1.5, 4.0), "weak_prob": 0.15, "transient_bias": 0.10, "chirp_bias": 0.20, "candidate_pool": 3},
    "hard": {"noise": (0.03, 0.10), "components": (3, 5), "amplitude_imbalance": (2.0, 8.0), "weak_prob": 0.30, "transient_bias": 0.18, "chirp_bias": 0.28, "candidate_pool": 5},
    "very hard": {"noise": (0.05, 0.15), "components": (3, 5), "amplitude_imbalance": (3.0, 12.0), "weak_prob": 0.45, "transient_bias": 0.24, "chirp_bias": 0.36, "candidate_pool": 7},
    "overlap-heavy": {"noise": (0.02, 0.08), "components": (3, 5), "amplitude_imbalance": (1.5, 6.0), "weak_prob": 0.20, "transient_bias": 0.10, "chirp_bias": 0.20, "candidate_pool": 7, "objective": "overlap"},
    "noise-heavy": {"noise": (0.08, 0.20), "components": (2, 4), "amplitude_imbalance": (1.5, 5.0), "weak_prob": 0.15, "transient_bias": 0.12, "chirp_bias": 0.18, "candidate_pool": 4, "objective": "noise"},
    "chirp-heavy": {"noise": (0.03, 0.10), "components": (3, 5), "amplitude_imbalance": (2.0, 8.0), "weak_prob": 0.20, "transient_bias": 0.10, "chirp_bias": 0.60, "candidate_pool": 6, "objective": "chirp"},
    "weak-component-heavy": {"noise": (0.03, 0.10), "components": (3, 5), "amplitude_imbalance": (3.0, 15.0), "weak_prob": 0.65, "transient_bias": 0.10, "chirp_bias": 0.20, "candidate_pool": 6, "objective": "weak"},
    "mixed difficult cases": {"noise": (0.03, 0.14), "components": (3, 5), "amplitude_imbalance": (2.0, 12.0), "weak_prob": 0.35, "transient_bias": 0.20, "chirp_bias": 0.30, "candidate_pool": 6, "objective": "mixed"},
}


@dataclass
class StatisticalRunResult:
    """Result data returned by a workflow."""
    signal_table: pd.DataFrame
    model_table: pd.DataFrame
    component_table: pd.DataFrame
    signal_comparison_table: pd.DataFrame
    hard_case_table: pd.DataFrame
    artifacts: dict[str, dict[str, Any]]
    summary: dict[str, Any]
    warnings: list[str]


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _metric_value(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _component_type_weights(settings: dict[str, Any], preset: dict[str, Any]) -> tuple[list[str], np.ndarray]:
    component_types = list(settings.get("component_types") or ["harmonic", "amfm", "chirp", "trend", "transient"])
    base_weights = np.asarray([float(settings.get("component_type_weights", {}).get(name, 1.0)) for name in component_types], dtype=float)
    chirp_bias = float(preset.get("chirp_bias", 0.0))
    transient_bias = float(preset.get("transient_bias", 0.0))
    for idx, name in enumerate(component_types):
        if name == "chirp":
            base_weights[idx] *= 1.0 + 2.5 * chirp_bias
        if name == "transient":
            base_weights[idx] *= 1.0 + 2.5 * transient_bias
    if np.sum(base_weights) <= 0:
        base_weights[:] = 1.0
    return component_types, base_weights / np.sum(base_weights)


def _sample_component_types(rng: np.random.Generator, n_components: int, settings: dict[str, Any], preset: dict[str, Any]) -> list[str]:
    component_types, weights = _component_type_weights(settings, preset)
    choices = rng.choice(component_types, size=n_components, replace=True, p=weights)
    return [str(item) for item in choices.tolist()]


def _rescale_components(
    components: np.ndarray,
    rng: np.random.Generator,
    *,
    amplitude_imbalance_range: tuple[float, float],
    weak_component_probability: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    scaled = np.asarray(components, dtype=float).copy()
    n_components = scaled.shape[0]
    target_ratio = float(rng.uniform(amplitude_imbalance_range[0], amplitude_imbalance_range[1]))
    scales = np.ones(n_components, dtype=float)
    if n_components > 1 and target_ratio > 1.0:
        strongest = int(rng.integers(0, n_components))
        weakest = int(rng.integers(0, n_components - 1))
        if weakest >= strongest:
            weakest += 1
        scales[strongest] *= np.sqrt(target_ratio)
        scales[weakest] *= 1.0 / np.sqrt(target_ratio)
    if n_components > 0 and rng.random() < weak_component_probability:
        weak_index = int(rng.integers(0, n_components))
        scales[weak_index] *= float(rng.uniform(0.08, 0.30))
    scaled *= scales[:, None]
    return scaled, {
        "component_scales": scales.tolist(),
        "amplitude_imbalance_target": target_ratio,
    }


def _difficulty_objective(descriptors: dict[str, Any], preset_name: str) -> float:
    overlap = _safe_float(descriptors.get("max_pairwise_spectral_overlap"))
    noise = -_safe_float(descriptors.get("estimated_input_snr_db"))
    weak = 1.0 - _safe_float(descriptors.get("weakest_component_ratio"))
    chirp_scores = [_safe_float(item.get("chirp_likeness")) for item in descriptors.get("component_descriptors", [])]
    chirp = float(np.nanmean(chirp_scores)) if chirp_scores else float("nan")
    objective = preset_name.lower()
    if objective == "overlap-heavy":
        return overlap
    if objective == "noise-heavy":
        return noise
    if objective == "chirp-heavy":
        return chirp
    if objective == "weak-component-heavy":
        return weak
    if objective == "easy":
        return -overlap + _safe_float(descriptors.get("estimated_input_snr_db")) + _safe_float(descriptors.get("weakest_component_ratio"))
    return 0.45 * overlap + 0.25 * noise + 0.20 * weak + 0.10 * chirp


def generate_statistical_signal(
    signal_index: int,
    *,
    settings: dict[str, Any],
    base_seed: int,
) -> dict[str, Any]:
    """Generate statistical signal.
    
    Args:
        signal_index: Index mixed into the random seed.
        settings: Generation/evaluation settings.
        base_seed: Base seed for reproducible signal generation.
    """
    preset_name = str(settings.get("preset_name", "hard")).strip().lower()
    preset = DIFFICULTY_PRESETS.get(preset_name, DIFFICULTY_PRESETS["hard"])
    rng = np.random.default_rng(base_seed + signal_index)
    best_candidate = None
    best_score = None
    candidate_pool = int(settings.get("candidate_pool") or preset.get("candidate_pool", 4))
    signal_length_override = settings.get("signal_length")
    fs = int(settings.get("fs", 256))
    duration = float(settings.get("duration", 4.0))
    if signal_length_override:
        duration = max(0.25, float(signal_length_override) / float(fs))

    component_min, component_max = settings.get("num_components_range", preset["components"])
    noise_min, noise_max = settings.get("noise_level_range", preset["noise"])
    amplitude_imbalance_range = settings.get("amplitude_imbalance_range", preset["amplitude_imbalance"])
    weak_component_probability = float(settings.get("weak_component_probability", preset["weak_prob"]))

    for candidate_idx in range(candidate_pool):
        n_components = int(rng.integers(int(component_min), int(component_max) + 1))
        noise_level = float(rng.uniform(float(noise_min), float(noise_max)))
        component_types = _sample_component_types(rng, n_components, settings, preset)
        generator_seed = int(base_seed + signal_index * 1000 + candidate_idx)
        generated = generate_signal(
            SignalConfig(
                signal_type="mixed",
                n_components=n_components,
                duration=duration,
                fs=fs,
                noise_level=0.0,
                seed=generator_seed,
                generation_mode=PURE_MODE,
                selected_component_types=component_types,
            )
        )
        scaled_components, scale_meta = _rescale_components(
            generated.components,
            rng,
            amplitude_imbalance_range=(float(amplitude_imbalance_range[0]), float(amplitude_imbalance_range[1])),
            weak_component_probability=weak_component_probability,
        )
        mixture = np.sum(scaled_components, axis=0)
        mixture = mixture + rng.normal(0.0, noise_level, size=mixture.shape)
        descriptors = compute_signal_difficulty_descriptors(
            mixture,
            scaled_components,
            None,
            generated.component_names,
            fs=fs,
        )
        score = _difficulty_objective(descriptors, preset_name)
        candidate = {
            "signal_id": f"sig_{signal_index:04d}",
            "signal_index": signal_index,
            "preset_name": preset_name,
            "mixture": mixture,
            "components": scaled_components,
            "time_axis": generated.t,
            "component_names": generated.component_names,
            "component_types": component_types,
            "noise_level": noise_level,
            "seed": generator_seed,
            "generator_metadata": {
                "n_components": n_components,
                "noise_level": noise_level,
                "component_types": component_types,
                **scale_meta,
            },
            "descriptors": descriptors,
            "difficulty_score": score,
            "difficulty_text": build_signal_difficulty_text(descriptors),
        }
        if best_candidate is None or score > float(best_score):
            best_candidate = candidate
            best_score = score
    return best_candidate


def evaluate_model_on_signal(
    *,
    signal_case: dict[str, Any],
    model_name: str,
    display_name: str,
    model_type: str,
    family: str,
    checkpoint: str,
    parameter_count: int | None,
    inference_time_ms: float | None,
    prediction: np.ndarray,
    report: dict[str, Any],
    thresholds: FailureThresholds,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Evaluate model on signal."""
    aligned_prediction = np.asarray(report.get("aligned_prediction", prediction), dtype=float)
    component_table = compute_component_diagnostics(
        report,
        signal_case["component_names"],
        y_true=signal_case["components"],
        y_pred=aligned_prediction,
        fs=None,
        thresholds=thresholds,
    )
    macro = report.get("macro_average", {}) or {}
    clean = report.get("clean_sum_metrics", {}) or {}
    difficulty_summary = compute_signal_difficulty_descriptors(
        signal_case["mixture"],
        signal_case["components"],
        aligned_prediction,
        signal_case["component_names"],
        fs=None,
    )
    explanation = build_explanation_text(
        component_table,
        macro_metrics=macro,
        clean_metrics=clean,
        difficulty_summary=difficulty_summary,
    )
    failure_labels = sorted(
        {
            label
            for labels in component_table.get("failure_labels", pd.Series(dtype=object)).tolist()
            for label in (labels or [])
        }
    )
    model_row = {
        "signal_id": signal_case["signal_id"],
        "model_name": model_name,
        "display_name": display_name,
        "model_type": model_type,
        "family": family,
        "checkpoint": checkpoint,
        "parameter_count": parameter_count,
        "inference_time_ms": inference_time_ms,
        "macro_corr": _metric_value(macro.get("corr")),
        "macro_snr": _metric_value(macro.get("snr_db")),
        "macro_si_sdr": _metric_value(macro.get("si_sdr_db")),
        "test_loss": _metric_value(report.get("clean_sum_metrics", {}).get("mse")),
        "hard_case_score": compute_hard_case_score(component_table, macro, difficulty_summary),
        "failure_labels": ", ".join(failure_labels),
        "failure_rate": 1.0 if bool(failure_labels) else 0.0,
        "explanation": explanation,
        "signal_difficulty_score": signal_case["difficulty_score"],
        "noise_level": signal_case["generator_metadata"]["noise_level"],
        "num_components": signal_case["generator_metadata"]["n_components"],
        "spectral_overlap_score": _safe_float(difficulty_summary.get("max_pairwise_spectral_overlap")),
        "weakest_component_energy_ratio": _safe_float(difficulty_summary.get("weakest_component_ratio")),
        "amplitude_imbalance": _safe_float(difficulty_summary.get("amplitude_ratio_strongest_to_weakest")),
        "estimated_input_snr_db": _safe_float(difficulty_summary.get("estimated_input_snr_db")),
        "component_types": ", ".join(signal_case["component_types"]),
        "worst_component": component_table.sort_values("nmse", ascending=False, na_position="last").iloc[0]["component"]
        if not component_table.empty
        else "",
        "signal_difficulty_text": signal_case["difficulty_text"],
        "clean_sum_corr": _metric_value(clean.get("corr")),
        "clean_sum_nmse": _metric_value(clean.get("nmse")),
    }
    component_rows = component_table.copy()
    component_rows["signal_id"] = signal_case["signal_id"]
    component_rows["model_name"] = model_name
    component_rows["display_name"] = display_name
    component_rows["model_type"] = model_type
    component_rows["component_type"] = [
        signal_case["component_types"][idx] if idx < len(signal_case["component_types"]) else ""
        for idx in range(len(component_rows))
    ]
    return model_row, component_rows


def compute_hard_case_score(component_df: pd.DataFrame, macro_metrics: dict[str, Any], signal_summary: dict[str, Any]) -> float:
    """Compute hard case score.
    
    Args:
        component_df: Input dataframe.
        macro_metrics: Macro-level metric values.
        signal_summary: Per-signal summary table.
    """
    macro_corr = _metric_value(macro_metrics.get("corr"))
    macro_snr = _metric_value(macro_metrics.get("snr_db"))
    residual = _safe_float(component_df["residual_energy_ratio"].mean()) if not component_df.empty else float("nan")
    spectral = _safe_float(component_df["spectral_convergence"].mean()) if not component_df.empty else float("nan")
    overlap = _safe_float(signal_summary.get("max_pairwise_spectral_overlap"))
    return float(
        0.45 * max(0.0, 1.0 - macro_corr)
        + 0.20 * max(0.0, -macro_snr / 10.0)
        + 0.20 * max(0.0, residual)
        + 0.10 * max(0.0, spectral)
        + 0.05 * max(0.0, overlap)
    )


def compute_signal_level_comparison(model_table: pd.DataFrame) -> pd.DataFrame:
    """Compute signal level comparison.
    
    Args:
        model_table: Per-model result table.
    """
    if model_table.empty or "signal_id" not in model_table.columns:
        return pd.DataFrame(
            columns=[
                "signal_id",
                "best_snn_model",
                "best_dnn_model",
                "best_snn_macro_corr",
                "best_dnn_macro_corr",
                "best_snn_macro_snr",
                "best_dnn_macro_snr",
                "average_snn_macro_corr",
                "average_dnn_macro_corr",
                "average_snn_macro_snr",
                "average_dnn_macro_snr",
                "macro_corr_gap",
                "macro_snr_gap",
                "dnn_beats_snn",
                "snn_competitive",
                "snn_fails_dnn_succeeds",
            ]
        )
    rows: list[dict[str, Any]] = []
    for signal_id, group in model_table.groupby("signal_id", dropna=False):
        snn = group[group["model_type"] == "SNN"]
        dnn = group[group["model_type"] == "DNN"]
        best_snn = snn.sort_values("macro_corr", ascending=False).iloc[0] if not snn.empty else None
        best_dnn = dnn.sort_values("macro_corr", ascending=False).iloc[0] if not dnn.empty else None
        avg_snn_corr = _safe_float(snn["macro_corr"].mean()) if not snn.empty else float("nan")
        avg_dnn_corr = _safe_float(dnn["macro_corr"].mean()) if not dnn.empty else float("nan")
        avg_snn_snr = _safe_float(snn["macro_snr"].mean()) if not snn.empty else float("nan")
        avg_dnn_snr = _safe_float(dnn["macro_snr"].mean()) if not dnn.empty else float("nan")
        row = {
            "signal_id": signal_id,
            "best_snn_model": None if best_snn is None else best_snn["display_name"],
            "best_dnn_model": None if best_dnn is None else best_dnn["display_name"],
            "best_snn_macro_corr": None if best_snn is None else best_snn["macro_corr"],
            "best_dnn_macro_corr": None if best_dnn is None else best_dnn["macro_corr"],
            "best_snn_macro_snr": None if best_snn is None else best_snn["macro_snr"],
            "best_dnn_macro_snr": None if best_dnn is None else best_dnn["macro_snr"],
            "average_snn_macro_corr": avg_snn_corr,
            "average_dnn_macro_corr": avg_dnn_corr,
            "average_snn_macro_snr": avg_snn_snr,
            "average_dnn_macro_snr": avg_dnn_snr,
            "macro_corr_gap": None if best_snn is None or best_dnn is None else float(best_dnn["macro_corr"] - best_snn["macro_corr"]),
            "macro_snr_gap": None if best_snn is None or best_dnn is None else float(best_dnn["macro_snr"] - best_snn["macro_snr"]),
            "dnn_beats_snn": None if best_snn is None or best_dnn is None else bool((best_dnn["macro_corr"] - best_snn["macro_corr"]) > 0.05),
            "snn_competitive": None if best_snn is None or best_dnn is None else bool(abs(best_dnn["macro_corr"] - best_snn["macro_corr"]) <= 0.03),
            "snn_fails_dnn_succeeds": None
            if best_snn is None or best_dnn is None
            else bool(best_snn["macro_corr"] < 0.60 and best_dnn["macro_corr"] >= 0.75),
        }
        if best_snn is not None:
            for key in ["noise_level", "num_components", "spectral_overlap_score", "weakest_component_energy_ratio", "amplitude_imbalance", "component_types"]:
                row[key] = best_snn.get(key)
        elif best_dnn is not None:
            for key in ["noise_level", "num_components", "spectral_overlap_score", "weakest_component_energy_ratio", "amplitude_imbalance", "component_types"]:
                row[key] = best_dnn.get(key)
        rows.append(row)
    return pd.DataFrame(rows)


def build_hard_case_table(model_table: pd.DataFrame, signal_comparison: pd.DataFrame) -> pd.DataFrame:
    """Build hard case table.
    
    Args:
        model_table: Per-model result table.
        signal_comparison: Per-signal comparison table.
    """
    if model_table.empty or "signal_id" not in model_table.columns:
        return pd.DataFrame()
    comparison_columns = [
        column
        for column in ["signal_id", "macro_corr_gap", "macro_snr_gap", "dnn_beats_snn", "snn_competitive", "snn_fails_dnn_succeeds"]
        if column in signal_comparison.columns
    ]
    comparison_frame = signal_comparison[comparison_columns] if comparison_columns else pd.DataFrame()
    table = model_table.merge(comparison_frame, on="signal_id", how="left") if not comparison_frame.empty else model_table.copy()
    sort_cols = [column for column in ["model_type", "hard_case_score", "macro_corr_gap"] if column in table.columns]
    return table.sort_values(sort_cols, ascending=[True, False, False][: len(sort_cols)], na_position="last")


def summarize_statistical_results(signal_table: pd.DataFrame, model_table: pd.DataFrame, component_table: pd.DataFrame, signal_comparison: pd.DataFrame) -> dict[str, Any]:
    """Summarize statistical results."""
    summary: dict[str, Any] = {
        "num_signals": int(signal_table["signal_id"].nunique()) if not signal_table.empty and "signal_id" in signal_table.columns else 0,
        "num_model_evaluations": int(len(model_table)),
        "selected_models": sorted(model_table["display_name"].dropna().astype(str).unique().tolist()) if not model_table.empty and "display_name" in model_table.columns else [],
    }
    if "model_type" in model_table.columns:
        snn = model_table[model_table["model_type"] == "SNN"]
        dnn = model_table[model_table["model_type"] == "DNN"]
    else:
        snn = pd.DataFrame()
        dnn = pd.DataFrame()
    summary["snn_failure_rate"] = float(snn["failure_rate"].mean()) if not snn.empty else None
    summary["dnn_failure_rate"] = float(dnn["failure_rate"].mean()) if not dnn.empty else None
    summary["avg_snn_hard_case_score"] = float(snn["hard_case_score"].mean()) if not snn.empty else None
    summary["avg_dnn_hard_case_score"] = float(dnn["hard_case_score"].mean()) if not dnn.empty else None
    summary["dnn_beats_snn_rate"] = float(signal_comparison["dnn_beats_snn"].mean()) if "dnn_beats_snn" in signal_comparison and not signal_comparison.empty else None
    summary["snn_competitive_rate"] = float(signal_comparison["snn_competitive"].mean()) if "snn_competitive" in signal_comparison and not signal_comparison.empty else None
    summary["snn_fail_dnn_success_rate"] = float(signal_comparison["snn_fails_dnn_succeeds"].mean()) if "snn_fails_dnn_succeeds" in signal_comparison and not signal_comparison.empty else None
    summary["most_problematic_component_types"] = (
        component_table.groupby("component_type")["nmse"].mean().sort_values(ascending=False).head(5).index.tolist()
        if not component_table.empty and "component_type" in component_table.columns and "nmse" in component_table.columns
        else []
    )
    summary["most_common_failure_labels_snn"] = (
        model_table[model_table["model_type"] == "SNN"]["failure_labels"].str.split(", ").explode().dropna().value_counts().head(5).to_dict()
        if not model_table.empty and "model_type" in model_table.columns and "failure_labels" in model_table.columns
        else {}
    )
    summary["overall_text"] = build_overall_summary_text(model_table, component_table, signal_comparison)
    return summary


def build_overall_summary_text(model_table: pd.DataFrame, component_table: pd.DataFrame, signal_comparison: pd.DataFrame) -> str:
    """Build overall summary text.
    
    Args:
        model_table: Per-model result table.
        component_table: Per-component result table.
        signal_comparison: Per-signal comparison table.
    """
    statements = []
    if not signal_comparison.empty and signal_comparison["macro_corr_gap"].notna().any():
        gap_sorted = signal_comparison.sort_values("macro_corr_gap", ascending=False)
        high_gap = gap_sorted.head(min(20, len(gap_sorted)))
        overlap = _safe_float(high_gap["spectral_overlap_score"].mean()) if "spectral_overlap_score" in high_gap.columns else float("nan")
        noise = _safe_float(high_gap["noise_level"].mean()) if "noise_level" in high_gap.columns else float("nan")
        if np.isfinite(overlap) and overlap > 0.50:
            statements.append("SNN failures were most common in signals with high spectral overlap.")
        if np.isfinite(noise) and noise > 0.06:
            statements.append("Noise-heavy signals increased the SNN-DNN gap.")
    if not component_table.empty:
        comp_rank = component_table.groupby("component_type")[["corr", "snr_db"]].mean().sort_values("corr")
        if not comp_rank.empty:
            worst = str(comp_rank.index[0])
            best = str(comp_rank.index[-1])
            statements.append(f"{worst} components were the hardest on average, while {best} components were relatively stable.")
    if not model_table.empty and "model_type" in model_table.columns and "failure_labels" in model_table.columns:
        snn = model_table[model_table["model_type"] == "SNN"]
        if not snn.empty:
            failure_mix = snn["failure_labels"].str.split(", ").explode().dropna().value_counts()
            if not failure_mix.empty:
                common = ", ".join(failure_mix.head(3).index.tolist())
                statements.append(f"The most common SNN failure modes were {common}.")
    if not statements:
        return "No strong statistical hard-case pattern was detected from the current run."
    return " ".join(statements)


def _base_axis(figsize: tuple[float, float] = (9.5, 4.8)) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def _styled_xticklabels(ax: plt.Axes, labels: list[str], rotation: int = 30) -> None:
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels([fill(str(label), width=16) for label in labels], rotation=rotation, ha="right")


def boxplot_metric_by_model(model_table: pd.DataFrame, metric: str, title: str) -> plt.Figure | None:
    """Boxplot metric by model.
    
    Args:
        metric: Metric column or metric name to use.
        title: Chart title.
        model_table: Per-model result table.
    """
    subset = model_table.dropna(subset=[metric]).copy()
    if subset.empty:
        return None
    ordered_labels = subset.groupby("display_name")[metric].mean().sort_values(ascending=False).index.tolist()
    series = [subset.loc[subset["display_name"] == label, metric].to_numpy() for label in ordered_labels]
    fig, ax = _base_axis((max(10.0, 0.7 * len(ordered_labels)), 5.4))
    ax.boxplot(series, patch_artist=True)
    for patch, label in zip(ax.artists, ordered_labels):
        model_type = subset.loc[subset["display_name"] == label, "model_type"].iloc[0]
        patch.set_facecolor(MODEL_TYPE_COLORS.get(model_type, MODEL_TYPE_COLORS["Unknown"]))
    _styled_xticklabels(ax, ordered_labels)
    ax.set_title(title)
    ax.set_ylabel(metric.replace("_", " "))
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def violin_metric_by_model_type(model_table: pd.DataFrame, metric: str, title: str) -> plt.Figure | None:
    """Violin metric by model type.
    
    Args:
        metric: Metric column or metric name to use.
        title: Chart title.
        model_table: Per-model result table.
    """
    subset = model_table.dropna(subset=[metric, "model_type"]).copy()
    if subset.empty:
        return None
    labels = [item for item in ["SNN", "DNN", "Classical", "Unknown"] if item in subset["model_type"].unique()]
    series = [subset.loc[subset["model_type"] == label, metric].to_numpy() for label in labels]
    fig, ax = _base_axis()
    parts = ax.violinplot(series, showmeans=True, showextrema=False)
    for body, label in zip(parts["bodies"], labels):
        body.set_facecolor(MODEL_TYPE_COLORS.get(label, MODEL_TYPE_COLORS["Unknown"]))
        body.set_alpha(0.7)
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_title(title)
    ax.set_ylabel(metric.replace("_", " "))
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def histogram_hardcase_scores(model_table: pd.DataFrame) -> plt.Figure | None:
    """Histogram hardcase scores.
    
    Args:
        model_table: Per-model result table.
    """
    subset = model_table.dropna(subset=["hard_case_score"]).copy()
    if subset.empty:
        return None
    fig, ax = _base_axis()
    for model_type, group in subset.groupby("model_type", dropna=False):
        ax.hist(group["hard_case_score"], bins=20, alpha=0.5, label=model_type, color=MODEL_TYPE_COLORS.get(model_type, MODEL_TYPE_COLORS["Unknown"]))
    ax.set_title("Hard-case score distribution for SNNs vs DNNs")
    ax.set_xlabel("Hard-case score")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def bar_failure_rate_per_model(model_table: pd.DataFrame) -> plt.Figure | None:
    """Bar failure rate per model.
    
    Args:
        model_table: Per-model result table.
    """
    grouped = model_table.groupby(["display_name", "model_type"], dropna=False)["failure_rate"].mean().reset_index().sort_values("failure_rate", ascending=False)
    if grouped.empty:
        return None
    fig, ax = _base_axis((max(10.0, 0.75 * len(grouped)), 5.4))
    positions = np.arange(len(grouped))
    ax.bar(positions, grouped["failure_rate"], color=[MODEL_TYPE_COLORS.get(item, MODEL_TYPE_COLORS["Unknown"]) for item in grouped["model_type"]])
    _styled_xticklabels(ax, grouped["display_name"].tolist())
    ax.set_title("Failure rate per model")
    ax.set_ylabel("Failure rate")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def bar_average_hardcase_per_model(model_table: pd.DataFrame) -> plt.Figure | None:
    """Bar average hardcase per model.
    
    Args:
        model_table: Per-model result table.
    """
    grouped = model_table.groupby(["display_name", "model_type"], dropna=False)["hard_case_score"].mean().reset_index().sort_values("hard_case_score", ascending=False)
    if grouped.empty:
        return None
    fig, ax = _base_axis((max(10.0, 0.75 * len(grouped)), 5.4))
    positions = np.arange(len(grouped))
    ax.bar(positions, grouped["hard_case_score"], color=[MODEL_TYPE_COLORS.get(item, MODEL_TYPE_COLORS["Unknown"]) for item in grouped["model_type"]])
    _styled_xticklabels(ax, grouped["display_name"].tolist())
    ax.set_title("Average hard-case score per model")
    ax.set_ylabel("Hard-case score")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def robustness_ranking_chart(model_table: pd.DataFrame) -> plt.Figure | None:
    """Robustness ranking chart.
    
    Args:
        model_table: Per-model result table.
    """
    grouped = model_table.groupby(["display_name", "model_type"], dropna=False)["macro_corr"].mean().reset_index().sort_values("macro_corr", ascending=False)
    if grouped.empty:
        return None
    fig, ax = _base_axis((max(10.0, 0.75 * len(grouped)), 5.4))
    positions = np.arange(len(grouped))
    ax.bar(positions, grouped["macro_corr"], color=[MODEL_TYPE_COLORS.get(item, MODEL_TYPE_COLORS["Unknown"]) for item in grouped["model_type"]])
    _styled_xticklabels(ax, grouped["display_name"].tolist())
    ax.set_title("Model robustness ranking on difficult signals")
    ax.set_ylabel("Average macro correlation")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def scatter_best_snn_vs_best_dnn(signal_comparison: pd.DataFrame, metric: str, title: str) -> plt.Figure | None:
    """Scatter best snn vs best dnn.
    
    Args:
        metric: Metric column or metric name to use.
        title: Chart title.
        signal_comparison: Per-signal comparison table.
    """
    required = [f"best_snn_{metric}", f"best_dnn_{metric}"]
    subset = signal_comparison.dropna(subset=required).copy()
    if subset.empty:
        return None
    fig, ax = _base_axis()
    ax.scatter(subset[required[0]], subset[required[1]], alpha=0.75, color="#6a3d9a")
    lo = min(subset[required[0]].min(), subset[required[1]].min())
    hi = max(subset[required[0]].max(), subset[required[1]].max())
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="#333333", linewidth=1.0)
    ax.set_xlabel(f"Best SNN {metric}")
    ax.set_ylabel(f"Best DNN {metric}")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def scatter_gap_vs_condition(signal_comparison: pd.DataFrame, condition: str, title: str) -> plt.Figure | None:
    """Scatter gap vs condition.
    
    Args:
        title: Chart title.
        signal_comparison: Per-signal comparison table.
        condition: Project value for this call.
    """
    subset = signal_comparison.dropna(subset=["macro_corr_gap", condition]).copy()
    if subset.empty:
        return None
    fig, ax = _base_axis()
    ax.scatter(subset[condition], subset["macro_corr_gap"], alpha=0.75, color="#ff7f0e")
    ax.set_xlabel(condition.replace("_", " "))
    ax.set_ylabel("Macro correlation gap (DNN - SNN)")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def bar_competitiveness(signal_comparison: pd.DataFrame) -> plt.Figure | None:
    """Bar competitiveness.
    
    Args:
        signal_comparison: Per-signal comparison table.
    """
    if signal_comparison.empty or "dnn_beats_snn" not in signal_comparison.columns:
        return None
    values = {
        "DNN clearly better": float(signal_comparison["dnn_beats_snn"].mean()),
        "SNN competitive": float(signal_comparison["snn_competitive"].mean()) if "snn_competitive" in signal_comparison.columns else float("nan"),
        "SNN fails/DNN succeeds": float(signal_comparison["snn_fails_dnn_succeeds"].mean()) if "snn_fails_dnn_succeeds" in signal_comparison.columns else float("nan"),
    }
    fig, ax = _base_axis((8.5, 4.6))
    labels = list(values.keys())
    vals = list(values.values())
    ax.bar(labels, vals, color=["#d62728", "#1f77b4", "#9467bd"])
    ax.set_ylim(0.0, 1.0)
    ax.set_title("SNN vs DNN outcome rates")
    ax.set_ylabel("Fraction of signals")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def scatter_metric_vs_condition(model_table: pd.DataFrame, condition: str, metric: str, title: str) -> plt.Figure | None:
    """Scatter metric vs condition.
    
    Args:
        metric: Metric column or metric name to use.
        title: Chart title.
    """
    subset = model_table.dropna(subset=[condition, metric]).copy()
    if subset.empty:
        return None
    fig, ax = _base_axis()
    for model_type, group in subset.groupby("model_type", dropna=False):
        ax.scatter(group[condition], group[metric], alpha=0.55, label=model_type, color=MODEL_TYPE_COLORS.get(model_type, MODEL_TYPE_COLORS["Unknown"]))
    ax.set_xlabel(condition.replace("_", " "))
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig


def component_type_bar(component_table: pd.DataFrame, metric: str, title: str, *, aggregate: str = "mean") -> plt.Figure | None:
    """Component type bar.
    
    Args:
        metric: Metric column or metric name to use.
        title: Chart title.
    """
    subset = component_table.dropna(subset=[metric, "component_type"]).copy()
    if subset.empty:
        return None
    if aggregate == "failure_rate":
        grouped = subset.assign(flag=subset[metric].astype(bool)).groupby("component_type")["flag"].mean().sort_values(ascending=False)
    else:
        grouped = subset.groupby("component_type")[metric].mean().sort_values(ascending=(metric in {"spectral_convergence", "residual_energy_ratio"}))
    fig, ax = _base_axis((8.5, 4.6))
    ax.bar(grouped.index.tolist(), grouped.values, color="#2ca02c")
    ax.set_title(title)
    ax.set_ylabel(metric.replace("_", " "))
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def _explode_failure_labels(model_table: pd.DataFrame, model_types: list[str] | None = None) -> pd.DataFrame:
    subset = model_table.copy()
    if model_types:
        subset = subset[subset["model_type"].isin(model_types)]
    required_columns = {"model_type", "failure_labels"}
    if subset.empty or not required_columns.issubset(subset.columns):
        return pd.DataFrame(columns=["evaluation_id", "model_type", "failure_label"])
    exploded = subset[["model_type", "failure_labels"]].copy()
    exploded["evaluation_id"] = exploded.index
    exploded["failure_label"] = exploded["failure_labels"].fillna("").astype(str).str.split(", ")
    exploded = exploded.explode("failure_label")
    exploded["failure_label"] = exploded["failure_label"].astype(str)
    exploded = exploded[exploded["failure_label"].str.len() > 0]
    return exploded[["evaluation_id", "model_type", "failure_label"]]


def _filter_universal_failure_labels(exploded: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"evaluation_id", "failure_label"}
    if exploded.empty or not required_columns.issubset(exploded.columns):
        return exploded
    total_evaluations = exploded["evaluation_id"].nunique()
    if total_evaluations <= 1:
        return exploded
    label_coverage = exploded.groupby("failure_label")["evaluation_id"].nunique()
    universal_labels = label_coverage[label_coverage >= total_evaluations].index
    if len(universal_labels) == 0:
        return exploded
    filtered = exploded[~exploded["failure_label"].isin(universal_labels)].copy()
    return filtered


def failure_label_frequency(model_table: pd.DataFrame, model_type: str | None = None) -> plt.Figure | None:
    """Failure label frequency.
    
    Args:
        model_table: Per-model result table.
        model_type: Project value for this call.
    """
    exploded = _explode_failure_labels(model_table, [model_type] if model_type else None)
    exploded = _filter_universal_failure_labels(exploded)
    if exploded.empty:
        return None
    counts = exploded["failure_label"].value_counts().head(12)
    fig, ax = _base_axis((9.0, 4.8))
    ax.barh(counts.index.tolist(), counts.values, color=MODEL_TYPE_COLORS.get(model_type or "Unknown", "#7f7f7f"))
    ax.set_title(f"Failure label frequency{' for ' + model_type if model_type else ''}")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    return fig


def failure_label_frequency_snn_vs_dnn(model_table: pd.DataFrame, *, top_n: int = 12, normalize: bool = True) -> plt.Figure | None:
    """Failure label frequency snn vs dnn.
    
    Args:
        model_table: Per-model result table.
        top_n: Project value for this call.
        normalize: Project value for this call.
    """
    model_types = ["SNN", "DNN"]
    subset = model_table[model_table["model_type"].isin(model_types)].copy()
    exploded = _explode_failure_labels(subset, model_types)
    exploded = _filter_universal_failure_labels(exploded)
    if exploded.empty:
        return None
    counts = exploded.groupby(["failure_label", "model_type"]).size().unstack(fill_value=0)
    counts = counts.reindex(columns=model_types, fill_value=0)
    if normalize:
        totals = subset.groupby("model_type").size().reindex(model_types, fill_value=0).replace(0, np.nan)
        counts = counts.div(totals, axis=1)
        x_label = "Fraction of evaluations with label"
        title = "Failure label frequency for SNN vs DNN (normalized)"
    else:
        x_label = "Count"
        title = "Failure label frequency for SNN vs DNN"
    counts = counts.loc[counts.max(axis=1).sort_values(ascending=False).index].head(top_n)
    if counts.empty:
        return None
    labels = [fill(str(label), width=20) for label in counts.index.tolist()]
    positions = np.arange(len(counts))
    height = 0.38
    fig, ax = _base_axis((10.5, max(4.8, 0.5 * len(counts) + 1.8)))
    ax.barh(positions - height / 2, counts["SNN"].to_numpy(), height=height, color=MODEL_TYPE_COLORS["SNN"], label="SNN")
    ax.barh(positions + height / 2, counts["DNN"].to_numpy(), height=height, color=MODEL_TYPE_COLORS["DNN"], label="DNN")
    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel(x_label)
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    return fig


def failure_condition_heatmap(model_table: pd.DataFrame, condition: str) -> plt.Figure | None:
    """Failure condition heatmap.
    
    Args:
        model_table: Per-model result table.
        condition: Project value for this call.
    """
    subset = model_table[["failure_labels", condition]].dropna().copy()
    if subset.empty:
        return None
    subset["condition_bin"] = pd.qcut(subset[condition], q=min(4, subset[condition].nunique()), duplicates="drop")
    exploded = subset.assign(evaluation_id=subset.index, failure_label=subset["failure_labels"].str.split(", ")).explode("failure_label")
    exploded = exploded[exploded["failure_label"].astype(str).str.len() > 0]
    exploded = _filter_universal_failure_labels(exploded)
    if exploded.empty:
        return None
    pivot = exploded.pivot_table(index="failure_label", columns="condition_bin", values=condition, aggfunc="count", fill_value=0)
    fig, ax = _base_axis((10.5, max(4.5, 0.5 * len(pivot) + 1.5)))
    image = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="Oranges")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([fill(str(item), width=18) for item in pivot.index])
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([fill(str(item), width=18) for item in pivot.columns], rotation=25, ha="right")
    ax.set_title(f"Signal condition versus failure label: {condition}")
    fig.colorbar(image, ax=ax, shrink=0.8)
    fig.tight_layout()
    return fig


def efficiency_scatter(model_table: pd.DataFrame, x_metric: str, y_metric: str, title: str) -> plt.Figure | None:
    """Efficiency scatter.
    
    Args:
        title: Chart title.
    """
    subset = model_table.dropna(subset=[x_metric, y_metric]).copy()
    if subset.empty:
        return None
    fig, ax = _base_axis()
    for model_type, group in subset.groupby("model_type", dropna=False):
        ax.scatter(group[x_metric], group[y_metric], alpha=0.6, label=model_type, color=MODEL_TYPE_COLORS.get(model_type, MODEL_TYPE_COLORS["Unknown"]))
    ax.set_xlabel(x_metric.replace("_", " "))
    ax.set_ylabel(y_metric.replace("_", " "))
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig


def thesis_summary_where_snns_fail(model_table: pd.DataFrame) -> plt.Figure | None:
    """Thesis summary where snns fail.
    
    Args:
        model_table: Per-model result table.
    """
    snn = model_table[model_table["model_type"] == "SNN"]
    if snn.empty:
        return None
    grouped = snn.groupby("display_name")[["hard_case_score", "macro_corr"]].mean().sort_values("hard_case_score", ascending=False)
    fig, ax = _base_axis((max(9.5, 0.75 * len(grouped)), 5.2))
    positions = np.arange(len(grouped))
    ax.bar(positions, grouped["hard_case_score"], color="#d62728", alpha=0.75, label="Hard-case score")
    ax2 = ax.twinx()
    ax2.plot(positions, grouped["macro_corr"], color="#1f77b4", marker="o", linewidth=1.6, label="Macro corr")
    _styled_xticklabels(ax, grouped.index.tolist())
    ax.set_ylabel("Hard-case score")
    ax2.set_ylabel("Macro correlation")
    ax.set_title("Where SNNs fail")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def export_statistical_bundle(
    *,
    result: StatisticalRunResult,
    config: dict[str, Any],
    figures: dict[str, plt.Figure],
) -> Path:
    """Export statistical bundle.
    
    Args:
        config: Configuration object for the operation.
        result: Project value for this call.
        figures: Project value for this call.
    """
    bundle_dir = create_export_bundle_dir(STATISTICAL_HARDCASE_EXPORT_SUBDIR)
    export_dataframe_to_dir(result.signal_table, bundle_dir / "tables" / "signals.csv")
    export_dataframe_to_dir(result.model_table, bundle_dir / "tables" / "model_results.csv")
    export_dataframe_to_dir(result.component_table, bundle_dir / "tables" / "component_results.csv")
    export_dataframe_to_dir(result.signal_comparison_table, bundle_dir / "tables" / "signal_comparison.csv")
    export_dataframe_to_dir(result.hard_case_table, bundle_dir / "tables" / "hard_cases.csv")
    export_json_to_dir(result.summary, bundle_dir / "summary.json")
    export_json_to_dir(config, bundle_dir / "run_config.json")
    export_json_to_dir({"warnings": result.warnings}, bundle_dir / "warnings.json")
    artifacts_dir = bundle_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    for signal_id, artifact in result.artifacts.items():
        np.savez_compressed(
            artifacts_dir / f"{signal_id}.npz",
            time_axis=np.asarray(artifact["time_axis"], dtype=float),
            mixture=np.asarray(artifact["mixture"], dtype=float),
            true_components=np.asarray(artifact["components"], dtype=float),
        )
        (artifacts_dir / f"{signal_id}.json").write_text(json.dumps(_jsonable_artifact_metadata(artifact), indent=2), encoding="utf-8")
    for name, fig in figures.items():
        export_figure_to_dir(fig, bundle_dir / "figures" / f"{name}.png")
        export_figure_to_dir(fig, bundle_dir / "figures" / f"{name}.svg")
        export_figure_to_dir(fig, bundle_dir / "figures" / f"{name}.pdf")
    return bundle_dir


def _jsonable_artifact_metadata(artifact: dict[str, Any]) -> dict[str, Any]:
    return {
        "signal_id": artifact.get("signal_id"),
        "component_names": artifact.get("component_names"),
        "component_types": artifact.get("component_types"),
        "generator_metadata": artifact.get("generator_metadata"),
        "descriptors": artifact.get("descriptors"),
        "difficulty_score": artifact.get("difficulty_score"),
        "difficulty_text": artifact.get("difficulty_text"),
        "model_predictions": {
            key: {
                "display_name": value.get("display_name"),
                "model_type": value.get("model_type"),
                "macro_corr": value.get("macro_corr"),
                "macro_snr": value.get("macro_snr"),
                "hard_case_score": value.get("hard_case_score"),
                "failure_labels": value.get("failure_labels"),
                "explanation": value.get("explanation"),
            }
            for key, value in artifact.get("model_predictions", {}).items()
        },
    }
