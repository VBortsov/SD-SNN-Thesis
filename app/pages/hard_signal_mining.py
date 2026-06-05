from __future__ import annotations

from io import BytesIO
from datetime import datetime
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from app.services.inference_service import count_parameters, evaluate_prediction, load_model, load_train_config, run_inference
from app.services.paths import APP_EXPORTS_DIR, REPO_ROOT, SAVED_MODELS_DIR, ensure_app_dirs
from app.services.signal_service import (
    PURE_MODE,
    SUPPORTED_COMPONENT_TYPES,
    SUPPORTED_PRESETS,
    SignalConfig,
    generate_signal,
    validate_pure_component_types,
)
from app.services.statistical_hardcase_service import (
    DIFFICULTY_PRESETS,
    StatisticalRunResult,
    bar_average_hardcase_per_model,
    bar_competitiveness,
    bar_failure_rate_per_model,
    boxplot_metric_by_model,
    build_hard_case_table,
    compare_model_results as compare_direct_model_results,
    component_type_bar,
    compute_signal_level_comparison,
    efficiency_scatter,
    export_statistical_bundle,
    evaluate_model_on_signal,
    failure_condition_heatmap,
    failure_label_frequency,
    failure_label_frequency_snn_vs_dnn,
    generate_statistical_signal,
    histogram_hardcase_scores,
    robustness_ranking_chart,
    scatter_best_snn_vs_best_dnn,
    scatter_gap_vs_condition,
    scatter_metric_vs_condition,
    summarize_statistical_results,
    thesis_summary_where_snns_fail,
    violin_metric_by_model_type,
)
from app.services.test_signal_diagnostics import (
    build_explanation_text,
    build_signal_difficulty_text,
    build_thresholds,
    compare_model_results,
    compute_component_diagnostics,
    compute_signal_difficulty_descriptors,
    figure_component_metric_comparison,
    figure_failure_label_heatmap,
    figure_prediction_preview,
    save_diagnostic_artifact,
    thresholds_to_dict,
)


EPS = 1e-12


def _select_checkpoint_from_run_dir(run_dir: Path) -> Path | None:
    for pattern in ["*_best.pt", "*_last.pt", "*_epoch_*.pt", "*.pt"]:
        matches = [item for item in sorted(run_dir.glob(pattern)) if not item.name.endswith("_weights_only.pt")]
        if matches:
            return matches[0]
    return None


def _checkpoint_train_samples(checkpoint_path: str | None) -> int | None:
    config = load_train_config(checkpoint_path)
    try:
        value = config.get("train_samples") if isinstance(config, dict) else None
        return int(value)
    except (TypeError, ValueError):
        return None


def _available_checkpoints(selected_model: dict) -> list[str]:
    model_key = str(selected_model.get("key", selected_model.get("model_key", ""))).strip()
    candidates: list[str] = []
    if model_key and SAVED_MODELS_DIR.exists():
        discovered: list[tuple[str, str]] = []
        for run_dir in sorted(SAVED_MODELS_DIR.iterdir()):
            if not run_dir.is_dir():
                continue
            if run_dir.name != model_key and not run_dir.name.startswith(f"{model_key}_"):
                continue
            checkpoint = _select_checkpoint_from_run_dir(run_dir)
            if checkpoint is None:
                continue
            rel_path = str(checkpoint.relative_to(REPO_ROOT))
            if _checkpoint_train_samples(rel_path) == 8000:
                discovered.append((run_dir.name, rel_path))
        candidates.extend([path for _, path in sorted(discovered)])

    default_ckpt = str(selected_model.get("default_checkpoint", "")).strip()
    if default_ckpt and _checkpoint_train_samples(default_ckpt) == 8000:
        candidates.insert(0, default_ckpt)

    unique: list[str] = []
    seen: set[str] = set()
    for checkpoint in candidates:
        if checkpoint and checkpoint not in seen:
            unique.append(checkpoint)
            seen.add(checkpoint)
    return unique


def _dominant_frequency(x: np.ndarray, fs: int | None) -> float:
    x = np.asarray(x, dtype=float)
    if x.size < 4:
        return float("nan")
    centered = x - np.mean(x)
    mag = np.abs(np.fft.rfft(centered))
    if mag.size <= 1:
        return float("nan")
    mag[0] = 0.0
    d = 1.0 / fs if fs and fs > 0 else 1.0
    freq = np.fft.rfftfreq(x.size, d=d)
    return float(freq[int(np.argmax(mag))])


def _spectral_centroid(x: np.ndarray, fs: int | None) -> float:
    x = np.asarray(x, dtype=float)
    if x.size < 4:
        return float("nan")
    mag = np.abs(np.fft.rfft(x - np.mean(x)))
    d = 1.0 / fs if fs and fs > 0 else 1.0
    freq = np.fft.rfftfreq(x.size, d=d)
    return float(np.sum(freq * mag) / (np.sum(mag) + EPS))


def _window_centroids(x: np.ndarray, fs: int | None, windows: int = 8) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    if x.size < windows * 4:
        return np.array([], dtype=float), np.array([], dtype=float)
    chunks = np.array_split(x, windows)
    centers = []
    values = []
    for idx, chunk in enumerate(chunks):
        if chunk.size < 4:
            continue
        centers.append((idx + 0.5) / len(chunks))
        values.append(_spectral_centroid(chunk, fs))
    return np.asarray(centers, dtype=float), np.asarray(values, dtype=float)


def _amplitude_modulation_depth(x: np.ndarray) -> float:
    """Envelope proxy based on smoothed absolute amplitude spread."""
    x = np.asarray(x, dtype=float)
    if x.size < 8:
        return float("nan")
    envelope = np.abs(x - np.mean(x))
    window = max(3, min(51, x.size // 20))
    kernel = np.ones(window, dtype=float) / window
    smooth = np.convolve(envelope, kernel, mode="same")
    low = float(np.percentile(smooth, 5))
    high = float(np.percentile(smooth, 95))
    return float((high - low) / (high + low + EPS))


def _chirp_rate_proxy(x: np.ndarray, fs: int | None, duration: float | None) -> float:
    """Estimate frequency drift as a linear slope of windowed spectral centroids."""
    centers, values = _window_centroids(x, fs)
    if centers.size < 3 or not np.all(np.isfinite(values)):
        return float("nan")
    time_scale = float(duration) if duration and duration > 0 else 1.0
    slope, _ = np.polyfit(centers * time_scale, values, deg=1)
    return float(slope)


def _nonlinearity_proxy(x: np.ndarray, fs: int | None) -> float:
    """Residual centroid variation after removing a linear frequency trend."""
    centers, values = _window_centroids(x, fs)
    if centers.size < 4 or not np.all(np.isfinite(values)):
        return float("nan")
    fit = np.polyval(np.polyfit(centers, values, deg=1), centers)
    return float(np.std(values - fit) / (np.mean(np.abs(values)) + EPS))


def _spectral_overlap(a: np.ndarray, b: np.ndarray) -> float:
    spec_a = np.abs(np.fft.rfft(a - np.mean(a)))
    spec_b = np.abs(np.fft.rfft(b - np.mean(b)))
    return float(np.sum(np.minimum(spec_a, spec_b)) / (min(np.sum(spec_a), np.sum(spec_b)) + EPS))


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) < EPS or np.std(b) < EPS:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def estimate_descriptors(
    mixture: np.ndarray,
    components: np.ndarray,
    component_names: list[str],
    *,
    fs: int | None,
    duration: float | None,
) -> tuple[dict, pd.DataFrame]:
    """Best-effort descriptors derived from observed mixture and clean components."""
    clean_sum = np.sum(components, axis=0)
    noise = mixture - clean_sum if mixture.shape == clean_sum.shape else None
    noise_power = float(np.mean(noise**2)) if noise is not None else float("nan")
    clean_power = float(np.mean(clean_sum**2))
    snr = float(10.0 * np.log10((clean_power + EPS) / (noise_power + EPS))) if noise is not None else float("nan")

    energies = np.sum(components**2, axis=1)
    total_energy = float(np.sum(energies))
    dominant_freqs = [_dominant_frequency(comp, fs) for comp in components]

    comp_rows = []
    for idx, comp in enumerate(components):
        comp_rows.append(
            {
                "component": component_names[idx] if idx < len(component_names) else f"C{idx + 1}",
                "energy": float(energies[idx]),
                "energy_share": float(energies[idx] / (total_energy + EPS)),
                "dominant_frequency": dominant_freqs[idx],
                "amplitude_modulation_depth": _amplitude_modulation_depth(comp),
                "chirp_rate_proxy": _chirp_rate_proxy(comp, fs, duration),
                "nonlinearity_proxy": _nonlinearity_proxy(comp, fs),
            }
        )

    freq_diffs = [
        abs(dominant_freqs[i] - dominant_freqs[j])
        for i in range(len(dominant_freqs))
        for j in range(i + 1, len(dominant_freqs))
        if np.isfinite(dominant_freqs[i]) and np.isfinite(dominant_freqs[j])
    ]
    spectral_overlaps = []
    corr_overlaps = []
    for i in range(components.shape[0]):
        for j in range(i + 1, components.shape[0]):
            spectral_overlaps.append(_spectral_overlap(components[i], components[j]))
            corr_overlaps.append(abs(_corr(components[i], components[j])))

    positive_energies = energies[energies > EPS]
    summary = {
        "estimated_noise_power": noise_power,
        "estimated_snr_db": snr,
        "sampling_rate": int(fs) if fs is not None else None,
        "sampling_rate_units": "Hz" if fs is not None else "sample-domain",
        "dominant_to_weakest_energy_ratio": float(np.max(positive_energies) / np.min(positive_energies))
        if positive_energies.size
        else float("nan"),
        "min_pairwise_frequency_distance": float(np.min(freq_diffs)) if freq_diffs else float("nan"),
        "mean_spectral_overlap": float(np.mean(spectral_overlaps)) if spectral_overlaps else float("nan"),
        "max_spectral_overlap": float(np.max(spectral_overlaps)) if spectral_overlaps else float("nan"),
        "mean_abs_component_correlation": float(np.mean(corr_overlaps)) if corr_overlaps else float("nan"),
        "max_abs_component_correlation": float(np.max(corr_overlaps)) if corr_overlaps else float("nan"),
    }
    return summary, pd.DataFrame(comp_rows)


def compute_hardness(report: dict, weights: dict[str, float]) -> tuple[float, dict]:
    """Combine component and reconstruction failures into one adjustable score."""
    component_metrics = report.get("component_metrics", [])
    component_nmse = [float(item.get("nmse", np.nan)) for item in component_metrics]
    component_corr = [float(item.get("corr", np.nan)) for item in component_metrics]
    clean_sum = report.get("clean_sum_metrics", {}) or {}

    reconstruction_nmse = float(clean_sum.get("nmse", np.nan))
    mean_component_nmse = float(np.nanmean(component_nmse)) if component_nmse else float("nan")
    worst_component_nmse = float(np.nanmax(component_nmse)) if component_nmse else float("nan")
    macro_corr = float(np.nanmean(component_corr)) if component_corr else float("nan")
    corr_disagreement = float(1.0 - np.clip(macro_corr, -1.0, 1.0))

    terms = {
        "reconstruction_nmse": reconstruction_nmse,
        "mean_component_nmse": mean_component_nmse,
        "worst_component_nmse": worst_component_nmse,
        "correlation_disagreement": corr_disagreement,
    }
    score = 0.0
    for key, value in terms.items():
        if np.isfinite(value):
            score += float(weights.get(key, 0.0)) * value
    return float(score), terms


def _plot_hard_case(case: dict):
    t = case["t"]
    mixture = case["mixture"]
    y_true = case["components"]
    y_pred = case["prediction"]
    names = case["component_names"]
    clean_true = np.sum(y_true, axis=0)
    clean_pred = np.sum(y_pred, axis=0)

    rows = y_true.shape[0] + 2
    fig, axes = plt.subplots(rows, 1, figsize=(12, 2.4 * rows), sharex=True)
    axes[0].plot(t, mixture, color="black", linewidth=1.1, label="Observed/noisy")
    axes[0].plot(t, clean_true, linewidth=1.0, alpha=0.8, label="Clean sum")
    axes[0].set_title("Observed Signal")
    axes[0].legend(loc="upper right")

    axes[1].plot(t, clean_true, linewidth=1.1, label="Clean sum true")
    axes[1].plot(t, clean_pred, linewidth=1.0, alpha=0.85, label="Predicted reconstruction")
    axes[1].plot(t, clean_true - clean_pred, linewidth=0.8, alpha=0.7, label="Residual")
    axes[1].set_title("Clean-Sum Reconstruction")
    axes[1].legend(loc="upper right")

    for idx in range(y_true.shape[0]):
        label = names[idx] if idx < len(names) else f"C{idx + 1}"
        axes[idx + 2].plot(t, y_true[idx], linewidth=1.0, label=f"True {label}")
        axes[idx + 2].plot(t, y_pred[idx], linewidth=0.95, alpha=0.85, label=f"Pred {label}")
        axes[idx + 2].plot(t, y_true[idx] - y_pred[idx], linewidth=0.75, alpha=0.65, label="Residual")
        axes[idx + 2].legend(loc="upper right")
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    return fig


def _json_ready(value):
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_ready(v) for v in value]
    if isinstance(value, tuple):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return value.item()
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _save_hard_case(case: dict, search_settings: dict, fig) -> tuple[Path, Path, Path]:
    ensure_app_dirs()
    out_dir = APP_EXPORTS_DIR / "hard_cases"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"hard_case_{case['model_key']}_seed{case['candidate_seed']}_{stamp}"
    json_path = out_dir / f"{base}.json"
    npz_path = out_dir / f"{base}.npz"
    png_path = out_dir / f"{base}.png"

    clean_reconstruction = np.sum(case["components"], axis=0)
    predicted_reconstruction = np.sum(case["prediction"], axis=0)
    np.savez_compressed(
        npz_path,
        t=case["t"],
        observed_signal=case["mixture"],
        clean_components=case["components"],
        predicted_components=case["prediction"],
        clean_reconstruction=clean_reconstruction,
        predicted_reconstruction=predicted_reconstruction,
        reconstruction_residual=clean_reconstruction - predicted_reconstruction,
    )
    fig.savefig(png_path, dpi=160, bbox_inches="tight")

    metadata = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": case["model_key"],
        "display_name": case["display_name"],
        "checkpoint": case["checkpoint"],
        "candidate_index": case["candidate_index"],
        "candidate_seed": case["candidate_seed"],
        "hardness_score": case["hardness_score"],
        "hardness_terms": case["hardness_terms"],
        "metrics": case["report"],
        "descriptors": case["descriptor_summary"],
        "component_descriptors": case["descriptor_frame"].to_dict("records"),
        "component_names": case["component_names"],
        "generator_config": case["generator_config"],
        "search_settings": search_settings,
        "arrays_file": str(npz_path.relative_to(REPO_ROOT)),
        "figure_file": str(png_path.relative_to(REPO_ROOT)),
    }
    json_path.write_text(json.dumps(_json_ready(metadata), indent=2), encoding="utf-8")
    return json_path, npz_path, png_path


def _case_summary_row(case: dict) -> dict:
    macro = case["report"].get("macro_average", {}) or {}
    clean = case["report"].get("clean_sum_metrics", {}) or {}
    return {
        "rank_score": case["hardness_score"],
        "model": case["display_name"],
        "candidate_seed": case["candidate_seed"],
        "macro_corr": float(macro.get("corr", np.nan)),
        "mean_component_nmse": case["hardness_terms"].get("mean_component_nmse", np.nan),
        "worst_component_nmse": case["hardness_terms"].get("worst_component_nmse", np.nan),
        "reconstruction_nmse": float(clean.get("nmse", np.nan)),
        "estimated_snr_db": case["descriptor_summary"].get("estimated_snr_db", np.nan),
        "min_freq_distance": case["descriptor_summary"].get("min_pairwise_frequency_distance", np.nan),
        "max_spectral_overlap": case["descriptor_summary"].get("max_spectral_overlap", np.nan),
    }


def _infer_model_type(model_spec: dict) -> str:
    family = str(model_spec.get("family", "")).strip().lower()
    depth = str(model_spec.get("depth_label", "")).strip().lower()
    if depth == "shallow":
        return "SNN"
    if depth == "deep":
        return "DNN"
    if family in {"classical", "baseline"}:
        return "Classical"
    return "Unknown"


def _metric_value(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _figure_download_bytes(fig) -> bytes:
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=160, bbox_inches="tight")
    buffer.seek(0)
    return buffer.getvalue()


def _load_uploaded_signal(uploaded_file) -> tuple[dict | None, str | None]:
    try:
        data = np.load(uploaded_file, allow_pickle=True)
    except Exception as exc:
        return None, f"Failed to read uploaded signal file: {exc}"

    keys = set(data.files)
    mixture_key = next((key for key in ["mixture", "observed_signal"] if key in keys), None)
    if mixture_key is None:
        return None, "Uploaded NPZ must contain `mixture` or `observed_signal`."
    mixture = np.asarray(data[mixture_key], dtype=float).reshape(-1)

    component_key = next((key for key in ["components", "true_components", "clean_components"] if key in keys), None)
    components = None
    if component_key is not None:
        components = np.asarray(data[component_key], dtype=float)
        if components.ndim != 2:
            return None, "Uploaded components must have shape [n_components, signal_length]."

    time_key = next((key for key in ["t", "time_axis"] if key in keys), None)
    if time_key is not None:
        time_axis = np.asarray(data[time_key], dtype=float).reshape(-1)
    else:
        time_axis = np.arange(mixture.size, dtype=float)

    fs = None
    if "fs" in keys:
        try:
            fs = int(np.asarray(data["fs"]).reshape(-1)[0])
        except Exception:
            fs = None

    names = []
    if "component_names" in keys:
        try:
            names = [str(item) for item in np.asarray(data["component_names"]).tolist()]
        except Exception:
            names = []
    if not names and components is not None:
        names = [f"component_{idx + 1}" for idx in range(components.shape[0])]

    return {
        "source": uploaded_file.name,
        "mixture": mixture,
        "components": components,
        "time_axis": time_axis,
        "fs": fs,
        "component_names": names,
    }, None


def _selected_model_specs_for_statistical(registry: list[dict], preset: str, custom_names: list[str]) -> list[tuple[str, dict, str]]:
    enabled = [item for item in registry if item.get("enabled", True)]
    display_map = {item["display_name"]: item for item in enabled}
    if preset == "All SNN models":
        names = [item["display_name"] for item in enabled if str(item.get("depth_label", "")).lower() == "shallow"]
    elif preset == "All DNN models":
        names = [item["display_name"] for item in enabled if str(item.get("depth_label", "")).lower() == "deep"]
    elif preset == "Best SNN vs Best DNN":
        snn = [item for item in enabled if str(item.get("depth_label", "")).lower() == "shallow"]
        dnn = [item for item in enabled if str(item.get("depth_label", "")).lower() == "deep"]
        names = []
        if snn:
            names.append(snn[0]["display_name"])
        if dnn:
            names.append(dnn[0]["display_name"])
    elif preset == "Custom":
        names = custom_names
    else:
        names = [item["display_name"] for item in enabled]
    rows = []
    for name in names:
        if name not in display_map:
            continue
        spec = display_map[name]
        checkpoints = _available_checkpoints(spec)
        if checkpoints:
            rows.append((name, spec, checkpoints[0]))
    return rows


def _build_statistical_figures(model_table: pd.DataFrame, component_table: pd.DataFrame, signal_comparison: pd.DataFrame) -> dict[str, plt.Figure]:
    figures: dict[str, plt.Figure] = {}
    for key, fig in {
        "macro_corr_box_by_model": boxplot_metric_by_model(model_table, "macro_corr", "Macro correlation by model"),
        "macro_snr_box_by_model": boxplot_metric_by_model(model_table, "macro_snr", "Macro SNR by model"),
        "macro_corr_violin_by_type": violin_metric_by_model_type(model_table, "macro_corr", "Macro correlation by model type"),
        "hard_case_histogram": histogram_hardcase_scores(model_table),
        "failure_rate_per_model": bar_failure_rate_per_model(model_table),
        "hard_case_score_per_model": bar_average_hardcase_per_model(model_table),
        "robustness_ranking": robustness_ranking_chart(model_table),
        "best_snn_vs_best_dnn_corr": scatter_best_snn_vs_best_dnn(signal_comparison, "macro_corr", "Best SNN vs best DNN macro correlation per signal"),
        "best_snn_vs_best_dnn_snr": scatter_best_snn_vs_best_dnn(signal_comparison, "macro_snr", "Best SNN vs best DNN macro SNR per signal"),
        "gap_vs_overlap": scatter_gap_vs_condition(signal_comparison, "spectral_overlap_score", "SNN-DNN performance gap vs spectral overlap"),
        "gap_vs_noise": scatter_gap_vs_condition(signal_comparison, "noise_level", "SNN-DNN performance gap vs noise level"),
        "competitiveness": bar_competitiveness(signal_comparison),
        "corr_vs_noise": scatter_metric_vs_condition(model_table, "noise_level", "macro_corr", "Macro correlation vs noise level"),
        "snr_vs_noise": scatter_metric_vs_condition(model_table, "noise_level", "macro_snr", "Macro SNR vs noise level"),
        "corr_vs_overlap": scatter_metric_vs_condition(model_table, "spectral_overlap_score", "macro_corr", "Macro correlation vs spectral overlap"),
        "snr_vs_overlap": scatter_metric_vs_condition(model_table, "spectral_overlap_score", "macro_snr", "Macro SNR vs spectral overlap"),
        "corr_vs_weakest_ratio": scatter_metric_vs_condition(model_table, "weakest_component_energy_ratio", "macro_corr", "Macro correlation vs weakest component energy ratio"),
        "snr_vs_weakest_ratio": scatter_metric_vs_condition(model_table, "weakest_component_energy_ratio", "macro_snr", "Macro SNR vs weakest component energy ratio"),
        "corr_vs_imbalance": scatter_metric_vs_condition(model_table, "amplitude_imbalance", "macro_corr", "Macro correlation vs amplitude imbalance"),
        "corr_vs_num_components": scatter_metric_vs_condition(model_table, "num_components", "macro_corr", "Macro correlation vs number of components"),
        "component_failure_rate": component_type_bar(component_table.assign(component_failed=component_table["failure_label_text"].astype(str).str.len() > 0), "component_failed", "Per-component failure rate by component type", aggregate="failure_rate"),
        "component_average_corr": component_type_bar(component_table, "corr", "Average correlation by component type"),
        "component_average_snr": component_type_bar(component_table, "snr_db", "Average SNR by component type"),
        "component_spectral_mismatch": component_type_bar(component_table, "spectral_convergence", "Spectral mismatch by component type"),
        "failure_labels_snn_vs_dnn": failure_label_frequency_snn_vs_dnn(model_table),
        "failure_labels_snn": failure_label_frequency(model_table, "SNN"),
        "failure_labels_dnn": failure_label_frequency(model_table, "DNN"),
        "failure_vs_overlap_heatmap": failure_condition_heatmap(model_table, "spectral_overlap_score"),
        "failure_vs_noise_heatmap": failure_condition_heatmap(model_table, "noise_level"),
        "corr_vs_inference_time": efficiency_scatter(model_table, "inference_time_ms", "macro_corr", "Macro correlation vs inference time"),
        "failure_vs_parameters": efficiency_scatter(model_table, "parameter_count", "failure_rate", "Failure rate vs parameter count"),
        "where_snns_fail": thesis_summary_where_snns_fail(model_table),
    }.items():
        if fig is not None:
            figures[key] = fig
    return figures


def _render_statistical_hardcase_testing(registry: list[dict]) -> None:
    st.caption("Run many difficult synthetic signals through selected models and analyze where SNNs break down.")
    enabled = [item for item in registry if item.get("enabled", True)]
    if not enabled:
        st.warning("No enabled models are available.")
        return

    model_display_names = [item["display_name"] for item in enabled]
    with st.sidebar:
        st.subheader("Statistical Hard-Case Testing")
        st.caption("Only checkpoints with `train_samples = 8000` are used.")
        model_preset = st.selectbox(
            "Model set selector",
            ["All enabled", "All SNN models", "All DNN models", "Best SNN vs Best DNN", "Custom"],
            key="stat_hardcase_model_preset",
        )
        custom_models = st.multiselect("Custom model group", model_display_names, default=model_display_names[: min(4, len(model_display_names))], key="stat_hardcase_models")
        preset_name = st.selectbox("Difficulty preset", list(DIFFICULTY_PRESETS.keys()), index=list(DIFFICULTY_PRESETS.keys()).index("hard"), key="stat_hardcase_preset")
        num_signals = int(st.number_input("Number of generated test signals", min_value=4, max_value=500, value=40, step=4, key="stat_hardcase_num_signals"))
        random_seed = int(st.number_input("Random seed", min_value=0, max_value=99999999, value=2026, step=1, key="stat_hardcase_seed"))
        fs = int(st.number_input("Sampling rate", min_value=64, max_value=4096, value=256, step=64, key="stat_hardcase_fs"))
        duration = float(st.slider("Duration (s)", min_value=0.5, max_value=8.0, value=4.0, step=0.1, key="stat_hardcase_duration"))
        signal_length = int(st.number_input("Signal length override (0=derived from fs*duration)", min_value=0, max_value=16384, value=0, step=128, key="stat_hardcase_signal_length"))
        num_components_range = st.slider("Number of components range", min_value=1, max_value=5, value=(3, 5), step=1, key="stat_hardcase_components")
        noise_level_range = st.slider("Noise level range", min_value=0.0, max_value=0.3, value=(0.03, 0.10), step=0.005, key="stat_hardcase_noise")
        amplitude_imbalance_range = st.slider("Amplitude imbalance range", min_value=1.0, max_value=20.0, value=(2.0, 8.0), step=0.5, key="stat_hardcase_imbalance")
        weak_component_probability = float(st.slider("Weak component probability", min_value=0.0, max_value=1.0, value=0.30, step=0.05, key="stat_hardcase_weak_prob"))
        candidate_pool = int(st.slider("Candidate search per signal", min_value=1, max_value=12, value=int(DIFFICULTY_PRESETS[preset_name]["candidate_pool"]), step=1, key="stat_hardcase_candidate_pool"))
        use_cache = st.checkbox("Cache generated signals", value=True, key="stat_hardcase_cache")
        compare_only_selected = st.checkbox("Gracefully continue if one model fails", value=True, key="stat_hardcase_continue_on_error")
        component_type_weights = {}
        st.caption("Component type mixture")
        for component_type in SUPPORTED_COMPONENT_TYPES:
            component_type_weights[component_type] = float(
                st.slider(
                    f"{component_type} weight",
                    min_value=0.0,
                    max_value=5.0,
                    value=1.0 if component_type in {"harmonic", "amfm", "chirp"} else 0.5,
                    step=0.25,
                    key=f"stat_hardcase_weight_{component_type}",
                )
            )
        hard_corr_threshold = st.slider("Hard-case macro correlation threshold", min_value=-1.0, max_value=1.0, value=0.60, step=0.05, key="stat_hardcase_corr_thresh")
        hard_snr_threshold = st.slider("Hard-case macro SNR threshold", min_value=-20.0, max_value=20.0, value=0.0, step=0.5, key="stat_hardcase_snr_thresh")
        dnn_gap_threshold = st.slider("Hard-case SNN-DNN gap threshold", min_value=0.0, max_value=1.0, value=0.10, step=0.01, key="stat_hardcase_gap_thresh")

    selected_models = _selected_model_specs_for_statistical(registry, model_preset, custom_models)
    if not selected_models:
        st.info("Select at least one model.")
        return

    config = {
        "model_preset": model_preset,
        "selected_models": [item[1].get("key", item[1].get("model_key", "")) for item in selected_models],
        "preset_name": preset_name,
        "num_signals": num_signals,
        "random_seed": random_seed,
        "fs": fs,
        "duration": duration,
        "signal_length": signal_length,
        "num_components_range": num_components_range,
        "noise_level_range": noise_level_range,
        "amplitude_imbalance_range": amplitude_imbalance_range,
        "weak_component_probability": weak_component_probability,
        "candidate_pool": candidate_pool,
        "component_types": SUPPORTED_COMPONENT_TYPES,
        "component_type_weights": component_type_weights,
        "continue_on_error": compare_only_selected,
        "thresholds": {"macro_corr": hard_corr_threshold, "macro_snr": hard_snr_threshold, "snn_dnn_gap": dnn_gap_threshold},
    }
    cache_key = json.dumps(config, sort_keys=True)

    run_clicked = st.button("Run statistical hard-case testing", type="primary", key="stat_hardcase_run")
    if run_clicked:
        model_entries = []
        for display_name, model_spec, checkpoint in selected_models:
            try:
                model, load_msg = load_model(
                    model_spec.get("model_key", model_spec["key"]),
                    out_channels=max(1, num_components_range[1]),
                    checkpoint_path=checkpoint,
                )
                probe_length = signal_length if signal_length > 0 else max(128, int(round(duration * fs)))
                probe_prediction, _ = run_inference(model, np.zeros(probe_length, dtype=np.float32))
                model_entries.append(
                    {
                        "display_name": display_name,
                        "model_name": model_spec.get("key", model_spec.get("model_key", "")),
                        "model_type": _infer_model_type(model_spec),
                        "family": model_spec.get("family", ""),
                        "checkpoint": checkpoint,
                        "model": model,
                        "load_msg": load_msg,
                        "parameter_count": count_parameters(model),
                        "supported_out_channels": int(probe_prediction.shape[0]),
                    }
                )
            except Exception as exc:
                st.error(f"Failed to load {display_name}: {exc}")
        if not model_entries:
            st.error("No selected models could be loaded.")
            return

        supported_counts = {entry["supported_out_channels"] for entry in model_entries}
        if len(supported_counts) != 1:
            st.error(
                "Selected models do not agree on the number of output components. "
                "Choose models trained for the same decomposition setup."
            )
            return
        supported_components = supported_counts.pop()
        if num_components_range != (supported_components, supported_components):
            st.warning(
                f"Selected checkpoints support {supported_components} output components. "
                "The statistical test will use that component count for all generated signals."
            )
        run_config = dict(config)
        run_config["num_components_range"] = (supported_components, supported_components)
        run_cache_key = json.dumps(run_config, sort_keys=True)

        progress = st.progress(0.0)
        status = st.empty()
        warnings: list[str] = []
        artifacts: dict[str, dict[str, Any]] = {}
        signal_rows: list[dict[str, Any]] = []
        model_rows: list[dict[str, Any]] = []
        component_frames: list[pd.DataFrame] = []
        thresholds = build_thresholds()
        cached_signals = st.session_state.get("statistical_hardcase_signal_cache", {})
        signal_cases = cached_signals.get(run_cache_key) if use_cache else None
        if signal_cases is None:
            signal_cases = [
                generate_statistical_signal(index, settings=run_config, base_seed=random_seed)
                for index in range(num_signals)
            ]
            if use_cache:
                cached_signals[run_cache_key] = signal_cases
                st.session_state["statistical_hardcase_signal_cache"] = cached_signals

        for index, signal_case in enumerate(signal_cases):
            signal_rows.append(
                {
                    "signal_id": signal_case["signal_id"],
                    "preset_name": signal_case["preset_name"],
                    "noise_level": signal_case["generator_metadata"]["noise_level"],
                    "num_components": signal_case["generator_metadata"]["n_components"],
                    "spectral_overlap_score": _metric_value(signal_case["descriptors"].get("max_pairwise_spectral_overlap")),
                    "weakest_component_energy_ratio": _metric_value(signal_case["descriptors"].get("weakest_component_ratio")),
                    "amplitude_imbalance": _metric_value(signal_case["descriptors"].get("amplitude_ratio_strongest_to_weakest")),
                    "estimated_input_snr_db": _metric_value(signal_case["descriptors"].get("estimated_input_snr_db")),
                    "component_types": ", ".join(signal_case["component_types"]),
                    "difficulty_score": signal_case["difficulty_score"],
                    "difficulty_text": signal_case["difficulty_text"],
                }
            )
            artifact_entry = {
                "signal_id": signal_case["signal_id"],
                "time_axis": signal_case["time_axis"],
                "mixture": signal_case["mixture"],
                "components": signal_case["components"],
                "component_names": signal_case["component_names"],
                "component_types": signal_case["component_types"],
                "generator_metadata": signal_case["generator_metadata"],
                "descriptors": signal_case["descriptors"],
                "difficulty_score": signal_case["difficulty_score"],
                "difficulty_text": signal_case["difficulty_text"],
                "model_predictions": {},
            }
            direct_compare_payloads = []
            for model_entry in model_entries:
                try:
                    prediction, infer_ms = run_inference(model_entry["model"], signal_case["mixture"])
                    if prediction.shape != signal_case["components"].shape:
                        raise ValueError(f"prediction shape {prediction.shape} does not match target shape {signal_case['components'].shape}")
                    report = evaluate_prediction(signal_case["components"], prediction, signal_case["mixture"], permutation_invariant=True)
                    model_row, component_rows = evaluate_model_on_signal(
                        signal_case=signal_case,
                        model_name=model_entry["model_name"],
                        display_name=model_entry["display_name"],
                        model_type=model_entry["model_type"],
                        family=model_entry["family"],
                        checkpoint=model_entry["checkpoint"],
                        parameter_count=model_entry["parameter_count"],
                        inference_time_ms=infer_ms,
                        prediction=prediction,
                        report=report,
                        thresholds=thresholds,
                    )
                    model_rows.append(model_row)
                    component_frames.append(component_rows)
                    direct_compare_payloads.append(
                        {
                            "display_name": model_entry["display_name"],
                            "model_type": model_entry["model_type"],
                            "macro_corr": model_row["macro_corr"],
                            "macro_snr": model_row["macro_snr"],
                            "component_table": component_rows,
                        }
                    )
                    artifact_entry["model_predictions"][model_entry["model_name"]] = {
                        **model_row,
                        "prediction": report.get("aligned_prediction", prediction),
                        "component_table": component_rows.to_dict("records"),
                    }
                except Exception as exc:
                    warnings.append(f"{signal_case['signal_id']} | {model_entry['display_name']} failed: {exc}")
                    if not compare_only_selected:
                        st.error(warnings[-1])
                        return
            artifacts[signal_case["signal_id"]] = artifact_entry
            progress.progress((index + 1) / max(len(signal_cases), 1))
            status.caption(f"Processed {index + 1}/{len(signal_cases)} difficult signals.")

        signal_table = pd.DataFrame(signal_rows)
        model_table = pd.DataFrame(model_rows)
        component_table = pd.concat(component_frames, ignore_index=True, sort=False) if component_frames else pd.DataFrame()
        signal_comparison = compute_signal_level_comparison(model_table)
        hard_case_table = build_hard_case_table(model_table, signal_comparison)
        summary = summarize_statistical_results(signal_table, model_table, component_table, signal_comparison)
        result = StatisticalRunResult(
            signal_table=signal_table,
            model_table=model_table,
            component_table=component_table,
            signal_comparison_table=signal_comparison,
            hard_case_table=hard_case_table,
            artifacts=artifacts,
            summary=summary,
            warnings=warnings,
        )
        st.session_state["statistical_hardcase_result"] = result
        st.session_state["statistical_hardcase_config"] = config

    result: StatisticalRunResult | None = st.session_state.get("statistical_hardcase_result")
    if result is None:
        st.info("Configure a model set and difficulty settings, then run the statistical hard-case test.")
        return
    if result.model_table.empty or "signal_id" not in result.model_table.columns:
        st.error("The statistical run completed, but no model evaluations were recorded successfully.")
        if result.warnings:
            st.subheader("Run Warnings")
            st.dataframe(pd.DataFrame({"warning": result.warnings}), use_container_width=True, hide_index=True)
        else:
            st.info("No warnings were captured. This usually means the selected models failed before metrics were recorded.")
        return

    figures = _build_statistical_figures(result.model_table, result.component_table, result.signal_comparison_table)
    summary = result.summary
    row = st.columns(5)
    row[0].metric("Signals", str(summary.get("num_signals", 0)))
    row[1].metric("Evaluations", str(summary.get("num_model_evaluations", 0)))
    row[2].metric("SNN failure rate", "n/a" if summary.get("snn_failure_rate") is None else f"{summary['snn_failure_rate']:.3f}")
    row[3].metric("DNN failure rate", "n/a" if summary.get("dnn_failure_rate") is None else f"{summary['dnn_failure_rate']:.3f}")
    row[4].metric("DNN beats SNN", "n/a" if summary.get("dnn_beats_snn_rate") is None else f"{summary['dnn_beats_snn_rate']:.3f}")
    st.write(summary.get("overall_text", ""))

    main_tabs = st.tabs(
        [
            "Summary Results",
            "Charts",
            "Hard-Case Table",
            "Detailed Inspection",
            "Export",
        ]
    )

    with main_tabs[0]:
        st.subheader("Summary Results")
        summary_rows = [{"statistic": key, "value": value} for key, value in summary.items() if key != "overall_text"]
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
        if result.warnings:
            with st.expander("Warnings", expanded=False):
                for warning in result.warnings:
                    st.warning(warning)

    with main_tabs[1]:
        chart_groups = {
            "Overall performance": [
                "macro_corr_box_by_model",
                "macro_snr_box_by_model",
                "macro_corr_violin_by_type",
                "hard_case_histogram",
                "failure_rate_per_model",
                "hard_case_score_per_model",
                "robustness_ranking",
            ],
            "SNN vs DNN": [
                "best_snn_vs_best_dnn_corr",
                "best_snn_vs_best_dnn_snr",
                "gap_vs_overlap",
                "gap_vs_noise",
                "competitiveness",
            ],
            "Signal conditions": [
                "corr_vs_noise",
                "snr_vs_noise",
                "corr_vs_overlap",
                "snr_vs_overlap",
                "corr_vs_weakest_ratio",
                "snr_vs_weakest_ratio",
                "corr_vs_imbalance",
                "corr_vs_num_components",
            ],
            "Component-level": [
                "component_failure_rate",
                "component_average_corr",
                "component_average_snr",
                "component_spectral_mismatch",
            ],
            "Failure explanations": [
                "failure_labels_snn_vs_dnn",
                "failure_labels_snn",
                "failure_labels_dnn",
                "failure_vs_overlap_heatmap",
                "failure_vs_noise_heatmap",
            ],
            "Efficiency context": [
                "corr_vs_inference_time",
                "failure_vs_parameters",
            ],
            "Thesis summary": [
                "where_snns_fail",
            ],
        }
        for group_name, keys in chart_groups.items():
            available = [key for key in keys if key in figures]
            if not available:
                continue
            st.subheader(group_name)
            for key in available:
                st.pyplot(figures[key], clear_figure=False, use_container_width=True)

    with main_tabs[2]:
        st.subheader("Hard-Case Table")
        sort_by = st.selectbox(
            "Sort by",
            [
                "hard_case_score",
                "macro_corr_gap",
                "spectral_overlap_score",
                "noise_level",
                "weakest_component_energy_ratio",
            ],
            index=0,
            key="stat_hardcase_sort_by",
        )
        model_filter = st.multiselect("Model filter", sorted(result.hard_case_table["display_name"].dropna().astype(str).unique().tolist()), default=[], key="stat_hardcase_model_filter")
        label_filter = st.text_input("Failure label filter", value="", key="stat_hardcase_label_filter")
        table = result.hard_case_table.copy()
        if model_filter:
            table = table[table["display_name"].isin(model_filter)]
        if label_filter:
            table = table[table["failure_labels"].astype(str).str.contains(label_filter, case=False, na=False)]
        ascending = sort_by in {"macro_corr", "weakest_component_energy_ratio"}
        if sort_by in table.columns:
            table = table.sort_values(sort_by, ascending=ascending, na_position="last")
        st.dataframe(table, use_container_width=True, hide_index=True)

    with main_tabs[3]:
        st.subheader("Detailed Hard-Case Inspection")
        signal_ids = result.hard_case_table["signal_id"].dropna().astype(str).unique().tolist()
        if not signal_ids:
            st.info("No hard-case signals available.")
        else:
            signal_id = st.selectbox("Signal ID", signal_ids, key="stat_hardcase_signal_id")
            artifact = result.artifacts.get(signal_id, {})
            model_rows = result.model_table[result.model_table["signal_id"] == signal_id]
            model_display = st.selectbox("Model", model_rows["display_name"].tolist(), key="stat_hardcase_model_pick")
            selected_model = model_rows[model_rows["display_name"] == model_display].iloc[0]
            prediction_payload = artifact.get("model_predictions", {}).get(selected_model["model_name"], {})
            comparison = compare_direct_model_results(
                [
                    {
                        "display_name": row["display_name"],
                        "model_type": row["model_type"],
                        "macro_corr": row["macro_corr"],
                        "macro_snr": row["macro_snr"],
                        "component_table": pd.DataFrame(artifact.get("model_predictions", {}).get(row["model_name"], {}).get("component_table", [])),
                    }
                    for _, row in model_rows.iterrows()
                ]
            )
            preview = figure_prediction_preview(
                np.asarray(artifact.get("time_axis"), dtype=float),
                np.asarray(artifact.get("mixture"), dtype=float),
                np.asarray(artifact.get("components"), dtype=float),
                np.asarray(prediction_payload.get("prediction"), dtype=float),
                artifact.get("component_names"),
                selected_model["display_name"],
            )
            st.pyplot(preview, clear_figure=False, use_container_width=True)
            comp_df = pd.DataFrame(prediction_payload.get("component_table", []))
            if not comp_df.empty:
                st.dataframe(comp_df, use_container_width=True, hide_index=True)
            st.write(selected_model.get("explanation", ""))
            if comparison.get("component_comparison", pd.DataFrame()).empty is False:
                st.caption("Best SNN vs best DNN comparison on this signal")
                st.dataframe(comparison["component_comparison"], use_container_width=True, hide_index=True)
            st.caption("Signal descriptors")
            descriptor_rows = [{"descriptor": key, "value": value} for key, value in artifact.get("descriptors", {}).items() if key not in {"component_correlation_matrix", "pairwise_spectral_overlap", "component_descriptors"}]
            st.dataframe(pd.DataFrame(descriptor_rows), use_container_width=True, hide_index=True)

    with main_tabs[4]:
        st.subheader("Export")
        st.download_button(
            "Download full results CSV",
            result.model_table.to_csv(index=False).encode("utf-8"),
            file_name="statistical_hardcase_model_results.csv",
            mime="text/csv",
        )
        st.download_button(
            "Download hard-case table CSV",
            result.hard_case_table.to_csv(index=False).encode("utf-8"),
            file_name="statistical_hardcase_hardcases.csv",
            mime="text/csv",
        )
        st.download_button(
            "Download statistical summary JSON",
            json.dumps(summary, indent=2).encode("utf-8"),
            file_name="statistical_hardcase_summary.json",
            mime="application/json",
        )
        if st.button("Export all charts, tables, config, and artifacts", key="stat_hardcase_export_bundle"):
            bundle = export_statistical_bundle(
                result=result,
                config=st.session_state.get("statistical_hardcase_config", {}),
                figures=figures,
            )
            st.success(f"Saved bundle to `{bundle.relative_to(REPO_ROOT)}`")


def _render_signal_diagnostics(registry: list[dict]) -> None:
    st.caption("Generate or load a signal, run one or more models, and inspect why the case is easy or hard.")
    enabled_models = [item for item in registry if item.get("enabled", True)]
    if not enabled_models:
        st.warning("No enabled models are available.")
        return

    model_map = {item["display_name"]: item for item in enabled_models}
    defaults = [name for name, spec in model_map.items() if spec.get("depth_label") == "shallow"][:1]
    defaults += [name for name, spec in model_map.items() if spec.get("depth_label") == "deep"][:1]
    defaults = list(dict.fromkeys(defaults)) or list(model_map.keys())[:1]

    with st.sidebar:
        st.subheader("Diagnostic Thresholds")
        threshold_values = {
            "corr_shape_failure": st.slider("Shape failure corr threshold", min_value=-1.0, max_value=1.0, value=0.45, step=0.05),
            "cosine_shape_failure": st.slider("Shape failure cosine threshold", min_value=-1.0, max_value=1.0, value=0.50, step=0.05),
            "corr_amplitude_ok": st.slider("Amplitude mismatch corr threshold", min_value=-1.0, max_value=1.0, value=0.75, step=0.05),
            "mse_amplitude_mismatch": st.number_input("Amplitude mismatch MSE", min_value=0.0, max_value=10.0, value=0.08, step=0.01),
            "mae_high_error": st.number_input("High MAE threshold", min_value=0.0, max_value=10.0, value=0.18, step=0.01),
            "rmse_high_error": st.number_input("High RMSE threshold", min_value=0.0, max_value=10.0, value=0.28, step=0.01),
            "max_abs_error_high": st.number_input("High max abs error", min_value=0.0, max_value=10.0, value=0.70, step=0.05),
            "energy_ratio_low": st.number_input("Suppression energy ratio", min_value=0.0, max_value=5.0, value=0.60, step=0.05),
            "energy_ratio_high": st.number_input("Overprediction energy ratio", min_value=0.0, max_value=10.0, value=1.50, step=0.05),
            "snr_poor": st.number_input("Poor SNR threshold", min_value=-40.0, max_value=40.0, value=0.0, step=0.5),
            "si_sdr_poor": st.number_input("Poor SI-SDR threshold", min_value=-40.0, max_value=40.0, value=0.0, step=0.5),
            "explained_variance_bad": st.number_input("Bad explained variance", min_value=-5.0, max_value=1.0, value=0.0, step=0.05),
            "spectral_convergence_high": st.number_input("High spectral convergence", min_value=0.0, max_value=5.0, value=0.60, step=0.05),
            "log_spectral_distance_high": st.number_input("High log spectral distance", min_value=0.0, max_value=10.0, value=1.00, step=0.05),
            "fft_magnitude_l1_high": st.number_input("High FFT magnitude L1", min_value=0.0, max_value=50.0, value=3.0, step=0.25),
            "dominant_error_ratio": st.number_input("Dominant error ratio", min_value=1.0, max_value=10.0, value=1.20, step=0.05),
        }
    thresholds = build_thresholds(threshold_values)

    setup_tab, results_tab = st.tabs(["Signal Setup", "Diagnostics"])

    with setup_tab:
        source_mode = st.radio("Signal source", ["Generate synthetic signal", "Upload NPZ signal"], horizontal=True)
        generated_signal = None
        signal_payload = None

        if source_mode == "Generate synthetic signal":
            mode_label = st.radio("Signal mode", ["Non-pure preset", "Pure manual components"], horizontal=True)
            generation_mode = PURE_MODE if mode_label.startswith("Pure") else "preset"
            signal_type = "mixed"
            selected_component_types: list[str] = []
            if generation_mode == PURE_MODE:
                component_cols = st.columns(len(SUPPORTED_COMPONENT_TYPES))
                for idx, component_type in enumerate(SUPPORTED_COMPONENT_TYPES):
                    if component_cols[idx].checkbox(
                        component_type,
                        value=component_type in ["harmonic", "amfm", "chirp"],
                        key=f"diag_component_{component_type}",
                    ):
                        selected_component_types.append(component_type)
                n_components = len(selected_component_types)
            else:
                col1, col2 = st.columns(2)
                signal_type = col1.selectbox("Generator preset", SUPPORTED_PRESETS, index=0)
                n_components = int(col2.slider("Number of components", min_value=1, max_value=5, value=3, step=1))

            fs = int(st.number_input("Sampling rate", min_value=64, max_value=4096, value=256, step=64, key="diag_fs"))
            duration = float(st.slider("Duration (s)", min_value=0.5, max_value=8.0, value=4.0, step=0.1, key="diag_duration"))
            noise_level = float(st.slider("Noise level std", min_value=0.0, max_value=0.3, value=0.04, step=0.005, key="diag_noise"))
            seed = int(st.number_input("Seed", min_value=0, max_value=99999999, value=1234, step=1, key="diag_seed"))
            if generation_mode == PURE_MODE:
                valid_components, component_msg = validate_pure_component_types(selected_component_types)
                if not valid_components:
                    st.warning(component_msg)
            else:
                valid_components = True

            if st.button("Generate test signal", type="primary", disabled=not valid_components):
                config = SignalConfig(
                    signal_type=signal_type,
                    n_components=n_components,
                    duration=duration,
                    fs=fs,
                    noise_level=noise_level,
                    seed=seed,
                    generation_mode=generation_mode,
                    selected_component_types=selected_component_types,
                )
                generated_signal = generate_signal(config)
                signal_payload = {
                    "source": "generated",
                    "mixture": generated_signal.mixture,
                    "components": generated_signal.components,
                    "time_axis": generated_signal.t,
                    "fs": fs,
                    "component_names": generated_signal.component_names,
                    "config": {
                        "signal_type": signal_type,
                        "n_components": n_components,
                        "duration": duration,
                        "fs": fs,
                        "noise_level": noise_level,
                        "seed": seed,
                        "generation_mode": generation_mode,
                        "selected_component_types": selected_component_types if generation_mode == PURE_MODE else [],
                    },
                }
                st.session_state["test_signal_payload"] = signal_payload
        else:
            uploaded = st.file_uploader("Upload signal artifact", type=["npz"], help="Use NPZ with at least `mixture` and optionally `components`, `t`/`time_axis`, `fs`, and `component_names`.")
            out_channels = int(st.number_input("Fallback output components", min_value=1, max_value=16, value=3, step=1, help="Used only when the uploaded file has no true components."))
            if uploaded is not None:
                signal_payload, error = _load_uploaded_signal(uploaded)
                if error:
                    st.error(error)
                elif signal_payload is not None:
                    if signal_payload["components"] is None:
                        signal_payload["component_names"] = signal_payload["component_names"] or [f"component_{idx + 1}" for idx in range(out_channels)]
                    signal_payload["config"] = {"uploaded_file": uploaded.name, "fallback_out_channels": out_channels}
                    st.session_state["test_signal_payload"] = signal_payload
                    st.success(f"Loaded signal from {uploaded.name}")

        current_signal = st.session_state.get("test_signal_payload")
        if current_signal:
            st.subheader("Model Selection")
            st.caption("Only checkpoints with `train_samples = 8000` are available in this tool.")
            selected_names = st.multiselect("Models", list(model_map.keys()), default=defaults, key="diag_models")
            configure_checkpoints = st.checkbox("Choose checkpoints per model", value=False, key="diag_ckpt_toggle")
            selected_models = []
            unavailable_models = []
            for name in selected_names:
                model_spec = model_map[name]
                checkpoints = _available_checkpoints(model_spec)
                if not checkpoints:
                    unavailable_models.append(name)
                    continue
                checkpoint = checkpoints[0]
                if configure_checkpoints:
                    checkpoint = st.selectbox(f"{name} checkpoint", checkpoints, key=f"diag_ckpt_{model_spec['key']}")
                selected_models.append((name, model_spec, checkpoint))
            if unavailable_models:
                st.warning(
                    "Skipped models without a train_samples=8000 checkpoint: "
                    + ", ".join(unavailable_models)
                )

            if st.button("Run diagnostics", type="primary", disabled=not selected_models):
                results: list[dict] = []
                components = current_signal.get("components")
                component_names = current_signal.get("component_names", [])
                out_channels = components.shape[0] if isinstance(components, np.ndarray) and components.ndim == 2 else max(len(component_names), 1)
                for display_name, model_spec, checkpoint in selected_models:
                    try:
                        model, load_msg = load_model(
                            model_spec.get("model_key", model_spec["key"]),
                            out_channels=out_channels,
                            checkpoint_path=checkpoint,
                        )
                        prediction, elapsed_ms = run_inference(model, current_signal["mixture"])
                        report = None
                        component_table = pd.DataFrame()
                        macro_metrics = {}
                        clean_metrics = {}
                        if isinstance(components, np.ndarray) and prediction.shape == components.shape:
                            report = evaluate_prediction(
                                components,
                                prediction,
                                current_signal["mixture"],
                                permutation_invariant=True,
                            )
                            prediction = report["aligned_prediction"]
                            component_table = compute_component_diagnostics(
                                report,
                                component_names,
                                y_true=components,
                                y_pred=prediction,
                                fs=current_signal.get("fs"),
                                thresholds=thresholds,
                            )
                            macro_metrics = report.get("macro_average", {}) or {}
                            clean_metrics = report.get("clean_sum_metrics", {}) or {}
                        difficulty_summary = compute_signal_difficulty_descriptors(
                            current_signal["mixture"],
                            components,
                            prediction,
                            component_names,
                            fs=current_signal.get("fs"),
                        )
                        explanation = build_explanation_text(
                            component_table,
                            macro_metrics=macro_metrics,
                            clean_metrics=clean_metrics,
                            difficulty_summary=difficulty_summary,
                        )
                        results.append(
                            {
                                "model_name": model_spec.get("key", model_spec.get("model_key", "")),
                                "display_name": display_name,
                                "model_type": _infer_model_type(model_spec),
                                "family": model_spec.get("family", ""),
                                "checkpoint": checkpoint,
                                "load_msg": load_msg,
                                "prediction": prediction,
                                "elapsed_ms": elapsed_ms,
                                "parameter_count": count_parameters(model),
                                "report": report or {},
                                "component_table": component_table,
                                "macro_metrics": macro_metrics,
                                "clean_sum_metrics": clean_metrics,
                                "difficulty_summary": difficulty_summary,
                                "signal_difficulty_text": build_signal_difficulty_text(difficulty_summary),
                                "explanation": explanation,
                                "component_names": component_names,
                            }
                        )
                    except Exception as exc:
                        st.error(f"{display_name} failed: {exc}")
                st.session_state["test_signal_results"] = results
                st.session_state["test_signal_thresholds"] = thresholds_to_dict(thresholds)

    with results_tab:
        current_signal = st.session_state.get("test_signal_payload")
        results = st.session_state.get("test_signal_results", [])
        if not current_signal:
            st.info("Generate or upload a signal first.")
            return
        if not results:
            st.info("Run diagnostics on at least one model to inspect interpretations.")
            return
        comparison = compare_model_results(results)

        overview_rows = []
        for item in results:
            overview_rows.append(
                {
                    "model": item["display_name"],
                    "model_type": item["model_type"],
                    "macro_corr": _metric_value(item["macro_metrics"].get("corr")),
                    "macro_snr_db": _metric_value(item["macro_metrics"].get("snr_db")),
                    "test_latency_ms": item["elapsed_ms"],
                    "parameter_count": item["parameter_count"],
                    "checkpoint": item["checkpoint"],
                }
            )
        st.subheader("Overall Metrics")
        st.dataframe(pd.DataFrame(overview_rows), use_container_width=True, hide_index=True)

        inspected_name = st.selectbox("Inspect model result", [item["display_name"] for item in results], key="diag_inspect_model")
        inspected = next(item for item in results if item["display_name"] == inspected_name)
        preview_fig = figure_prediction_preview(
            np.asarray(current_signal["time_axis"], dtype=float),
            np.asarray(current_signal["mixture"], dtype=float),
            current_signal.get("components"),
            np.asarray(inspected["prediction"], dtype=float),
            inspected.get("component_names"),
            inspected["display_name"],
        )

        section_tabs = st.tabs(
            [
                "Prediction Preview",
                "Per-Component Metrics",
                "Hard-Case Interpretation",
                "Signal Difficulty Analysis",
                "SNN vs DNN Comparison",
                "Save / Export",
            ]
        )

        with section_tabs[0]:
            st.subheader("Prediction Preview")
            st.pyplot(preview_fig, clear_figure=False, use_container_width=True)
            st.download_button(
                "Download preview figure",
                _figure_download_bytes(preview_fig),
                file_name=f"{inspected['model_name']}_prediction_preview.png",
                mime="image/png",
            )
            st.caption(inspected.get("load_msg", ""))

        with section_tabs[1]:
            st.subheader("Per-Component Metrics")
            component_table = inspected.get("component_table", pd.DataFrame())
            if component_table.empty:
                st.info("No true component targets were available, so per-component metrics could not be computed.")
            else:
                st.dataframe(component_table, use_container_width=True, hide_index=True)
                st.download_button(
                    "Download per-component metrics CSV",
                    component_table.to_csv(index=False).encode("utf-8"),
                    file_name=f"{inspected['model_name']}_component_metrics.csv",
                    mime="text/csv",
                )

        with section_tabs[2]:
            st.subheader("Hard-Case Interpretation")
            st.write(inspected.get("explanation", "No explanation available."))
            labels_df = inspected.get("component_table", pd.DataFrame())
            if not labels_df.empty:
                st.caption("Failure labels")
                st.dataframe(labels_df[["component", "failure_label_text", "primary_failure_mode"]], use_container_width=True, hide_index=True)
            st.caption("Signal-level interpretation")
            st.write(inspected.get("signal_difficulty_text", ""))

        with section_tabs[3]:
            st.subheader("Signal Difficulty Analysis")
            difficulty_summary = inspected.get("difficulty_summary", {})
            summary_rows = [{"descriptor": key, "value": value} for key, value in difficulty_summary.items() if key not in {"component_correlation_matrix", "pairwise_spectral_overlap", "component_descriptors"}]
            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
            component_descriptors = difficulty_summary.get("component_descriptors", [])
            if component_descriptors:
                st.caption("Per-component signal descriptors")
                st.dataframe(pd.DataFrame(component_descriptors), use_container_width=True, hide_index=True)
            overlaps = difficulty_summary.get("pairwise_spectral_overlap", [])
            if overlaps:
                st.caption("Pairwise spectral overlap")
                st.dataframe(pd.DataFrame(overlaps), use_container_width=True, hide_index=True)
            corr_matrix = difficulty_summary.get("component_correlation_matrix")
            if corr_matrix:
                names = inspected.get("component_names") or [f"component_{idx + 1}" for idx in range(len(corr_matrix))]
                st.caption("Component correlation matrix")
                st.dataframe(pd.DataFrame(corr_matrix, index=names, columns=names), use_container_width=True)

        with section_tabs[4]:
            st.subheader("SNN vs DNN Comparison")
            best_snn = comparison.get("best_snn")
            best_dnn = comparison.get("best_dnn")
            if best_snn is None or best_dnn is None:
                st.info("Select at least one SNN and one DNN to enable the comparison view.")
            else:
                cols = st.columns(4)
                cols[0].metric("Best SNN", best_snn["display_name"])
                cols[1].metric("Best DNN", best_dnn["display_name"])
                cols[2].metric("Macro corr gap", f"{comparison.get('macro_corr_gap', float('nan')):.3f}")
                cols[3].metric("Macro SNR gap", f"{comparison.get('macro_snr_gap', float('nan')):.3f}")
                comp_df = comparison.get("component_comparison", pd.DataFrame())
                if not comp_df.empty:
                    st.dataframe(comp_df, use_container_width=True, hide_index=True)
                    for metric_spec in [
                        ("corr_snn", "corr_dnn", "SNN vs DNN per-component correlation", "Correlation"),
                        ("snr_snn", "snr_dnn", "SNN vs DNN per-component SNR", "SNR (dB)"),
                        ("residual_energy_ratio_snn", "residual_energy_ratio_dnn", "SNN vs DNN residual energy", "Residual energy ratio"),
                        ("spectral_error_snn", "spectral_error_dnn", "SNN vs DNN spectral error", "Spectral convergence"),
                    ]:
                        fig = figure_component_metric_comparison(
                            comp_df,
                            metric_snn=metric_spec[0],
                            metric_dnn=metric_spec[1],
                            title=metric_spec[2],
                            ylabel=metric_spec[3],
                        )
                        if fig is not None:
                            st.pyplot(fig, clear_figure=False, use_container_width=True)
                    heatmap = figure_failure_label_heatmap(comp_df)
                    if heatmap is not None:
                        st.pyplot(heatmap, clear_figure=False, use_container_width=True)

        with section_tabs[5]:
            st.subheader("Save / Export")
            explanation_text = inspected.get("explanation", "")
            st.download_button(
                "Download explanation text",
                explanation_text.encode("utf-8"),
                file_name=f"{inspected['model_name']}_explanation.txt",
                mime="text/plain",
            )
            failure_labels = inspected.get("component_table", pd.DataFrame())
            if not failure_labels.empty:
                st.download_button(
                    "Download failure labels JSON",
                    json.dumps(failure_labels[["component", "failure_label_text"]].to_dict("records"), indent=2).encode("utf-8"),
                    file_name=f"{inspected['model_name']}_failure_labels.json",
                    mime="application/json",
                )
            artifact_name = f"diagnostic_{inspected['model_name']}_{current_signal.get('source', 'signal')}"
            if st.button("Save hard-case artifact", type="primary"):
                metadata_path, arrays_path = save_diagnostic_artifact(
                    base_name=artifact_name,
                    time_axis=np.asarray(current_signal["time_axis"], dtype=float),
                    mixture=np.asarray(current_signal["mixture"], dtype=float),
                    components=current_signal.get("components"),
                    model_result=inspected,
                    signal_settings=current_signal.get("config", {}),
                    comparison_summary={
                        "best_snn": (comparison.get("best_snn") or {}).get("display_name"),
                        "best_dnn": (comparison.get("best_dnn") or {}).get("display_name"),
                        "macro_corr_gap": comparison.get("macro_corr_gap"),
                        "macro_snr_gap": comparison.get("macro_snr_gap"),
                    },
                )
                st.success(
                    "Saved artifact: "
                    f"`{metadata_path.relative_to(REPO_ROOT)}`, "
                    f"`{arrays_path.relative_to(REPO_ROOT)}`"
                )


def _render_hard_signal_search(registry: list[dict]) -> None:
    st.caption("Randomly search synthetic signals and surface failure cases for decomposition models.")

    enabled_models = [m for m in registry if m.get("enabled", True)]
    if not enabled_models:
        st.error("No enabled models in registry.")
        return

    model_map = {m["display_name"]: m for m in enabled_models}
    shallow_defaults = [m["display_name"] for m in enabled_models if m.get("depth_label") == "shallow"]

    with st.sidebar:
        st.subheader("Model + Search")
        st.caption("Only checkpoints with `train_samples = 8000` are available in this tool.")
        selected_names = st.multiselect(
            "Models",
            list(model_map.keys()),
            default=shallow_defaults[: min(2, len(shallow_defaults))] or list(model_map.keys())[:1],
        )
        configure_checkpoints = st.checkbox("Choose checkpoints per model", value=False)
        selected_models = []
        unavailable_models = []
        for name in selected_names:
            model_spec = model_map[name]
            checkpoints = _available_checkpoints(model_spec)
            if not checkpoints:
                unavailable_models.append(name)
                continue
            checkpoint = checkpoints[0]
            if configure_checkpoints:
                checkpoint = st.selectbox(f"{name} checkpoint", checkpoints, key=f"hard_mining_ckpt_{model_spec['key']}")
            selected_models.append((name, model_spec, checkpoint))
        if unavailable_models:
            st.warning(
                "Skipped models without a train_samples=8000 checkpoint: "
                + ", ".join(unavailable_models)
            )

        candidate_count = st.number_input("Candidate signals", min_value=1, max_value=5000, value=100, step=10)
        base_seed = st.number_input("Base seed", min_value=0, max_value=99999999, value=1234, step=1)
        progress_chunk = st.number_input("Progress update chunk", min_value=1, max_value=500, value=10, step=1)
        top_k = st.slider("Keep top-k hard cases", min_value=1, max_value=20, value=5, step=1)

        st.subheader("Signal Generator")
        generation_label = st.radio("Component mode", ["Non-pure preset", "Pure manual components"])
        generation_mode = PURE_MODE if generation_label.startswith("Pure") else "preset"
        signal_type = "mixed"
        selected_component_types: list[str] = []
        if generation_mode == PURE_MODE:
            for component_type in SUPPORTED_COMPONENT_TYPES:
                if st.checkbox(component_type, value=component_type in ["harmonic", "amfm", "chirp"], key=f"hard_component_{component_type}"):
                    selected_component_types.append(component_type)
            n_components = len(selected_component_types)
        else:
            signal_type = st.selectbox("Generator preset", SUPPORTED_PRESETS, index=0)
            n_components = st.slider("Number of components", min_value=1, max_value=5, value=3, step=1)

        fs = st.number_input("Sampling rate", min_value=64, max_value=4096, value=256, step=64)
        duration = st.slider("Duration (s)", min_value=0.5, max_value=8.0, value=4.0, step=0.1)
        noise_level = st.slider("Noise level std", min_value=0.0, max_value=0.3, value=0.04, step=0.005)
        permutation_invariant = st.checkbox("Permutation-invariant scoring", value=True)

        with st.expander("Hardness score weights", expanded=False):
            weights = {
                "reconstruction_nmse": st.number_input("Reconstruction NMSE", min_value=0.0, max_value=10.0, value=0.5, step=0.1),
                "mean_component_nmse": st.number_input("Mean component NMSE", min_value=0.0, max_value=10.0, value=1.0, step=0.1),
                "worst_component_nmse": st.number_input("Worst component NMSE", min_value=0.0, max_value=10.0, value=1.0, step=0.1),
                "correlation_disagreement": st.number_input("Correlation disagreement", min_value=0.0, max_value=10.0, value=0.5, step=0.1),
            }

        run_search = st.button("Run hard-signal search", type="primary", use_container_width=True)

    if not run_search and "hard_signal_cases" not in st.session_state:
        st.info("Configure the search and run hard-signal mining.")
        return

    if run_search:
        if not selected_models:
            st.warning("Select at least one model.")
            return
        if generation_mode == PURE_MODE:
            valid, msg = validate_pure_component_types(selected_component_types)
            if not valid:
                st.warning(msg)
                return

        component_count = n_components
        loaded_models = []
        with st.spinner("Loading selected models..."):
            for display_name, model_spec, checkpoint in selected_models:
                try:
                    model, load_msg = load_model(
                        model_spec.get("model_key", model_spec["key"]),
                        out_channels=component_count,
                        checkpoint_path=checkpoint,
                    )
                    loaded_models.append((display_name, model_spec, checkpoint, model, load_msg))
                except Exception as exc:
                    st.error(f"Could not load {display_name}: {exc}")

        if not loaded_models:
            st.error("No selected models could be loaded.")
            return

        progress = st.progress(0.0)
        status = st.empty()
        top_cases: list[dict] = []
        failed = []

        for idx in range(int(candidate_count)):
            candidate_seed = int(base_seed) + idx
            config = SignalConfig(
                signal_type=signal_type,
                n_components=component_count,
                duration=float(duration),
                fs=int(fs),
                noise_level=float(noise_level),
                seed=candidate_seed,
                generation_mode=generation_mode,
                selected_component_types=selected_component_types,
            )
            try:
                generated = generate_signal(config)
            except Exception as exc:
                failed.append({"candidate_index": idx, "seed": candidate_seed, "status": f"generation failed: {exc}"})
                continue

            descriptor_summary, descriptor_frame = estimate_descriptors(
                generated.mixture,
                generated.components,
                generated.component_names,
                fs=int(fs),
                duration=float(duration),
            )

            for display_name, model_spec, checkpoint, model, load_msg in loaded_models:
                try:
                    y_pred, elapsed_ms = run_inference(model, generated.mixture)
                    if y_pred.shape != generated.components.shape:
                        raise ValueError(f"prediction shape {y_pred.shape} does not match target shape {generated.components.shape}")
                    report = evaluate_prediction(
                        generated.components,
                        y_pred,
                        generated.mixture,
                        permutation_invariant=permutation_invariant,
                    )
                    aligned_pred = report["aligned_prediction"] if permutation_invariant else y_pred
                    score, terms = compute_hardness(report, weights)
                    case = {
                        "t": generated.t,
                        "mixture": generated.mixture,
                        "components": generated.components,
                        "prediction": aligned_pred,
                        "raw_prediction": y_pred,
                        "component_names": generated.component_names,
                        "model_key": model_spec.get("key", model_spec.get("model_key", "")),
                        "model_constructor_key": model_spec.get("model_key", model_spec.get("key", "")),
                        "display_name": display_name,
                        "checkpoint": checkpoint,
                        "load_msg": load_msg,
                        "candidate_index": idx,
                        "candidate_seed": candidate_seed,
                        "elapsed_ms": elapsed_ms,
                        "hardness_score": score,
                        "hardness_terms": terms,
                        "report": report,
                        "descriptor_summary": descriptor_summary,
                        "descriptor_frame": descriptor_frame,
                        "generator_config": {
                            "signal_type": signal_type,
                            "n_components": component_count,
                            "duration": float(duration),
                            "fs": int(fs),
                            "noise_level": float(noise_level),
                            "seed": candidate_seed,
                            "generation_mode": generation_mode,
                            "selected_component_types": selected_component_types if generation_mode == PURE_MODE else [],
                        },
                    }
                    top_cases.append(case)
                    top_cases = sorted(top_cases, key=lambda item: item["hardness_score"], reverse=True)[: int(top_k)]
                except Exception as exc:
                    failed.append(
                        {
                            "candidate_index": idx,
                            "seed": candidate_seed,
                            "model": display_name,
                            "status": f"inference/scoring failed: {exc}",
                        }
                    )

            if idx % int(progress_chunk) == 0 or idx == int(candidate_count) - 1:
                progress.progress((idx + 1) / int(candidate_count))
                best = top_cases[0]["hardness_score"] if top_cases else float("nan")
                status.caption(f"Scanned {idx + 1}/{int(candidate_count)} candidates. Current best score: {best:.4f}")

        st.session_state["hard_signal_cases"] = top_cases
        st.session_state["hard_signal_failed"] = failed
        st.session_state["hard_signal_search_settings"] = {
            "candidate_count": int(candidate_count),
            "base_seed": int(base_seed),
            "top_k": int(top_k),
            "models": [item[1].get("key", item[1].get("model_key", "")) for item in selected_models],
            "weights": weights,
            "permutation_invariant": permutation_invariant,
            "generator": {
                "signal_type": signal_type,
                "n_components": component_count,
                "duration": float(duration),
                "fs": int(fs),
                "noise_level": float(noise_level),
                "generation_mode": generation_mode,
                "selected_component_types": selected_component_types if generation_mode == PURE_MODE else [],
            },
        }

    cases: list[dict] = st.session_state.get("hard_signal_cases", [])
    if not cases:
        st.warning("No valid hard cases found.")
        failed = st.session_state.get("hard_signal_failed", [])
        if failed:
            st.dataframe(pd.DataFrame(failed), use_container_width=True, hide_index=True)
        return

    summary_df = pd.DataFrame([_case_summary_row(case) for case in cases])
    st.subheader("Top Hard Cases")
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    labels = [
        f"#{idx + 1} | {case['display_name']} | seed {case['candidate_seed']} | score {case['hardness_score']:.4f}"
        for idx, case in enumerate(cases)
    ]
    selected_label = st.selectbox("Inspect hard case", labels)
    selected_idx = labels.index(selected_label)
    case = cases[selected_idx]

    st.subheader("Hardest Sample Summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Hardness", f"{case['hardness_score']:.4f}")
    col2.metric("Model", case["display_name"])
    col3.metric("Seed", str(case["candidate_seed"]))
    col4.metric("Latency", f"{case['elapsed_ms']:.2f} ms")

    term_cols = st.columns(4)
    for col, (name, value) in zip(term_cols, case["hardness_terms"].items()):
        col.metric(name.replace("_", " ").title(), f"{value:.4f}" if np.isfinite(value) else "N/A")

    fig = _plot_hard_case(case)
    st.pyplot(fig, clear_figure=False)

    st.subheader("Estimated Descriptors")
    descriptor_summary = case["descriptor_summary"]
    desc_cols = st.columns(4)
    desc_cols[0].metric("Estimated SNR (dB)", f"{descriptor_summary.get('estimated_snr_db', np.nan):.3f}")
    desc_cols[1].metric("Energy Ratio", f"{descriptor_summary.get('dominant_to_weakest_energy_ratio', np.nan):.3f}")
    desc_cols[2].metric("Min Freq Distance", f"{descriptor_summary.get('min_pairwise_frequency_distance', np.nan):.3f}")
    desc_cols[3].metric("Max Spectral Overlap", f"{descriptor_summary.get('max_spectral_overlap', np.nan):.3f}")
    st.json(_json_ready(descriptor_summary))
    st.dataframe(case["descriptor_frame"], use_container_width=True, hide_index=True)

    st.subheader("Detailed Metrics")
    component_rows = []
    for idx, metrics in enumerate(case["report"].get("component_metrics", [])):
        row = {"component": case["component_names"][idx] if idx < len(case["component_names"]) else f"C{idx + 1}"}
        row.update(metrics)
        component_rows.append(row)
    if component_rows:
        st.dataframe(pd.DataFrame(component_rows), use_container_width=True, hide_index=True)

    with st.expander("Search failures / skipped cases", expanded=False):
        failed = st.session_state.get("hard_signal_failed", [])
        if failed:
            st.dataframe(pd.DataFrame(failed), use_container_width=True, hide_index=True)
        else:
            st.caption("No failures recorded.")

    csv_bytes = summary_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download top-k summary CSV", csv_bytes, file_name="hard_signal_topk_summary.csv", mime="text/csv")

    if st.button("Save hard case", type="primary"):
        json_path, npz_path, png_path = _save_hard_case(
            case,
            st.session_state.get("hard_signal_search_settings", {}),
            fig,
        )
        st.success(
            "Saved hard case: "
            f"`{json_path.relative_to(REPO_ROOT)}`, "
            f"`{npz_path.relative_to(REPO_ROOT)}`, "
            f"`{png_path.relative_to(REPO_ROOT)}`"
        )


def render(registry: list[dict]) -> None:
    """Streamlit tool for direct signal diagnostics and hard-sample mining."""
    st.title("Test Signal")
    tabs = st.tabs(["Signal Diagnostics", "Statistical Hard-Case Testing", "Hard-Signal Mining"])
    with tabs[0]:
        _render_signal_diagnostics(registry)
    with tabs[1]:
        _render_statistical_hardcase_testing(registry)
    with tabs[2]:
        _render_hard_signal_search(registry)
