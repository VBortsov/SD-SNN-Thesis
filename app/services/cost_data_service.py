from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from app.services.inference_service import evaluate_prediction, load_model, run_inference
from app.services.research_service import (
    ChartSpec,
    DEFAULT_EXPORT_FORMATS,
    METRIC_COLUMNS,
    MODEL_TYPE_COLORS,
    MODEL_TYPE_ORDER,
    ResearchDataset,
    SIGNAL_CONDITION_COLUMNS,
    _style_model_xaxis,
    best_row,
    build_results_tables,
    compute_pareto_frontier,
    prepare_chart_generation_dataframe,
    prepare_research_dataset,
)
from app.services.signal_service import PURE_MODE, SignalConfig, generate_signal

plt.rcParams["figure.max_open_warning"] = 0

COST_RESEARCH_EXPORT_SUBDIR = "cost_data_dashboard"

COST_METRIC_LABELS = {
    **METRIC_COLUMNS,
    "peak_memory_mb": "Peak memory (MB)",
    "train_fraction": "Train fraction",
    "accuracy_per_training_second": "Macro correlation per training second",
    "snr_per_training_second": "Macro SNR per training second",
    "accuracy_per_1000_parameters": "Macro correlation per 1,000 parameters",
    "snr_per_1000_parameters": "Macro SNR per 1,000 parameters",
    "accuracy_per_inference_ms": "Macro correlation per inference millisecond",
    "snr_per_inference_ms": "Macro SNR per inference millisecond",
    "accuracy_per_sample": "Macro correlation per training sample",
}

FRACTION_PLOT_MODEL_ORDER = [
    "mlp_singlehead",
    "conv1dnetwork",
    "multiscale_dilated",
    "multiple_head_multiscale_branches",
    "attention_stem_bilinear_fusion_multiple_head_multiscale_branches",
    "attention_stem_multi_head_multiscale_tcn",
    "sepformer",
    "unet1d",
    "tasnet",
]

FRACTION_PLOT_MODEL_LABELS = {
    "mlp_singlehead": "MLP",
    "conv1dnetwork": "Shallow Conv1D",
    "multiscale_dilated": "Multiscale Dilated",
    "multiple_head_multiscale_branches": "Multi Head",
    "attention_stem_bilinear_fusion_multiple_head_multiscale_branches": "Attention Stem Bilinear Fusion",
    "attention_stem_multi_head_multiscale_tcn": "Attention stem Multihead and multiscale TCN",
    "sepformer": "SepFormer",
    "unet1d": "U-Net1d",
    "tasnet": "tasnet",
}

FRACTION_PLOT_COLORS = {
    "mlp_singlehead": "#0072B2",
    "conv1dnetwork": "#E69F00",
    "multiscale_dilated": "#D55E00",
    "multiple_head_multiscale_branches": "#56B4E9",
    "attention_stem_bilinear_fusion_multiple_head_multiscale_branches": "#000000",
    "attention_stem_multi_head_multiscale_tcn": "#6A3D9A",
    "sepformer": "#1B9E77",
    "unet1d": "#7570B3",
    "tasnet": "#E7298A",
}

FRACTION_PLOT_DNN_MODELS = {"sepformer", "unet1d", "tasnet"}
FRACTION_PLOT_ALLOWED_VALUES = [0.05, 0.10, 0.25, 0.50, 0.75, 1.00]


@dataclass
class CostResearchDataset:
    """Dataset wrapper."""
    raw: pd.DataFrame
    aggregated: pd.DataFrame
    warnings: list[str]
    source_labels: list[str]
    has_seed_repeats: bool
    has_data_availability: bool


def _safe_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _bootstrap_mean_ci(values, confidence: float = 0.95, n_resamples: int = 2000, seed: int = 12345) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        return float(arr[0]), float(arr[0])
    rng = np.random.default_rng(seed)
    draws = rng.choice(arr, size=(n_resamples, arr.size), replace=True).mean(axis=1)
    alpha = 1.0 - confidence
    return float(np.quantile(draws, alpha / 2.0)), float(np.quantile(draws, 1.0 - alpha / 2.0))


def _coerce_numeric(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    for column in columns:
        if column not in df.columns:
            df[column] = pd.NA
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def prepare_cost_research_dataset(
    reports_df: pd.DataFrame,
    registry: list[dict] | None,
    uploaded_files: list | None = None,
) -> CostResearchDataset:
    """Prepare cost research dataset.
    
    Args:
        reports_df: Evaluation reports table.
        registry: Model registry entries.
        uploaded_files: Files uploaded through Streamlit.
    """
    base = prepare_research_dataset(reports_df, registry, uploaded_files=uploaded_files)
    warnings = list(base.warnings)
    raw = base.dataframe.copy()
    if raw.empty:
        return CostResearchDataset(
            raw=raw,
            aggregated=raw,
            warnings=warnings,
            source_labels=base.source_labels,
            has_seed_repeats=False,
            has_data_availability=False,
        )

    raw = _normalize_cost_dataframe(raw, warnings)
    aggregated = aggregate_cost_dataset(raw)
    has_seed_repeats = bool("seed_count" in aggregated.columns and aggregated["seed_count"].fillna(0).gt(1).any())
    has_data_availability = bool("train_fraction" in raw.columns and raw["train_fraction"].notna().any())
    return CostResearchDataset(
        raw=raw,
        aggregated=aggregated,
        warnings=warnings,
        source_labels=base.source_labels,
        has_seed_repeats=has_seed_repeats,
        has_data_availability=has_data_availability,
    )


def _normalize_cost_dataframe(df: pd.DataFrame, warnings: list[str]) -> pd.DataFrame:
    normalized = df.copy()
    numeric_columns = [
        "train_fraction",
        "train_samples",
        "val_samples",
        "test_samples",
        "parameters",
        "inference_time_ms",
        "training_time_s",
        "peak_memory_mb",
        "total_flops",
        "estimated_energy",
        "epochs",
        "batch_size",
        "noise_level",
        "overlap_level",
        "nonstationarity_level",
        "num_components",
        "macro_corr",
        "macro_snr",
        "test_loss",
        "seed",
    ]
    normalized = _coerce_numeric(normalized, numeric_columns)
    if "peak_memory_mb" not in normalized.columns:
        normalized["peak_memory_mb"] = pd.NA
    if "hardware_info" not in normalized.columns:
        normalized["hardware_info"] = ""
    if "device" not in normalized.columns:
        normalized["device"] = ""
    if "gpu_name" not in normalized.columns:
        normalized["gpu_name"] = ""
    if "config_name" not in normalized.columns:
        normalized["config_name"] = ""

    if ("train_fraction" not in normalized.columns or not normalized["train_fraction"].notna().any()) and "train_samples" in normalized.columns:
        group_cols = [column for column in ["model_name", "config_name"] if column in normalized.columns]
        if group_cols and normalized["train_samples"].notna().any():
            max_samples = normalized.groupby(group_cols)["train_samples"].transform("max")
            can_infer = max_samples.notna() & max_samples.gt(0)
            inferred = normalized["train_samples"] / max_samples.where(can_infer, np.nan)
            if inferred.notna().any():
                normalized["train_fraction"] = inferred
                warnings.append(
                    "Inferred `train_fraction` from `train_samples / max(train_samples)` within each model/config group."
                )

    normalized["accuracy_per_training_second"] = normalized["macro_corr"] / normalized["training_time_s"].replace(0, np.nan)
    normalized["snr_per_training_second"] = normalized["macro_snr"] / normalized["training_time_s"].replace(0, np.nan)
    normalized["accuracy_per_1000_parameters"] = normalized["macro_corr"] / (normalized["parameters"] / 1000.0).replace(0, np.nan)
    normalized["snr_per_1000_parameters"] = normalized["macro_snr"] / (normalized["parameters"] / 1000.0).replace(0, np.nan)
    normalized["accuracy_per_inference_ms"] = normalized["macro_corr"] / normalized["inference_time_ms"].replace(0, np.nan)
    normalized["snr_per_inference_ms"] = normalized["macro_snr"] / normalized["inference_time_ms"].replace(0, np.nan)
    normalized["accuracy_per_sample"] = normalized["macro_corr"] / normalized["train_samples"].replace(0, np.nan)
    normalized["limited_data_bucket"] = np.where(
        normalized["train_fraction"].notna() & normalized["train_fraction"].le(0.25),
        "limited",
        np.where(normalized["train_fraction"].notna(), "full_or_mid", "unknown"),
    )
    return normalized


def aggregate_cost_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate cost dataset.
    
    Args:
        df: Input dataframe.
    """
    group_cols = [column for column in ["model_name", "display_name", "model_type", "family", "architecture", "config_name", "train_fraction", "train_samples"] if column in df.columns]
    if not group_cols:
        return df.copy()

    numeric_cols = [
        "macro_corr",
        "macro_snr",
        "test_loss",
        "module_count",
        "parameters",
        "inference_time_ms",
        "training_time_s",
        "peak_memory_mb",
        "total_flops",
        "estimated_energy",
        "accuracy_per_training_second",
        "snr_per_training_second",
        "accuracy_per_1000_parameters",
        "snr_per_1000_parameters",
        "accuracy_per_inference_ms",
        "snr_per_inference_ms",
        "accuracy_per_sample",
    ]
    available_numeric = [column for column in numeric_cols if column in df.columns]
    aggregated = (
        df.groupby(group_cols, dropna=False)[available_numeric]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
    )
    aggregated.columns = [
        "_".join([part for part in column if part]).rstrip("_")
        if isinstance(column, tuple)
        else column
        for column in aggregated.columns
    ]
    rename_map = {f"{column}_mean": column for column in available_numeric}
    aggregated = aggregated.rename(columns=rename_map)
    for column in available_numeric:
        count_col = f"{column}_count"
        if count_col in aggregated.columns:
            aggregated["seed_count"] = aggregated.get("seed_count", aggregated[count_col]).combine_first(aggregated[count_col])
    if "seed_count" not in aggregated.columns:
        aggregated["seed_count"] = 1
    for metric in ["macro_corr", "macro_snr", "test_loss"]:
        std_col = f"{metric}_std"
        if std_col in aggregated.columns:
            aggregated[f"{metric}_sem"] = aggregated[std_col] / aggregated["seed_count"].clip(lower=1).pow(0.5)
            aggregated[f"{metric}_ci95"] = 1.96 * aggregated[f"{metric}_sem"]
    return aggregated


def filter_cost_dataset(
    dataset: CostResearchDataset,
    *,
    selected_models: list[str] | None = None,
    selected_types: list[str] | None = None,
    selected_families: list[str] | None = None,
    selected_sources: list[str] | None = None,
    data_regime: str = "all",
) -> CostResearchDataset:
    """Filter cost dataset."""
    raw = dataset.raw.copy()
    aggregated = dataset.aggregated.copy()
    if selected_models:
        raw = raw[raw["display_name"].isin(selected_models) | raw["model_name"].isin(selected_models)]
        aggregated = aggregated[aggregated["display_name"].isin(selected_models) | aggregated["model_name"].isin(selected_models)]
    if selected_types:
        raw = raw[raw["model_type"].isin(selected_types)]
        aggregated = aggregated[aggregated["model_type"].isin(selected_types)]
    if selected_families:
        raw = raw[raw["family"].isin(selected_families)]
        aggregated = aggregated[aggregated["family"].isin(selected_families)]
    if selected_sources and "report_source" in raw.columns:
        raw = raw[raw["report_source"].isin(selected_sources)]
    if data_regime == "limited":
        raw = raw[raw["limited_data_bucket"] == "limited"]
        aggregated = aggregated[aggregated["train_fraction"].notna() & aggregated["train_fraction"].le(0.25)]
    elif data_regime == "full":
        raw = raw[raw["train_fraction"].isna() | raw["train_fraction"].ge(0.99)]
        aggregated = aggregated[aggregated["train_fraction"].isna() | aggregated["train_fraction"].ge(0.99)]
    return CostResearchDataset(
        raw=raw.reset_index(drop=True),
        aggregated=aggregated.reset_index(drop=True),
        warnings=dataset.warnings,
        source_labels=dataset.source_labels,
        has_seed_repeats=dataset.has_seed_repeats,
        has_data_availability=dataset.has_data_availability,
    )


def compute_cost_summary(dataset: CostResearchDataset) -> dict[str, object]:
    """Compute cost summary.
    
    Args:
        dataset: Prepared dataset object.
    """
    df = dataset.aggregated
    summary: dict[str, object] = {}
    snn_df = df[df["model_type"] == "SNN"]
    dnn_df = df[df["model_type"] == "DNN"]

    best_snn = best_row(snn_df, "macro_corr")
    best_dnn = best_row(dnn_df, "macro_corr")
    summary["best_snn_model"] = None if best_snn is None else best_snn.get("display_name")
    summary["best_dnn_model"] = None if best_dnn is None else best_dnn.get("display_name")
    summary["best_snn_macro_corr"] = None if best_snn is None else float(best_snn["macro_corr"])
    summary["best_dnn_macro_corr"] = None if best_dnn is None else float(best_dnn["macro_corr"])
    summary["absolute_performance_gap"] = None
    summary["relative_performance_gap"] = None
    summary["snn_parameter_reduction_factor"] = None
    summary["snn_inference_speedup_factor"] = None
    summary["snn_training_speedup_factor"] = None
    if best_snn is not None and best_dnn is not None:
        summary["absolute_performance_gap"] = float(best_dnn["macro_corr"]) - float(best_snn["macro_corr"])
        summary["relative_performance_gap"] = summary["absolute_performance_gap"] / max(abs(float(best_dnn["macro_corr"])), 1e-8)
        summary["snn_parameter_reduction_factor"] = _ratio(best_dnn.get("parameters"), best_snn.get("parameters"))
        summary["snn_inference_speedup_factor"] = _ratio(best_dnn.get("inference_time_ms"), best_snn.get("inference_time_ms"))
        summary["snn_training_speedup_factor"] = _ratio(best_dnn.get("training_time_s"), best_snn.get("training_time_s"))

    limited = dataset.aggregated[dataset.aggregated["train_fraction"].notna() & dataset.aggregated["train_fraction"].le(0.25)]
    full = dataset.aggregated[dataset.aggregated["train_fraction"].isna() | dataset.aggregated["train_fraction"].ge(0.99)]
    best_limited = best_row(limited, "macro_corr")
    best_full = best_row(full if not full.empty else dataset.aggregated, "macro_corr")
    summary["best_model_under_limited_data"] = None if best_limited is None else best_limited.get("display_name")
    summary["best_model_under_full_data"] = None if best_full is None else best_full.get("display_name")
    summary["best_accuracy_per_training_second_model"] = _best_name(dataset.aggregated, "accuracy_per_training_second")
    summary["best_accuracy_per_parameter_model"] = _best_name(dataset.aggregated, "accuracy_per_1000_parameters")
    summary["best_accuracy_per_sample_model"] = _best_name(dataset.aggregated, "accuracy_per_sample")

    limited_conclusions = compute_limited_data_conclusions(dataset)
    summary.update(limited_conclusions)
    return summary


def _ratio(numerator, denominator) -> float | None:
    num = _safe_float(numerator)
    den = _safe_float(denominator)
    if num > 0 and den > 0:
        return num / den
    return None


def _best_name(df: pd.DataFrame, metric: str) -> str | None:
    row = best_row(df, metric)
    return None if row is None else row.get("display_name")


def compute_limited_data_conclusions(dataset: CostResearchDataset) -> dict[str, object]:
    """Compute limited data conclusions.
    
    Args:
        dataset: Prepared dataset object.
    """
    aggregated = dataset.aggregated
    if "train_fraction" not in aggregated.columns or not aggregated["train_fraction"].notna().any():
        return {
            "best_snn_per_fraction": pd.DataFrame(),
            "best_dnn_per_fraction": pd.DataFrame(),
            "best_model_per_fraction": pd.DataFrame(),
            "dnn_overtakes_at_fraction": None,
            "closest_fraction": None,
            "best_tradeoff_fraction": None,
            "low_data_snn_advantage": None,
        }

    best_model = aggregated.dropna(subset=["train_fraction", "macro_corr"]).sort_values("macro_corr", ascending=False).groupby("train_fraction", as_index=False).first()
    best_snn = aggregated[(aggregated["model_type"] == "SNN")].dropna(subset=["train_fraction", "macro_corr"]).sort_values("macro_corr", ascending=False).groupby("train_fraction", as_index=False).first()
    best_dnn = aggregated[(aggregated["model_type"] == "DNN")].dropna(subset=["train_fraction", "macro_corr"]).sort_values("macro_corr", ascending=False).groupby("train_fraction", as_index=False).first()

    merged = best_snn.merge(best_dnn, on="train_fraction", suffixes=("_snn", "_dnn"))
    overtakes = merged[merged["macro_corr_dnn"] - merged["macro_corr_snn"] > 0.02]
    closest = merged.assign(abs_gap=(merged["macro_corr_dnn"] - merged["macro_corr_snn"]).abs())
    best_tradeoff = aggregated.dropna(subset=["train_fraction", "accuracy_per_training_second"]).sort_values("accuracy_per_training_second", ascending=False).groupby("train_fraction", as_index=False).first()
    low_data = merged[merged["train_fraction"] <= 0.25] if not merged.empty else merged

    return {
        "best_snn_per_fraction": best_snn,
        "best_dnn_per_fraction": best_dnn,
        "best_model_per_fraction": best_model,
        "dnn_overtakes_at_fraction": None if overtakes.empty else float(overtakes["train_fraction"].min()),
        "closest_fraction": None if closest.empty else float(closest.sort_values("abs_gap").iloc[0]["train_fraction"]),
        "best_tradeoff_fraction": None if best_tradeoff.empty else float(best_tradeoff.sort_values("accuracy_per_training_second", ascending=False).iloc[0]["train_fraction"]),
        "low_data_snn_advantage": None if low_data.empty else bool((low_data["macro_corr_snn"] >= low_data["macro_corr_dnn"]).any()),
    }


def build_cost_tables(dataset: CostResearchDataset) -> dict[str, pd.DataFrame]:
    """Build cost tables.
    
    Args:
        dataset: Prepared dataset object.
    """
    tables = build_results_tables(dataset.aggregated)
    tables["training_cost"] = dataset.aggregated[[column for column in ["display_name", "model_type", "training_time_s", "parameters", "inference_time_ms", "peak_memory_mb"] if column in dataset.aggregated.columns]].sort_values("training_time_s", ascending=True, na_position="last")
    tables["data_availability"] = dataset.aggregated[[column for column in ["display_name", "model_type", "train_fraction", "train_samples", "macro_corr", "macro_snr", "test_loss", "seed_count"] if column in dataset.aggregated.columns]].sort_values(["train_fraction", "macro_corr"], ascending=[True, False], na_position="last")
    conclusions = compute_limited_data_conclusions(dataset)
    tables["best_model_per_train_fraction"] = conclusions["best_model_per_fraction"]
    tables["best_snn_per_train_fraction"] = conclusions["best_snn_per_fraction"]
    tables["best_dnn_per_train_fraction"] = conclusions["best_dnn_per_fraction"]
    tables["cost_normalized_ranking"] = dataset.aggregated[[column for column in ["display_name", "model_type", "accuracy_per_training_second", "accuracy_per_1000_parameters", "accuracy_per_inference_ms", "accuracy_per_sample", "macro_corr"] if column in dataset.aggregated.columns]].sort_values(["accuracy_per_training_second", "macro_corr"], ascending=[False, False], na_position="last")
    tables["pareto_optimal_models"] = compute_pareto_frontier(dataset.aggregated, metric="macro_corr")
    tables["missing_data_report"] = missing_data_report(dataset.raw)
    return tables


def missing_data_report(df: pd.DataFrame) -> pd.DataFrame:
    """Missing data report.
    
    Args:
        df: Input dataframe.
    """
    tracked = [
        "training_time_s",
        "parameters",
        "inference_time_ms",
        "peak_memory_mb",
        "train_fraction",
        "train_samples",
        "macro_corr",
        "macro_snr",
        "test_loss",
        "seed",
        "gpu_name",
        "total_flops",
        "estimated_energy",
    ]
    rows = []
    for column in tracked:
        rows.append(
            {
                "column": column,
                "available": column in df.columns and bool(df[column].notna().any()),
                "non_null_rows": int(df[column].notna().sum()) if column in df.columns else 0,
            }
        )
    return pd.DataFrame(rows)


def _base_axis(fig_size: tuple[float, float] = (10.5, 4.8)) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=fig_size)
    return fig, ax


def _bar_chart(df: pd.DataFrame, metric: str, title: str, *, ascending: bool | None = None) -> plt.Figure | None:
    subset = df.dropna(subset=[metric]).copy()
    if subset.empty:
        return None
    if ascending is None:
        ascending = metric in {"training_time_s", "parameters", "inference_time_ms", "peak_memory_mb", "test_loss"}
    subset = subset.sort_values(metric, ascending=ascending)
    fig_width = max(10.5, 0.82 * len(subset))
    fig, ax = _base_axis((fig_width, 5.8))
    colors = [MODEL_TYPE_COLORS.get(model_type, MODEL_TYPE_COLORS["Unknown"]) for model_type in subset["model_type"]]
    positions = np.arange(len(subset))
    ax.bar(positions, subset[metric], color=colors)
    ax.set_title(title)
    ax.set_ylabel(COST_METRIC_LABELS.get(metric, metric))
    _style_model_xaxis(fig, ax, subset["display_name"])
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def _grouped_type_bar(df: pd.DataFrame, metric: str, title: str) -> plt.Figure | None:
    subset = df.dropna(subset=[metric, "model_type"]).copy()
    subset = subset[subset["model_type"].isin(["SNN", "DNN", "Classical"])]
    if subset.empty:
        return None
    grouped = subset.groupby("model_type")[metric].mean().reindex(["SNN", "DNN", "Classical"]).dropna()
    if grouped.empty:
        return None
    fig, ax = _base_axis((8.5, 4.4))
    ax.bar(grouped.index, grouped.values, color=[MODEL_TYPE_COLORS.get(index, MODEL_TYPE_COLORS["Unknown"]) for index in grouped.index])
    ax.set_title(title)
    ax.set_ylabel(COST_METRIC_LABELS.get(metric, metric))
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def _box_plot_by_type(df: pd.DataFrame, metric: str, title: str) -> plt.Figure | None:
    subset = df.dropna(subset=[metric, "model_type"]).copy()
    if subset.empty:
        return None
    ordered_types = [item for item in MODEL_TYPE_ORDER if item in subset["model_type"].unique()]
    series = [subset.loc[subset["model_type"] == model_type, metric] for model_type in ordered_types]
    if not any(len(values) for values in series):
        return None
    fig, ax = _base_axis((8.5, 4.4))
    ax.boxplot(series, tick_labels=ordered_types, patch_artist=True)
    ax.set_title(title)
    ax.set_ylabel(COST_METRIC_LABELS.get(metric, metric))
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def _scatter(df: pd.DataFrame, x: str, y: str, title: str, bubble: str | None = None) -> plt.Figure | None:
    subset = df.dropna(subset=[x, y]).copy()
    if subset.empty:
        return None
    fig, ax = _base_axis()
    bubble_sizes = None
    if bubble and bubble in subset.columns and subset[bubble].notna().any():
        scale = subset[bubble].clip(lower=1e-8)
        bubble_sizes = 180 * np.sqrt(scale / scale.max())
    for model_type, group in subset.groupby("model_type", dropna=False):
        size = bubble_sizes.loc[group.index] if bubble_sizes is not None else 70
        ax.scatter(group[x], group[y], s=size, alpha=0.7, label=model_type, color=MODEL_TYPE_COLORS.get(model_type, MODEL_TYPE_COLORS["Unknown"]))
    ax.set_xlabel(COST_METRIC_LABELS.get(x, SIGNAL_CONDITION_COLUMNS.get(x, x)))
    ax.set_ylabel(COST_METRIC_LABELS.get(y, SIGNAL_CONDITION_COLUMNS.get(y, y)))
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig


def _pareto_chart(df: pd.DataFrame, x_metric: str, y_metric: str, title: str) -> plt.Figure | None:
    subset = df.dropna(subset=[x_metric, y_metric]).copy()
    if subset.empty:
        return None
    frontier = _compute_single_cost_pareto(subset, x_metric, y_metric)
    fig, ax = _base_axis((10.5, 5.0))
    for model_type, group in subset.groupby("model_type", dropna=False):
        ax.scatter(group[x_metric], group[y_metric], s=65, alpha=0.55, label=model_type, color=MODEL_TYPE_COLORS.get(model_type, MODEL_TYPE_COLORS["Unknown"]))
    if not frontier.empty:
        ordered = frontier.sort_values(x_metric)
        ax.plot(ordered[x_metric], ordered[y_metric], color="#111111", linewidth=1.5, label="Pareto frontier")
    ax.set_xlabel(COST_METRIC_LABELS.get(x_metric, x_metric))
    ax.set_ylabel(COST_METRIC_LABELS.get(y_metric, y_metric))
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig


def _compute_single_cost_pareto(df: pd.DataFrame, x_metric: str, y_metric: str) -> pd.DataFrame:
    keep_indices: list[int] = []
    for idx, row in df.iterrows():
        dominated = False
        for other_idx, other in df.iterrows():
            if idx == other_idx:
                continue
            if other[y_metric] >= row[y_metric] and other[x_metric] <= row[x_metric]:
                if other[y_metric] > row[y_metric] or other[x_metric] < row[x_metric]:
                    dominated = True
                    break
        if not dominated:
            keep_indices.append(idx)
    return df.loc[keep_indices]


def _line_by_fraction(df: pd.DataFrame, metric: str, title: str, *, compare_mode: str = "all", error_bars: bool = True) -> plt.Figure | None:
    subset = df.dropna(subset=["train_fraction", metric]).copy()
    if subset.empty:
        return None
    subset = subset[subset["train_fraction"].apply(lambda value: any(math.isclose(float(value), allowed, rel_tol=0.0, abs_tol=1e-9) for allowed in FRACTION_PLOT_ALLOWED_VALUES))].copy()
    if subset.empty:
        return None

    if "model_name" in subset.columns:
        subset["model_name"] = subset["model_name"].astype(str).str.strip().str.lower()

    x_min: float | None = None
    x_max: float | None = None

    fig, ax = _base_axis((10.8, 5.0))
    if compare_mode == "all":
        if "model_name" not in subset.columns:
            return None
        subset = subset[subset["model_name"].isin(FRACTION_PLOT_MODEL_ORDER)].copy()
        if subset.empty:
            return None

        model_min_fraction = subset.groupby("model_name", dropna=False)["train_fraction"].min()
        if model_min_fraction.empty:
            return None
        shared_start_fraction = float(model_min_fraction.max())
        subset = subset[subset["train_fraction"].ge(shared_start_fraction)].copy()
        if subset.empty:
            return None
        x_min = float(subset["train_fraction"].min())
        x_max = float(subset["train_fraction"].max())

        subset["plot_label"] = subset["model_name"].map(FRACTION_PLOT_MODEL_LABELS).fillna(subset["display_name"])
        subset["plot_order"] = subset["model_name"].map({name: idx for idx, name in enumerate(FRACTION_PLOT_MODEL_ORDER)})
        ordered_models = (
            subset[["model_name", "plot_label", "plot_order"]]
            .drop_duplicates()
            .sort_values(["plot_order", "plot_label"], na_position="last")
        )
        for _, model_row in ordered_models.iterrows():
            model_name = str(model_row["model_name"])
            label = str(model_row["plot_label"])
            group = subset[subset["model_name"] == model_name].sort_values("train_fraction")
            if group.empty:
                continue
            line_style = "--" if model_name in FRACTION_PLOT_DNN_MODELS else "-"
            color = FRACTION_PLOT_COLORS.get(model_name)
            ax.plot(
                group["train_fraction"],
                group[metric],
                marker="s",
                markersize=5.5,
                linewidth=1.7,
                linestyle=line_style,
                label=label,
                color=color,
            )
            ci_col = f"{metric}_ci95"
            if error_bars and ci_col in group.columns and group[ci_col].notna().any():
                ax.fill_between(
                    group["train_fraction"],
                    group[metric] - group[ci_col],
                    group[metric] + group[ci_col],
                    alpha=0.10,
                    color=color,
                )
    elif compare_mode == "type_average":
        grouped = subset.groupby(["train_fraction", "model_type"], dropna=False)[metric].mean().reset_index()
        if not grouped.empty:
            x_min = float(grouped["train_fraction"].min())
            x_max = float(grouped["train_fraction"].max())
        for label, group in grouped.groupby("model_type", dropna=False):
            ordered = group.sort_values("train_fraction")
            marker = "s" if label in {"SNN", "DNN"} else "o"
            line_style = "--" if label == "DNN" else "-"
            ax.plot(ordered["train_fraction"], ordered[metric], marker=marker, linewidth=2.0, linestyle=line_style, label=label, color=MODEL_TYPE_COLORS.get(label, MODEL_TYPE_COLORS["Unknown"]))
    elif compare_mode == "best_snn_vs_dnn":
        best_snn = subset[subset["model_type"] == "SNN"].sort_values(metric, ascending=False).groupby("train_fraction", as_index=False).first()
        best_dnn = subset[subset["model_type"] == "DNN"].sort_values(metric, ascending=False).groupby("train_fraction", as_index=False).first()
        if not best_snn.empty or not best_dnn.empty:
            merged_fractions = pd.concat([best_snn["train_fraction"], best_dnn["train_fraction"]], ignore_index=True).dropna()
            if not merged_fractions.empty:
                x_min = float(merged_fractions.min())
                x_max = float(merged_fractions.max())
        if not best_snn.empty:
            ax.plot(best_snn["train_fraction"], best_snn[metric], marker="s", linewidth=2.0, linestyle="-", label="Best SNN", color=MODEL_TYPE_COLORS["SNN"])
        if not best_dnn.empty:
            ax.plot(best_dnn["train_fraction"], best_dnn[metric], marker="s", linewidth=2.0, linestyle="--", label="Best DNN", color=MODEL_TYPE_COLORS["DNN"])
    else:
        return None
    if x_min is not None and x_max is not None:
        if math.isclose(x_min, x_max):
            ax.set_xlim(max(0.0, x_min - 0.02), min(1.02, x_max + 0.02))
        else:
            pad = min(0.03, max((x_max - x_min) * 0.04, 0.005))
            ax.set_xlim(max(0.0, x_min - pad), min(1.02, x_max + pad))
    ax.set_xlabel(COST_METRIC_LABELS["train_fraction"])
    ax.set_ylabel(COST_METRIC_LABELS.get(metric, metric))
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
    return fig


def _gap_by_fraction(df: pd.DataFrame, metric: str, title: str) -> plt.Figure | None:
    subset = df.dropna(subset=["train_fraction", metric]).copy()
    if subset.empty:
        return None
    snn = subset[subset["model_type"] == "SNN"].sort_values(metric, ascending=False).groupby("train_fraction", as_index=False).first()
    dnn = subset[subset["model_type"] == "DNN"].sort_values(metric, ascending=False).groupby("train_fraction", as_index=False).first()
    merged = snn.merge(dnn, on="train_fraction", suffixes=("_snn", "_dnn"))
    if merged.empty:
        return None
    fig, ax = _base_axis((9.5, 4.8))
    gap = merged[f"{metric}_dnn"] - merged[f"{metric}_snn"]
    ax.bar(merged["train_fraction"], gap, width=0.05 if merged["train_fraction"].nunique() > 4 else 0.08, color="#9467bd")
    ax.set_xlabel(COST_METRIC_LABELS["train_fraction"])
    ax.set_ylabel(f"{COST_METRIC_LABELS.get(metric, metric)} gap (DNN - SNN)")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def _heatmap_by_fraction(df: pd.DataFrame, metric: str, title: str, *, by_type: bool = False) -> plt.Figure | None:
    subset = df.dropna(subset=["train_fraction", metric]).copy()
    if subset.empty:
        return None
    index_col = "model_type" if by_type else "display_name"
    pivot = subset.pivot_table(index=index_col, columns="train_fraction", values=metric, aggfunc="mean")
    if pivot.empty:
        return None
    fig, ax = _base_axis((10.8, max(4.0, 0.45 * len(pivot) + 1.5)))
    image = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="viridis")
    ax.set_title(title)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([f"{column:.2f}" for column in pivot.columns], rotation=45, ha="right")
    ax.set_xlabel(COST_METRIC_LABELS["train_fraction"])
    fig.colorbar(image, ax=ax, shrink=0.8)
    fig.tight_layout()
    return fig


def _ranking_by_fraction(df: pd.DataFrame, metric: str, title: str) -> plt.Figure | None:
    subset = df.dropna(subset=["train_fraction", metric]).copy()
    if subset.empty:
        return None
    best = subset.sort_values(metric, ascending=False).groupby("train_fraction", as_index=False).first()
    fig, ax = _base_axis((10.5, 4.8))
    ax.bar(best["train_fraction"], best[metric], color="#2ca02c")
    ax.set_title(title)
    ax.set_xlabel(COST_METRIC_LABELS["train_fraction"])
    ax.set_ylabel(COST_METRIC_LABELS.get(metric, metric))
    for _, row in best.iterrows():
        ax.annotate(str(row["display_name"]), (row["train_fraction"], row[metric]), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def _minimum_data_threshold(df: pd.DataFrame, threshold: float = 0.8) -> plt.Figure | None:
    subset = df.dropna(subset=["train_fraction", "macro_corr"]).copy()
    if subset.empty:
        return None
    rows = []
    for label, group in subset.groupby("display_name", dropna=False):
        reached = group[group["macro_corr"] >= threshold]
        if reached.empty:
            continue
        rows.append({"display_name": label, "train_fraction": reached["train_fraction"].min(), "model_type": group["model_type"].iloc[0]})
    result = pd.DataFrame(rows).sort_values("train_fraction")
    if result.empty:
        return None
    fig_width = max(10.5, 0.82 * len(result))
    fig, ax = _base_axis((fig_width, 5.8))
    positions = np.arange(len(result))
    ax.bar(positions, result["train_fraction"], color=[MODEL_TYPE_COLORS.get(item, MODEL_TYPE_COLORS["Unknown"]) for item in result["model_type"]])
    ax.set_title(f"Minimum train fraction to reach macro correlation >= {threshold:.2f}")
    ax.set_ylabel(COST_METRIC_LABELS["train_fraction"])
    _style_model_xaxis(fig, ax, result["display_name"])
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def _cost_normalized_ranking(df: pd.DataFrame) -> plt.Figure | None:
    subset = df.dropna(subset=["accuracy_per_training_second", "macro_corr"]).copy()
    if subset.empty:
        return None
    subset = subset.sort_values("accuracy_per_training_second", ascending=False).head(12)
    fig, ax = _base_axis((10.5, 4.8))
    ax.barh(subset["display_name"], subset["accuracy_per_training_second"], color=[MODEL_TYPE_COLORS.get(item, MODEL_TYPE_COLORS["Unknown"]) for item in subset["model_type"]])
    ax.set_title("Cost-normalized ranking by macro correlation per training second")
    ax.set_xlabel(COST_METRIC_LABELS["accuracy_per_training_second"])
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    return fig


def _ratio_chart(summary: dict[str, object], key: str, title: str) -> plt.Figure | None:
    value = summary.get(key)
    if value is None:
        return None
    fig, ax = _base_axis((6.0, 4.0))
    ax.bar([title], [value], color="#17becf")
    ax.set_ylabel("Factor (DNN / SNN)")
    ax.set_title(title)
    ax.text(0, value, f"{value:.2f}x", ha="center", va="bottom")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def _combined_accuracy_loss_vs_efficiency(df: pd.DataFrame) -> plt.Figure | None:
    best_dnn = best_row(df[df["model_type"] == "DNN"], "macro_corr")
    snn = df[(df["model_type"] == "SNN") & df["macro_corr"].notna()]
    if best_dnn is None or snn.empty or "training_time_s" not in snn.columns:
        return None
    dnn_train = _safe_float(best_dnn.get("training_time_s"))
    if dnn_train <= 0:
        return None
    plot_df = snn.dropna(subset=["training_time_s"]).copy()
    if plot_df.empty:
        return None
    plot_df["performance_loss"] = float(best_dnn["macro_corr"]) - plot_df["macro_corr"]
    plot_df["training_speedup"] = dnn_train / plot_df["training_time_s"].clip(lower=1e-8)
    fig, ax = _base_axis((10.5, 4.8))
    ax.scatter(plot_df["training_speedup"], plot_df["performance_loss"], s=85, alpha=0.8, color=MODEL_TYPE_COLORS["SNN"])
    for _, row in plot_df.iterrows():
        ax.annotate(str(row["display_name"]), (row["training_speedup"], row["performance_loss"]), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_title("Accuracy loss versus training speedup")
    ax.set_xlabel("Training speedup relative to best DNN")
    ax.set_ylabel("Macro correlation loss")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def build_cost_chart_catalog(dataset: CostResearchDataset) -> list[ChartSpec]:
    """Build cost chart catalog.
    
    Args:
        dataset: Prepared dataset object.
    """
    df = prepare_chart_generation_dataframe(dataset.aggregated)
    data_availability_df = dataset.aggregated
    chart_dataset = CostResearchDataset(
        raw=dataset.raw,
        aggregated=df,
        warnings=dataset.warnings,
        source_labels=dataset.source_labels,
        has_seed_repeats=dataset.has_seed_repeats,
        has_data_availability=dataset.has_data_availability,
    )
    charts: list[ChartSpec] = []

    def add(section: str, slug: str, title: str, figure: plt.Figure | None) -> None:
        if figure is not None:
            charts.append(ChartSpec(section=section, slug=slug, title=title, figure=figure))

    for metric, title in [
        ("training_time_s", "Training time per model"),
        ("parameters", "Parameter count per model"),
        ("inference_time_ms", "Inference time per model"),
        ("peak_memory_mb", "Peak memory per model"),
    ]:
        add("Training Cost", f"{metric}_per_model", title, _bar_chart(df, metric, title))
    add("Training Cost", "training_time_by_type", "Training time grouped by SNN vs DNN", _grouped_type_bar(df, "training_time_s", "Training time grouped by model type"))
    add("Training Cost", "training_time_box", "Training time distribution by model type", _box_plot_by_type(df, "training_time_s", "Training time distribution by model type"))
    for x, y, title in [
        ("training_time_s", "macro_corr", "Training time vs macro correlation"),
        ("training_time_s", "macro_snr", "Training time vs macro SNR"),
        ("parameters", "training_time_s", "Parameters vs training time"),
        ("inference_time_ms", "training_time_s", "Inference time vs training time"),
    ]:
        add("Training Cost", f"{y}_vs_{x}", title, _scatter(df, x, y, title))
    add("Training Cost", "bubble_training_time_corr", "Training time vs macro correlation bubble plot", _scatter(df, "training_time_s", "macro_corr", "Training time vs macro correlation", bubble="parameters"))
    add("Training Cost", "pareto_training_corr", "Compute-efficiency frontier: macro correlation vs training cost", _pareto_chart(df, "training_time_s", "macro_corr", "Macro correlation vs training time frontier"))
    add("Training Cost", "pareto_training_snr", "Compute-efficiency frontier: macro SNR vs training cost", _pareto_chart(df, "training_time_s", "macro_snr", "Macro SNR vs training time frontier"))

    for metric, title in [
        ("macro_corr", "Macro correlation vs train fraction"),
        ("macro_snr", "Macro SNR vs train fraction"),
        ("test_loss", "Test loss vs train fraction"),
    ]:
        add("Data Availability", f"{metric}_fraction_all", title, _line_by_fraction(data_availability_df, metric, title, compare_mode="all", error_bars=dataset.has_seed_repeats))
        add("Data Availability", f"{metric}_fraction_type_average", f"Average SNN vs DNN {title.lower()}", _line_by_fraction(data_availability_df, metric, f"Average SNN vs DNN {title.lower()}", compare_mode="type_average", error_bars=False))
    add("Data Availability", "best_snn_vs_dnn_fraction_corr", "Best SNN vs best DNN macro correlation across train fraction", _line_by_fraction(data_availability_df, "macro_corr", "Best SNN vs best DNN macro correlation across train fraction", compare_mode="best_snn_vs_dnn", error_bars=False))
    add("Data Availability", "best_snn_vs_dnn_fraction_snr", "Best SNN vs best DNN macro SNR across train fraction", _line_by_fraction(data_availability_df, "macro_snr", "Best SNN vs best DNN macro SNR across train fraction", compare_mode="best_snn_vs_dnn", error_bars=False))
    add("Data Availability", "data_efficiency_gap", "Data-efficiency gap chart", _gap_by_fraction(data_availability_df, "macro_corr", "Best DNN minus best SNN macro correlation at each train fraction"))
    add("Data Availability", "sample_efficiency", "Sample-efficiency chart", _line_by_fraction(data_availability_df, "accuracy_per_sample", "Macro correlation per training sample", compare_mode="all", error_bars=False))
    add("Data Availability", "minimum_data_threshold", "Minimum-data threshold chart", _minimum_data_threshold(data_availability_df))
    add("Data Availability", "ranking_by_fraction", "Best model at each train fraction", _ranking_by_fraction(data_availability_df, "macro_corr", "Best model at each train fraction"))
    add("Data Availability", "heatmap_model_fraction", "Model performance across train fraction", _heatmap_by_fraction(data_availability_df, "macro_corr", "Model performance heatmap across train fraction"))
    add("Data Availability", "heatmap_type_fraction", "Model-type performance across train fraction", _heatmap_by_fraction(data_availability_df, "macro_corr", "Model-type performance heatmap across train fraction", by_type=True))
    add("Data Availability", "limited_data_regime", "Data-limited regime chart", _gap_by_fraction(data_availability_df[data_availability_df["train_fraction"].notna() & data_availability_df["train_fraction"].le(0.5)], "macro_corr", "Data-limited regime: DNN-SNN macro correlation gap"))

    for metric, title in [
        ("accuracy_per_training_second", "Macro correlation per training second"),
        ("snr_per_training_second", "Macro SNR per training second"),
        ("accuracy_per_1000_parameters", "Macro correlation per 1,000 parameters"),
        ("snr_per_1000_parameters", "Macro SNR per 1,000 parameters"),
        ("accuracy_per_inference_ms", "Macro correlation per millisecond inference time"),
        ("snr_per_inference_ms", "Macro SNR per millisecond inference time"),
    ]:
        add("Trade-offs", f"{metric}_bar", title, _bar_chart(df, metric, title, ascending=False))
    add("Trade-offs", "performance_loss_vs_training_speedup", "Performance loss vs training speedup", _combined_accuracy_loss_vs_efficiency(df))
    add("Trade-offs", "performance_loss_vs_parameter_reduction", "Performance loss vs parameter reduction", _scatter(_performance_loss_table(df, "parameters"), "efficiency_gain", "performance_loss", "Performance loss vs parameter reduction"))
    add("Trade-offs", "performance_loss_vs_inference_speedup", "Performance loss vs inference speedup", _scatter(_performance_loss_table(df, "inference_time_ms"), "efficiency_gain", "performance_loss", "Performance loss vs inference speedup"))
    add("Trade-offs", "cost_normalized_ranking", "Cost-normalized ranking", _cost_normalized_ranking(df))
    add("Trade-offs", "pareto_accuracy_training", "Pareto frontier for accuracy vs training time", _pareto_chart(df, "training_time_s", "macro_corr", "Pareto frontier for accuracy vs training time"))
    add("Trade-offs", "pareto_accuracy_parameters", "Pareto frontier for accuracy vs parameters", _pareto_chart(df, "parameters", "macro_corr", "Pareto frontier for accuracy vs parameters"))
    add("Trade-offs", "pareto_accuracy_inference", "Pareto frontier for accuracy vs inference time", _pareto_chart(df, "inference_time_ms", "macro_corr", "Pareto frontier for accuracy vs inference time"))
    add("Trade-offs", "pareto_accuracy_memory", "Pareto frontier for accuracy vs peak memory", _pareto_chart(df, "peak_memory_mb", "macro_corr", "Pareto frontier for accuracy vs peak memory"))

    summary = compute_cost_summary(chart_dataset)
    add("SNN vs DNN", "best_snn_vs_dnn_bar", "Best SNN vs best DNN summary", _best_snn_dnn_summary(df))
    add("SNN vs DNN", "avg_snn_vs_dnn_bar", "Average SNN vs average DNN summary", _average_snn_dnn_summary(df))
    add("SNN vs DNN", "training_ratio", "SNN/DNN training time ratio", _ratio_chart(summary, "snn_training_speedup_factor", "Training time ratio"))
    add("SNN vs DNN", "parameter_ratio", "SNN/DNN parameter ratio", _ratio_chart(summary, "snn_parameter_reduction_factor", "Parameter ratio"))
    add("SNN vs DNN", "inference_ratio", "SNN/DNN inference time ratio", _ratio_chart(summary, "snn_inference_speedup_factor", "Inference ratio"))
    add("SNN vs DNN", "memory_ratio", "SNN/DNN peak memory ratio", _memory_ratio_chart(df))
    add("SNN vs DNN", "performance_gap", "Performance gap between best SNN and best DNN", _performance_gap_chart(summary))
    add("SNN vs DNN", "efficiency_gain", "Efficiency gain chart", _efficiency_gain_chart(summary))
    add("SNN vs DNN", "combined_accuracy_efficiency", "Accuracy loss versus efficiency gain", _combined_accuracy_loss_vs_efficiency(df))

    return charts


def retest_data_availability_models(
    df: pd.DataFrame,
    component_types: list[str],
    *,
    fs: int,
    duration: float,
    noise_level: float,
    seed: int,
    num_samples: int,
    permutation_invariant: bool = True,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Retest data availability models.
    
    Args:
        df: Input dataframe.
        seed: Random seed for reproducible output.
    """
    if df.empty:
        return pd.DataFrame()

    required_cols = ["model_name", "display_name", "checkpoint_path"]
    available = df.copy()
    for col in required_cols:
        if col not in available.columns:
            return pd.DataFrame()
    available = available[available["checkpoint_path"].notna() & available["checkpoint_path"].astype(str).str.len().gt(0)].copy()
    if available.empty:
        return pd.DataFrame()

    dedupe_cols = [col for col in ["model_name", "display_name", "checkpoint_path", "train_fraction", "train_samples", "model_type", "family"] if col in available.columns]
    available = available.drop_duplicates(subset=dedupe_cols, keep="first")

    rows: list[dict] = []
    component_count = len(component_types)
    for _, model_row in available.iterrows():
        model_key = str(model_row.get("model_name", "")).strip()
        checkpoint = str(model_row.get("checkpoint_path", "")).strip()
        display_name = str(model_row.get("display_name", model_key)).strip()
        row = {
            "model_name": model_key,
            "display_name": display_name,
            "model_type": model_row.get("model_type", ""),
            "family": model_row.get("family", ""),
            "checkpoint_path": checkpoint,
            "train_fraction": _safe_float(model_row.get("train_fraction")),
            "train_samples": _safe_float(model_row.get("train_samples")),
            "status": "ok",
        }
        try:
            model, _ = load_model(model_key=model_key, out_channels=component_count, checkpoint_path=checkpoint)
            macro_corr_samples: list[float] = []
            macro_snr_samples: list[float] = []
            observed_mse_samples: list[float] = []
            for sample_idx in range(num_samples):
                generated = generate_signal(
                    SignalConfig(
                        signal_type="mixed",
                        n_components=component_count,
                        duration=duration,
                        fs=fs,
                        noise_level=noise_level,
                        seed=seed + sample_idx,
                        generation_mode=PURE_MODE,
                        selected_component_types=component_types,
                    )
                )
                prediction, _ = run_inference(model, generated.mixture)
                if prediction.shape != generated.components.shape:
                    raise ValueError(f"prediction shape {prediction.shape} does not match target shape {generated.components.shape}")
                report = evaluate_prediction(
                    generated.components,
                    prediction,
                    generated.mixture,
                    permutation_invariant=permutation_invariant,
                )
                macro = report.get("macro_average", {})
                observed = report.get("observed_mixture_metrics", {})
                macro_corr_samples.append(float(macro.get("corr", np.nan)))
                macro_snr_samples.append(float(macro.get("snr_db", np.nan)))
                observed_mse_samples.append(float(observed.get("mse", np.nan)))

            corr_low, corr_high = _bootstrap_mean_ci(macro_corr_samples, confidence=confidence, seed=seed + 101)
            snr_low, snr_high = _bootstrap_mean_ci(macro_snr_samples, confidence=confidence, seed=seed + 202)
            mse_low, mse_high = _bootstrap_mean_ci(observed_mse_samples, confidence=confidence, seed=seed + 303)
            row.update(
                {
                    "macro_corr": float(np.nanmean(macro_corr_samples)),
                    "macro_corr_ci95_low": corr_low,
                    "macro_corr_ci95_high": corr_high,
                    "macro_snr": float(np.nanmean(macro_snr_samples)),
                    "macro_snr_ci95_low": snr_low,
                    "macro_snr_ci95_high": snr_high,
                    "observed_mse": float(np.nanmean(observed_mse_samples)),
                    "observed_mse_ci95_low": mse_low,
                    "observed_mse_ci95_high": mse_high,
                    "eval_samples": int(num_samples),
                }
            )
        except Exception as exc:
            row.update(
                {
                    "macro_corr": np.nan,
                    "macro_corr_ci95_low": np.nan,
                    "macro_corr_ci95_high": np.nan,
                    "macro_snr": np.nan,
                    "macro_snr_ci95_low": np.nan,
                    "macro_snr_ci95_high": np.nan,
                    "observed_mse": np.nan,
                    "observed_mse_ci95_low": np.nan,
                    "observed_mse_ci95_high": np.nan,
                    "eval_samples": int(num_samples),
                    "status": f"failed: {exc}",
                }
            )
        rows.append(row)

    return pd.DataFrame(rows)


def _performance_loss_table(df: pd.DataFrame, efficiency_column: str) -> pd.DataFrame:
    best_dnn = best_row(df[df["model_type"] == "DNN"], "macro_corr")
    snn_df = df[(df["model_type"] == "SNN") & df["macro_corr"].notna()].copy()
    if best_dnn is None or snn_df.empty:
        return pd.DataFrame(columns=["efficiency_gain", "performance_loss", "display_name", "model_type"])
    baseline = _safe_float(best_dnn.get(efficiency_column))
    if baseline <= 0:
        return pd.DataFrame(columns=["efficiency_gain", "performance_loss", "display_name", "model_type"])
    snn_df = snn_df.dropna(subset=[efficiency_column])
    if snn_df.empty:
        return pd.DataFrame(columns=["efficiency_gain", "performance_loss", "display_name", "model_type"])
    snn_df["efficiency_gain"] = baseline / snn_df[efficiency_column].clip(lower=1e-8)
    snn_df["performance_loss"] = float(best_dnn["macro_corr"]) - snn_df["macro_corr"]
    return snn_df


def _best_snn_dnn_summary(df: pd.DataFrame) -> plt.Figure | None:
    best_snn = best_row(df[df["model_type"] == "SNN"], "macro_corr")
    best_dnn = best_row(df[df["model_type"] == "DNN"], "macro_corr")
    if best_snn is None or best_dnn is None:
        return None
    compare = pd.DataFrame(
        [
            {"label": "Best SNN", "macro_corr": best_snn["macro_corr"], "macro_snr": best_snn.get("macro_snr"), "training_time_s": best_snn.get("training_time_s")},
            {"label": "Best DNN", "macro_corr": best_dnn["macro_corr"], "macro_snr": best_dnn.get("macro_snr"), "training_time_s": best_dnn.get("training_time_s")},
        ]
    )
    fig, ax = _base_axis((9.2, 4.6))
    x = np.arange(len(compare))
    width = 0.25
    for idx, metric in enumerate(["macro_corr", "macro_snr", "training_time_s"]):
        if compare[metric].notna().any():
            ax.bar(x + idx * width - width, compare[metric], width=width, label=COST_METRIC_LABELS.get(metric, metric))
    ax.set_xticks(x)
    ax.set_xticklabels(compare["label"])
    ax.set_title("Best SNN vs best DNN summary")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def _average_snn_dnn_summary(df: pd.DataFrame) -> plt.Figure | None:
    subset = df[df["model_type"].isin(["SNN", "DNN"])].copy()
    if subset.empty:
        return None
    grouped = subset.groupby("model_type")[["macro_corr", "macro_snr", "training_time_s"]].mean().reset_index()
    fig, ax = _base_axis((9.2, 4.6))
    x = np.arange(len(grouped))
    width = 0.25
    for idx, metric in enumerate(["macro_corr", "macro_snr", "training_time_s"]):
        if grouped[metric].notna().any():
            ax.bar(x + idx * width - width, grouped[metric], width=width, label=COST_METRIC_LABELS.get(metric, metric))
    ax.set_xticks(x)
    ax.set_xticklabels(grouped["model_type"])
    ax.set_title("Average SNN vs average DNN summary")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def _memory_ratio_chart(df: pd.DataFrame) -> plt.Figure | None:
    best_snn = best_row(df[df["model_type"] == "SNN"], "macro_corr")
    best_dnn = best_row(df[df["model_type"] == "DNN"], "macro_corr")
    if best_snn is None or best_dnn is None:
        return None
    ratio = _ratio(best_dnn.get("peak_memory_mb"), best_snn.get("peak_memory_mb"))
    if ratio is None:
        return None
    return _ratio_chart({"snn_memory_ratio": ratio}, "snn_memory_ratio", "Peak memory ratio")


def _performance_gap_chart(summary: dict[str, object]) -> plt.Figure | None:
    gap = summary.get("absolute_performance_gap")
    if gap is None:
        return None
    fig, ax = _base_axis((6.0, 4.0))
    ax.bar(["Best DNN - Best SNN"], [gap], color="#9467bd")
    ax.set_ylabel("Macro correlation gap")
    ax.set_title("Performance gap")
    ax.text(0, gap, f"{gap:.3f}", ha="center", va="bottom")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def _efficiency_gain_chart(summary: dict[str, object]) -> plt.Figure | None:
    rows = []
    for label, key in [
        ("Training", "snn_training_speedup_factor"),
        ("Inference", "snn_inference_speedup_factor"),
        ("Parameters", "snn_parameter_reduction_factor"),
    ]:
        value = summary.get(key)
        if value is not None:
            rows.append((label, value))
    if not rows:
        return None
    fig, ax = _base_axis((7.0, 4.2))
    labels, values = zip(*rows)
    ax.bar(labels, values, color="#17becf")
    ax.set_ylabel("Factor (DNN / SNN)")
    ax.set_title("Efficiency gain chart")
    for idx, value in enumerate(values):
        ax.text(idx, value, f"{value:.2f}x", ha="center", va="bottom")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig
