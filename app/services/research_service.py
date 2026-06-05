from __future__ import annotations

import json
import math
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from textwrap import fill
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, LogLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

plt.rcParams["figure.max_open_warning"] = 0


MODEL_TYPE_ORDER = ["SNN", "DNN", "Classical", "Unknown"]
MODEL_TYPE_COLORS = {
    "SNN": "#1f77b4",
    "DNN": "#d62728",
    "Classical": "#2ca02c",
    "Unknown": "#7f7f7f",
}
MODEL_TYPE_MARKERS = {
    "SNN": "o",
    "DNN": "s",
    "Classical": "^",
    "Unknown": "D",
}

RESEARCH_EXPORT_SUBDIR = "research_dashboard"
DEFAULT_EXPORT_FORMATS = ("png", "svg", "pdf")
CHART_REFERENCE_TRAIN_SAMPLES = 8000

METRIC_COLUMNS = {
    "macro_corr": "Macro correlation",
    "macro_snr": "Macro SNR (dB)",
    "test_loss": "Test loss",
    "parameters": "Parameter count",
    "inference_time_ms": "Inference time (ms)",
    "training_time_s": "Training time (s)",
}

PAPER_MODEL_LABELS = {
    "unet1d": "U-Net 1D",
    "rnnbased": "RNN",
    "autoencoderbased": "Autoencoder",
    "tasnet": "TasNet",
    "sepformer": "SepFormer",
    "conv1dnetwork": "Shallow Conv1D",
    "mlp_singlehead": "Single-Head MLP",
    "fuse_shallow": "Fuse Shallow",
    "multiscale_dilated": "Multiscale Dilated",
    "multiscale_branches": "Multiscale Branches",
    "multiple_head_multiscale_branches": "Multi-Head Multiscale Branches",
    "attention_stem_multiple_head_multiscale_branches": "Attention Stem + Multi-Head + Multiscale",
    "attention_stem_bilinear_fusion_multiple_head_multiscale_branches": "Attention Stem + Bilinear Fusion + Multi-Head + Multiscale",
    "attention_stem_multi_head_multiscale_tcn": "Attention Stem + Multi-Head + Multiscale TCN",
    "attention_stem_multi_head_multiscale_tcn_inference_optimized": "Attention Stem + Multi-Head + Multiscale TCN (Fast)",
}

SIGNAL_CONDITION_COLUMNS = {
    "noise_level": "Noise level",
    "overlap_level": "Frequency overlap",
    "nonstationarity_level": "Nonstationarity level",
    "num_components": "Number of components",
}

KNOWN_MODULES = [
    "attention_stem",
    "branch_attention",
    "multiscale_branches",
    "feature_fusion",
    "multiple_heads",
    "bilinear_fusion",
    "dilated_temporal_convs",
    "residual_connections",
    "tcn_backbone",
    "frequency_features",
]

MODULE_LABELS = {
    "attention_stem": "Attention stem",
    "branch_attention": "Branch attention",
    "multiscale_branches": "Multiscale branches",
    "feature_fusion": "1x1 fusion",
    "multiple_heads": "Multiple heads",
    "bilinear_fusion": "Bilinear fusion",
    "dilated_temporal_convs": "Dilated temporal convs",
    "residual_connections": "Residual connections",
    "tcn_backbone": "TCN backbone",
    "frequency_features": "Frequency features",
}

MODULE_LABEL_TO_KEY = {label: key for key, label in MODULE_LABELS.items()}


@dataclass
class ResearchDataset:
    """Dataset wrapper."""
    dataframe: pd.DataFrame
    warnings: list[str]
    module_columns: list[str]
    source_labels: list[str]


@dataclass
class ChartSpec:
    """Figure plus section metadata for dashboard/export use."""
    section: str
    slug: str
    title: str
    figure: plt.Figure


def _safe_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _coerce_numeric(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    for column in columns:
        if column not in df.columns:
            df[column] = pd.NA
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def _first_non_empty(row: pd.Series, columns: Iterable[str], default: str = "") -> str:
    for column in columns:
        if column not in row.index:
            continue
        value = row.get(column)
        if pd.isna(value):
            continue
        text = str(value).strip()
        if text:
            return text
    return default


def _paper_label_for_row(row: pd.Series) -> str:
    model_key = str(row.get("model_name", "")).strip().lower()
    if model_key in PAPER_MODEL_LABELS:
        return PAPER_MODEL_LABELS[model_key]
    return _first_non_empty(row, ["display_name", "model_name"], default=model_key or "model")


def _normalize_model_type(value: str) -> str:
    key = str(value).strip().lower()
    if key in {"snn", "shallow"}:
        return "SNN"
    if key in {"dnn", "deep"}:
        return "DNN"
    if key in {"classical", "baseline"}:
        return "Classical"
    return "Unknown"


def _normalize_modules(value) -> list[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, tuple):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    except json.JSONDecodeError:
        pass
    separators = [",", ";", "|"]
    for separator in separators:
        if separator in text:
            return [part.strip() for part in text.split(separator) if part.strip()]
    return [text]


def _infer_modules_from_name(model_name: str) -> tuple[list[str], bool]:
    key = str(model_name).strip().lower()
    mapping = {
        "multiscale_branches": ["multiscale_branches", "feature_fusion"],
        "multiple_head_multiscale_branches": ["multiscale_branches", "feature_fusion", "multiple_heads"],
        "attention_stem_multiple_head_multiscale_branches": [
            "attention_stem",
            "branch_attention",
            "multiscale_branches",
            "feature_fusion",
            "multiple_heads",
        ],
        "attention_stem_bilinear_fusion_multiple_head_multiscale_branches": [
            "attention_stem",
            "branch_attention",
            "multiscale_branches",
            "multiple_heads",
            "bilinear_fusion",
        ],
        "attention_stem_multi_head_multiscale_tcn": [
            "attention_stem",
            "branch_attention",
            "multiscale_branches",
            "feature_fusion",
            "multiple_heads",
            "dilated_temporal_convs",
            "residual_connections",
            "tcn_backbone",
        ],
        "multiscale_dilated": ["dilated_temporal_convs", "multiscale_branches"],
        "fuse_shallow": ["multiscale_branches", "feature_fusion"],
    }
    if key in mapping:
        return mapping[key], True
    if "attention_stem_multi_head_multiscale_tcn" in key:
        return mapping["attention_stem_multi_head_multiscale_tcn"], True
    return [], False


def _infer_model_type(row: pd.Series) -> tuple[str, bool]:
    explicit = _normalize_model_type(_first_non_empty(row, ["model_type"]))
    if explicit != "Unknown":
        return explicit, False

    depth_label = _normalize_model_type(_first_non_empty(row, ["depth_label"]))
    if depth_label != "Unknown":
        return depth_label, True

    family = _first_non_empty(row, ["family", "architecture"]).lower()
    if family in {"conv1d", "mlp", "shallow"}:
        return "SNN", True
    if family in {"unet", "rnn", "autoencoder", "tasnet", "sepformer"}:
        return "DNN", True
    if family in {"classical", "baseline"}:
        return "Classical", True

    model_name = _first_non_empty(row, ["model_name", "display_name"]).lower()
    shallow_tokens = ["shallow", "conv1d", "multiscale", "attention_stem", "tcn"]
    deep_tokens = ["tasnet", "sepformer", "unet", "rnn", "autoencoder"]
    if any(token in model_name for token in deep_tokens):
        return "DNN", True
    if any(token in model_name for token in shallow_tokens):
        return "SNN", True
    return "Unknown", False


def _parse_report_payload(payload: dict, source_name: str) -> dict:
    summary = payload.get("test_summary", {}) if isinstance(payload, dict) else {}
    macro = summary.get("macro_average", {}) if isinstance(summary, dict) else {}
    metadata = payload.get("training_metadata", {}) if isinstance(payload, dict) else {}
    eval_config = payload.get("eval_config", {}) if isinstance(payload, dict) else {}
    component_names = payload.get("component_names", [])
    row = {
        "model_name": payload.get("model_name") or payload.get("model_key") or Path(source_name).stem,
        "display_name": payload.get("display_name") or payload.get("model_name") or Path(source_name).stem,
        "test_loss": _safe_float(payload.get("test_loss")),
        "macro_corr": _safe_float(macro.get("corr")),
        "macro_snr": _safe_float(macro.get("snr_db")),
        "training_time_s": _safe_float(metadata.get("elapsed_seconds")),
        "train_samples": metadata.get("train_samples"),
        "val_samples": metadata.get("val_samples"),
        "test_samples": metadata.get("test_samples"),
        "total_samples": metadata.get("total_samples"),
        "epochs": metadata.get("epochs"),
        "signal_length": metadata.get("signal_length") or eval_config.get("signal_length"),
        "fs": metadata.get("fs") or eval_config.get("fs"),
        "num_components": len(component_names) if isinstance(component_names, list) else pd.NA,
        "report_source": source_name,
        "report_path": source_name,
        "config_name": payload.get("config_name", ""),
    }
    if "parameters" in payload:
        row["parameters"] = _safe_float(payload.get("parameters"))
    if "inference_time_ms" in payload:
        row["inference_time_ms"] = _safe_float(payload.get("inference_time_ms"))
    return row


def _load_uploaded_frame(name: str, data: bytes) -> tuple[pd.DataFrame | None, str | None]:
    suffix = Path(name).suffix.lower()
    try:
        if suffix == ".csv":
            return pd.read_csv(BytesIO(data)), None
        if suffix == ".json":
            payload = json.loads(data.decode("utf-8"))
            if isinstance(payload, list):
                return pd.DataFrame(payload), None
            if isinstance(payload, dict) and "test_summary" in payload:
                return pd.DataFrame([_parse_report_payload(payload, source_name=name)]), None
            if isinstance(payload, dict) and "models" in payload and isinstance(payload["models"], list):
                return pd.DataFrame(payload["models"]), None
            return pd.DataFrame([payload]), None
    except Exception as exc:
        return None, f"Failed to parse uploaded file {name}: {exc}"
    return None, f"Unsupported file type for {name}. Use CSV or JSON."


def prepare_research_dataset(
    reports_df: pd.DataFrame,
    registry: list[dict] | None,
    uploaded_files: list | None = None,
) -> ResearchDataset:
    """Prepare research dataset.
    
    Args:
        reports_df: Evaluation reports table.
        registry: Model registry entries.
        uploaded_files: Files uploaded through Streamlit.
    """
    warnings: list[str] = []
    source_labels: list[str] = []
    frames: list[pd.DataFrame] = []

    if reports_df is not None and not reports_df.empty:
        frames.append(reports_df.copy())
        source_labels.append("discovered evaluation reports")

    for uploaded in uploaded_files or []:
        frame, error = _load_uploaded_frame(uploaded.name, uploaded.getvalue())
        if error:
            warnings.append(error)
            continue
        if frame is not None and not frame.empty:
            frames.append(frame)
            source_labels.append(f"upload:{uploaded.name}")

    if not frames:
        return ResearchDataset(
            dataframe=pd.DataFrame(),
            warnings=["No research result files were available."],
            module_columns=[],
            source_labels=[],
        )

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined = _normalize_research_dataframe(combined, registry or [], warnings)
    module_columns = [column for column in KNOWN_MODULES if column in combined.columns]
    return ResearchDataset(
        dataframe=combined,
        warnings=warnings,
        module_columns=module_columns,
        source_labels=source_labels,
    )


def _normalize_research_dataframe(df: pd.DataFrame, registry: list[dict], warnings: list[str]) -> pd.DataFrame:
    normalized = df.copy()
    rename_map = {
        "macro_snr_db": "macro_snr",
        "parameter_count": "parameters",
        "training_time_sec": "training_time_s",
    }
    normalized = normalized.rename(columns={k: v for k, v in rename_map.items() if k in normalized.columns})

    registry_df = pd.DataFrame(registry) if registry else pd.DataFrame()
    if not registry_df.empty:
        registry_df = registry_df.rename(columns={"key": "model_name"})
        merge_columns = [column for column in ["model_name", "display_name", "family", "depth_label", "model_key", "notes"] if column in registry_df.columns]
        normalized = normalized.merge(
            registry_df[merge_columns].drop_duplicates(subset=["model_name"]),
            on="model_name",
            how="left",
            suffixes=("", "_registry"),
        )
        for column in ["display_name", "family", "depth_label"]:
            registry_column = f"{column}_registry"
            if registry_column in normalized.columns:
                normalized[column] = normalized[column].fillna(normalized[registry_column])
                normalized = normalized.drop(columns=[registry_column])

    normalized["model_name"] = normalized.get("model_name", pd.Series(dtype=object)).fillna("").astype(str)
    if "display_name" not in normalized.columns:
        normalized["display_name"] = normalized["model_name"]
    normalized["display_name"] = normalized["display_name"].fillna(normalized["model_name"]).astype(str)
    if "family" not in normalized.columns:
        normalized["family"] = "unknown"
    normalized["family"] = normalized["family"].fillna("unknown").astype(str)
    if "architecture" not in normalized.columns:
        normalized["architecture"] = normalized["family"]
    normalized["architecture"] = normalized["architecture"].fillna(normalized["family"]).astype(str)
    if "config_name" not in normalized.columns:
        normalized["config_name"] = ""

    numeric_columns = [
        "test_loss",
        "macro_corr",
        "macro_snr",
        "parameters",
        "inference_time_ms",
        "training_time_s",
        "noise_level",
        "overlap_level",
        "nonstationarity_level",
        "num_components",
        "seed",
        "epochs",
        "train_samples",
        "val_samples",
        "test_samples",
        "total_samples",
    ]
    normalized = _coerce_numeric(normalized, numeric_columns)

    model_types: list[str] = []
    inferred_model_type_count = 0
    module_inference_count = 0
    parsed_modules: list[list[str]] = []
    enabled_modules_text: list[str] = []
    for _, row in normalized.iterrows():
        model_type, inferred = _infer_model_type(row)
        model_types.append(model_type)
        if inferred:
            inferred_model_type_count += 1

        modules = _normalize_modules(row.get("enabled_modules"))
        inferred_modules, inferred_from_name = _infer_modules_from_name(_first_non_empty(row, ["model_name", "display_name"]))
        if not modules and inferred_from_name:
            modules = inferred_modules
            module_inference_count += 1
        parsed_modules.append(modules)
        enabled_modules_text.append(", ".join(sorted(modules)))

    normalized["model_type"] = model_types
    normalized["enabled_modules"] = enabled_modules_text

    if inferred_model_type_count:
        warnings.append(
            f"Inferred model_type for {inferred_model_type_count} rows from registry metadata or model naming."
        )
    if module_inference_count:
        warnings.append(
            f"Inferred enabled_modules for {module_inference_count} rows from known shallow-model naming patterns."
        )

    for module in KNOWN_MODULES:
        values = []
        for modules in parsed_modules:
            if modules:
                values.append(module in modules)
            else:
                values.append(pd.NA)
        normalized[module] = values

    module_frame = normalized[[module for module in KNOWN_MODULES if module in normalized.columns]].copy()
    for column in module_frame.columns:
        module_frame[column] = module_frame[column].astype("boolean").fillna(False).astype(int)
    normalized["module_count"] = module_frame.sum(axis=1)
    normalized["model_group"] = normalized["model_type"].where(normalized["model_type"] != "Unknown", normalized["depth_label"].fillna("unknown"))
    normalized["macro_corr_rank"] = normalized["macro_corr"].rank(ascending=False, method="min")

    signature_columns = [column for column in ["model_name", "config_name", "report_path"] if column in normalized.columns]
    if signature_columns:
        completeness = normalized.notna().sum(axis=1)
        normalized["_completeness"] = completeness
        normalized = (
            normalized.sort_values("_completeness", ascending=False)
            .drop_duplicates(subset=signature_columns, keep="first")
            .drop(columns=["_completeness"])
        )

    return normalized.reset_index(drop=True)


def available_columns(df: pd.DataFrame, columns: Iterable[str]) -> list[str]:
    """Available columns.
    
    Args:
        df: Input dataframe.
        columns: Columns to keep or display.
    """
    return [column for column in columns if column in df.columns and df[column].notna().any()]


def prepare_chart_generation_dataframe(
    df: pd.DataFrame,
    *,
    target_train_samples: int = CHART_REFERENCE_TRAIN_SAMPLES,
    dedupe_metric: str = "macro_corr",
) -> pd.DataFrame:
    """Prepare chart generation dataframe.
    
    Args:
        df: Input dataframe.
        target_train_samples: Preferred training sample count for deduplication.
        dedupe_metric: Metric used to choose one row per model.
    """
    if df.empty:
        return df.copy()

    prepared = df.copy()

    label_column = "display_name" if "display_name" in prepared.columns else "model_name"
    if label_column not in prepared.columns:
        return prepared.reset_index(drop=True)

    if "train_samples" in prepared.columns and prepared["train_samples"].notna().any():
        train_samples_numeric = pd.to_numeric(prepared["train_samples"], errors="coerce")
        prepared["_train_samples_numeric"] = train_samples_numeric
        group_best_rows: list[pd.DataFrame] = []
        for _, group in prepared.groupby(label_column, dropna=False):
            exact = group[group["_train_samples_numeric"].eq(target_train_samples)]
            if not exact.empty:
                group_best_rows.append(exact.copy())
                continue
            around_4000 = group[group["_train_samples_numeric"].between(3500, 5000, inclusive="both")]
            if not around_4000.empty:
                group_best_rows.append(around_4000.copy())
                continue
            with_samples = group[group["_train_samples_numeric"].notna()]
            if not with_samples.empty:
                max_samples = with_samples["_train_samples_numeric"].max()
                group_best_rows.append(with_samples[with_samples["_train_samples_numeric"].eq(max_samples)].copy())
            else:
                group_best_rows.append(group.copy())
        prepared = pd.concat(group_best_rows, ignore_index=True, sort=False)

    sort_columns: list[str] = []
    ascending: list[bool] = []
    for column, is_ascending in [
        (dedupe_metric, False),
        ("macro_snr", False),
        ("test_loss", True),
        ("train_samples", False),
    ]:
        if column in prepared.columns and prepared[column].notna().any():
            sort_columns.append(column)
            ascending.append(is_ascending)

    prepared["_completeness"] = prepared.notna().sum(axis=1)
    sort_columns.append("_completeness")
    ascending.append(False)

    prepared = (
        prepared.sort_values(sort_columns, ascending=ascending, na_position="last")
        .drop_duplicates(subset=[label_column], keep="first")
        .drop(columns=[col for col in ["_completeness", "_train_samples_numeric"] if col in prepared.columns])
        .reset_index(drop=True)
    )
    return prepared


def available_module_columns(df: pd.DataFrame) -> list[str]:
    """Available module columns.
    
    Args:
        df: Input dataframe.
    """
    return [column for column in KNOWN_MODULES if column in df.columns and df[column].notna().any()]


def module_display_names(module_columns: Iterable[str]) -> list[str]:
    """Module display names.
    
    Args:
        module_columns: Columns to read or display.
    """
    return [MODULE_LABELS.get(column, column) for column in module_columns]


def apply_ablation_scope(
    df: pd.DataFrame,
    *,
    selected_models: list[str] | None = None,
    selected_modules: list[str] | None = None,
    require_selected_module_presence: bool = False,
) -> pd.DataFrame:
    """Apply ablation scope.
    
    Args:
        df: Input dataframe.
    """
    scoped = df.copy()
    if scoped.empty:
        return scoped

    if selected_models:
        scoped = scoped[
            scoped["display_name"].isin(selected_models)
            | scoped["model_name"].isin(selected_models)
        ].copy()

    available_modules = available_module_columns(scoped)
    if selected_modules:
        selected_module_keys = [
            MODULE_LABEL_TO_KEY.get(module, module)
            for module in selected_modules
            if MODULE_LABEL_TO_KEY.get(module, module) in available_modules
        ]
        unselected_modules = [module for module in available_modules if module not in selected_module_keys]
        for module in unselected_modules:
            scoped[module] = pd.NA
        if require_selected_module_presence and selected_module_keys:
            presence_mask = scoped[selected_module_keys].eq(True).any(axis=1)
            scoped = scoped[presence_mask].copy()
        if "enabled_modules" in scoped.columns:
            def _selected_enabled_modules(value: str) -> str:
                parsed = _normalize_modules(value)
                filtered = [module for module in parsed if module in selected_module_keys]
                return ", ".join(filtered)

            scoped["enabled_modules"] = scoped["enabled_modules"].apply(_selected_enabled_modules)
        module_frame = scoped[[module for module in selected_module_keys if module in scoped.columns]].copy()
        if not module_frame.empty:
            for column in module_frame.columns:
                module_frame[column] = module_frame[column].astype("boolean").fillna(False).astype(int)
            scoped["module_count"] = module_frame.sum(axis=1)

    return scoped.reset_index(drop=True)


def best_row(df: pd.DataFrame, metric: str, *, higher_is_better: bool = True) -> pd.Series | None:
    """Best row.
    
    Args:
        df: Input dataframe.
        metric: Metric column or metric name to use.
        higher_is_better: Whether larger values should rank first.
    """
    if metric not in df.columns:
        return None
    subset = df.dropna(subset=[metric])
    if subset.empty:
        return None
    index = subset[metric].idxmax() if higher_is_better else subset[metric].idxmin()
    return subset.loc[index]


def group_summary(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Group summary.
    
    Args:
        df: Input dataframe.
        metric: Metric column or metric name to use.
    """
    if metric not in df.columns:
        return pd.DataFrame()
    subset = df.dropna(subset=[metric, "model_type"])
    if subset.empty:
        return pd.DataFrame()
    grouped = subset.groupby("model_type", dropna=False)[metric].mean().reset_index()
    grouped["model_type"] = pd.Categorical(grouped["model_type"], categories=MODEL_TYPE_ORDER, ordered=True)
    return grouped.sort_values("model_type")


def compute_summary_statistics(df: pd.DataFrame) -> dict[str, object]:
    """Compute summary statistics.
    
    Args:
        df: Input dataframe.
    """
    summary: dict[str, object] = {}
    snn_df = df[df["model_type"] == "SNN"]
    dnn_df = df[df["model_type"] == "DNN"]
    classical_df = df[df["model_type"] == "Classical"]

    best_snn_corr = best_row(snn_df, "macro_corr")
    best_dnn_corr = best_row(dnn_df, "macro_corr")
    best_snn_snr = best_row(snn_df, "macro_snr")
    best_dnn_snr = best_row(dnn_df, "macro_snr")

    summary["best_snn_macro_corr"] = None if best_snn_corr is None else float(best_snn_corr["macro_corr"])
    summary["best_dnn_macro_corr"] = None if best_dnn_corr is None else float(best_dnn_corr["macro_corr"])
    summary["best_snn_macro_snr"] = None if best_snn_snr is None else float(best_snn_snr["macro_snr"])
    summary["best_dnn_macro_snr"] = None if best_dnn_snr is None else float(best_dnn_snr["macro_snr"])

    if best_snn_corr is not None and best_dnn_corr is not None:
        snn_corr = float(best_snn_corr["macro_corr"])
        dnn_corr = float(best_dnn_corr["macro_corr"])
        summary["absolute_performance_gap"] = dnn_corr - snn_corr
        summary["relative_performance_gap"] = (dnn_corr - snn_corr) / max(abs(dnn_corr), 1e-8)

        snn_params = _safe_float(best_snn_corr.get("parameters"))
        dnn_params = _safe_float(best_dnn_corr.get("parameters"))
        snn_infer = _safe_float(best_snn_corr.get("inference_time_ms"))
        dnn_infer = _safe_float(best_dnn_corr.get("inference_time_ms"))
        snn_train = _safe_float(best_snn_corr.get("training_time_s"))
        dnn_train = _safe_float(best_dnn_corr.get("training_time_s"))

        summary["parameter_reduction_factor"] = dnn_params / snn_params if snn_params > 0 and dnn_params > 0 else None
        summary["inference_speedup_factor"] = dnn_infer / snn_infer if snn_infer > 0 and dnn_infer > 0 else None
        summary["training_speedup_factor"] = dnn_train / snn_train if snn_train > 0 and dnn_train > 0 else None

    pareto = compute_pareto_frontier(df, metric="macro_corr")
    summary["pareto_optimal_models"] = pareto["display_name"].tolist() if not pareto.empty else []
    best_tradeoff = best_accuracy_efficiency_tradeoff(df)
    summary["best_tradeoff_model"] = None if best_tradeoff is None else best_tradeoff.get("display_name")
    summary["best_classical_model"] = None if classical_df.empty else best_row(classical_df, "macro_corr").get("display_name")
    return summary


def best_accuracy_efficiency_tradeoff(df: pd.DataFrame) -> pd.Series | None:
    """Best accuracy efficiency tradeoff.
    
    Args:
        df: Input dataframe.
    """
    required = [column for column in ["macro_corr", "parameters", "inference_time_ms"] if column in df.columns]
    subset = df.dropna(subset=required)
    if subset.empty:
        return best_row(df, "macro_corr")

    scored = subset.copy()
    scored["corr_score"] = _minmax_series(scored["macro_corr"])
    scored["param_score"] = 1.0 - _minmax_series(np.log10(scored["parameters"].clip(lower=1.0)))
    scored["infer_score"] = 1.0 - _minmax_series(np.log10(scored["inference_time_ms"].clip(lower=1e-6)))
    if "training_time_s" in scored.columns and scored["training_time_s"].notna().any():
        scored["train_score"] = 1.0 - _minmax_series(np.log10(scored["training_time_s"].clip(lower=1e-6)))
        score_columns = ["corr_score", "param_score", "infer_score", "train_score"]
    else:
        score_columns = ["corr_score", "param_score", "infer_score"]
    scored["tradeoff_score"] = scored[score_columns].mean(axis=1)
    return scored.loc[scored["tradeoff_score"].idxmax()]


def _minmax_series(series: pd.Series) -> pd.Series:
    if series.nunique(dropna=True) <= 1:
        return pd.Series(np.ones(len(series)), index=series.index)
    minimum = series.min()
    maximum = series.max()
    return (series - minimum) / (maximum - minimum)


def compute_pareto_frontier(df: pd.DataFrame, metric: str = "macro_corr") -> pd.DataFrame:
    """Compute pareto frontier.
    
    Args:
        df: Input dataframe.
        metric: Metric column or metric name to use.
    """
    required = [metric, "parameters", "inference_time_ms"]
    subset = df.dropna(subset=[column for column in required if column in df.columns]).copy()
    if subset.empty or metric not in subset.columns:
        return pd.DataFrame(columns=df.columns)

    keep_indices: list[int] = []
    for idx, row in subset.iterrows():
        dominated = False
        for other_idx, other in subset.iterrows():
            if idx == other_idx:
                continue
            better_metric = other[metric] >= row[metric]
            lower_params = other["parameters"] <= row["parameters"]
            lower_infer = other["inference_time_ms"] <= row["inference_time_ms"]
            strictly_better = (
                other[metric] > row[metric]
                or other["parameters"] < row["parameters"]
                or other["inference_time_ms"] < row["inference_time_ms"]
            )
            if better_metric and lower_params and lower_infer and strictly_better:
                dominated = True
                break
        if not dominated:
            keep_indices.append(idx)
    return subset.loc[keep_indices].sort_values(metric, ascending=False)


def build_results_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build results tables.
    
    Args:
        df: Input dataframe.
    """
    tables: dict[str, pd.DataFrame] = {}
    sort_desc = df.sort_values("macro_corr", ascending=False, na_position="last")
    tables["full_results"] = sort_desc
    tables["snn_only"] = sort_desc[sort_desc["model_type"] == "SNN"]
    tables["dnn_only"] = sort_desc[sort_desc["model_type"] == "DNN"]
    tables["ablation_results"] = sort_desc[sort_desc["module_count"] > 0]
    efficiency_columns = [column for column in ["display_name", "model_type", "parameters", "inference_time_ms", "training_time_s"] if column in sort_desc.columns]
    tables["efficiency"] = sort_desc[efficiency_columns].dropna(how="all", subset=[col for col in efficiency_columns if col not in {"display_name", "model_type"}])
    best_metric_rows = []
    for metric, higher_is_better in [("macro_corr", True), ("macro_snr", True), ("test_loss", False), ("parameters", False), ("inference_time_ms", False)]:
        row = best_row(sort_desc, metric, higher_is_better=higher_is_better)
        if row is None:
            continue
        best_metric_rows.append(
            {
                "metric": metric,
                "display_name": row.get("display_name"),
                "model_type": row.get("model_type"),
                "value": row.get(metric),
            }
        )
    tables["best_per_metric"] = pd.DataFrame(best_metric_rows)
    tables["pareto_optimal"] = compute_pareto_frontier(sort_desc)
    return tables


def compute_module_contributions(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Compute module contributions.
    
    Args:
        df: Input dataframe.
        metric: Metric column or metric name to use.
    """
    module_columns = available_module_columns(df)
    if not module_columns or metric not in df.columns:
        return pd.DataFrame()

    rows = []
    subset = df.dropna(subset=[metric]).copy()
    for module in module_columns:
        with_module = subset[subset[module] == True]  # noqa: E712
        without_module = subset[subset[module] == False]  # noqa: E712
        if with_module.empty or without_module.empty:
            continue
        mean_with = with_module[metric].mean()
        mean_without = without_module[metric].mean()
        param_gain = with_module["parameters"].mean() - without_module["parameters"].mean() if "parameters" in subset.columns else np.nan
        infer_gain = with_module["inference_time_ms"].mean() - without_module["inference_time_ms"].mean() if "inference_time_ms" in subset.columns else np.nan
        rows.append(
            {
                "module": module,
                "module_label": MODULE_LABELS.get(module, module),
                f"{metric}_with_module": mean_with,
                f"{metric}_without_module": mean_without,
                f"{metric}_gain": mean_with - mean_without,
                "parameter_cost": param_gain,
                "inference_time_cost_ms": infer_gain,
                f"{metric}_gain_per_1000_parameters": (mean_with - mean_without) / (param_gain / 1000.0) if pd.notna(param_gain) and param_gain > 0 else np.nan,
                f"{metric}_gain_per_ms": (mean_with - mean_without) / infer_gain if pd.notna(infer_gain) and infer_gain > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(f"{metric}_gain", ascending=False)


def compute_ablation_removal_impact(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Compute ablation removal impact.
    
    Args:
        df: Input dataframe.
        metric: Metric column or metric name to use.
    """
    module_columns = available_module_columns(df)
    if not module_columns:
        return pd.DataFrame()
    subset = df.dropna(subset=[metric]).copy()
    if subset.empty:
        return pd.DataFrame()
    reference = subset.sort_values(["module_count", metric], ascending=[False, False]).iloc[0]
    rows = []
    for module in module_columns:
        candidates = subset[subset[module] == False]  # noqa: E712
        if candidates.empty or reference.get(module) is not True:
            continue
        best_candidate = candidates.sort_values(metric, ascending=False).iloc[0]
        metric_drop = float(reference[metric]) - float(best_candidate[metric])
        param_cost = _safe_float(reference.get("parameters")) - _safe_float(best_candidate.get("parameters"))
        infer_cost = _safe_float(reference.get("inference_time_ms")) - _safe_float(best_candidate.get("inference_time_ms"))
        rows.append(
            {
                "module": module,
                "module_label": MODULE_LABELS.get(module, module),
                "reference_model": reference.get("display_name"),
                "comparison_model": best_candidate.get("display_name"),
                f"{metric}_drop": metric_drop,
                "parameter_cost": param_cost,
                "inference_time_cost_ms": infer_cost,
                f"{metric}_gain_per_1000_parameters": metric_drop / (param_cost / 1000.0) if param_cost > 0 else np.nan,
                f"{metric}_gain_per_ms": metric_drop / infer_cost if infer_cost > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(f"{metric}_drop", ascending=False)


def build_ablation_sequence(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Build ablation sequence.
    
    Args:
        df: Input dataframe.
        metric: Metric column or metric name to use.
    """
    subset = df.dropna(subset=[metric]).copy()
    subset = subset[subset["module_count"] > 0]
    if subset.empty:
        return pd.DataFrame()
    sequence = [subset.sort_values(["module_count", metric], ascending=[True, False]).iloc[0]]
    used = {sequence[0].name}
    current_modules = set(_normalize_modules(sequence[0].get("enabled_modules")))

    while True:
        candidates = subset.loc[~subset.index.isin(used)]
        if candidates.empty:
            break
        enriched = []
        for idx, row in candidates.iterrows():
            modules = set(_normalize_modules(row.get("enabled_modules")))
            if not modules.issuperset(current_modules):
                continue
            added = modules - current_modules
            if not added:
                continue
            enriched.append((idx, len(added), float(row[metric])))
        if not enriched:
            break
        enriched.sort(key=lambda item: (item[1], -item[2]))
        next_idx = enriched[0][0]
        sequence.append(subset.loc[next_idx])
        used.add(next_idx)
        current_modules = set(_normalize_modules(subset.loc[next_idx].get("enabled_modules")))

    rows = []
    previous_metric = None
    previous_modules: set[str] = set()
    for row in sequence:
        modules = set(_normalize_modules(row.get("enabled_modules")))
        added_modules = modules - previous_modules
        metric_value = float(row[metric])
        rows.append(
            {
                "display_name": row.get("display_name"),
                metric: metric_value,
                "added_modules": ", ".join(MODULE_LABELS.get(module, module) for module in sorted(added_modules)) if added_modules else "Base configuration",
                "increment": metric_value - previous_metric if previous_metric is not None else metric_value,
            }
        )
        previous_metric = metric_value
        previous_modules = modules
    return pd.DataFrame(rows)


def compute_difficulty_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute difficulty summary.
    
    Args:
        df: Input dataframe.
    """
    available = [column for column in ["noise_level", "overlap_level", "nonstationarity_level"] if column in df.columns and df[column].notna().any()]
    if not available:
        return pd.DataFrame()
    subset = df.dropna(subset=available + ["macro_corr", "macro_snr", "model_type"]).copy()
    if subset.empty:
        return pd.DataFrame()
    normalized_columns = []
    for column in available:
        scaled = _minmax_series(subset[column])
        scaled_name = f"_{column}_scaled"
        subset[scaled_name] = scaled
        normalized_columns.append(scaled_name)
    subset["_difficulty_score"] = subset[normalized_columns].mean(axis=1)
    subset["difficulty_band"] = pd.qcut(
        subset["_difficulty_score"],
        q=min(3, subset["_difficulty_score"].nunique()),
        labels=["Easy", "Medium", "Hard"][: min(3, subset["_difficulty_score"].nunique())],
        duplicates="drop",
    )
    return (
        subset.groupby(["difficulty_band", "model_type"], dropna=False)[["macro_corr", "macro_snr", "test_loss"]]
        .mean()
        .reset_index()
    )


def _base_axis(fig_size: tuple[float, float] = (10.5, 4.8)) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=fig_size)
    return fig, ax


def _shorten_label(label: str, *, max_len: int = 42, wrap_width: int = 18) -> str:
    text = str(label).strip().replace("_", " ")
    if not text:
        return ""
    if len(text) > max_len:
        text = text[: max_len - 3].rstrip() + "..."
    return fill(text, width=wrap_width)


def _style_model_xaxis(
    fig: plt.Figure,
    ax: plt.Axes,
    labels: Iterable[str],
    *,
    rotation: int = 38,
    labelsize: int = 8,
) -> None:
    formatted = [_shorten_label(label) for label in labels]
    ax.set_xticks(np.arange(len(formatted)))
    ax.set_xticklabels(formatted, rotation=rotation, ha="right")
    ax.tick_params(axis="x", labelsize=labelsize)
    fig.subplots_adjust(bottom=0.34)


def _ordered_by_metric(df: pd.DataFrame, metric: str, *, ascending: bool | None = None, top_n: int | None = None) -> pd.DataFrame:
    subset = df.dropna(subset=[metric]).copy()
    if subset.empty:
        return subset
    if ascending is None:
        ascending = metric in {"test_loss", "parameters", "inference_time_ms", "training_time_s"}
    subset = subset.sort_values(metric, ascending=ascending)
    if top_n is not None:
        subset = subset.head(top_n)
    return subset


def figure_bar_by_model(df: pd.DataFrame, metric: str, title: str) -> plt.Figure | None:
    """Build the bar by model figure.
    
    Args:
        df: Input dataframe.
        metric: Metric column or metric name to use.
        title: Chart title.
    """
    ordered = _ordered_by_metric(df, metric)
    if ordered.empty:
        return None
    fig_width = max(10.5, 0.82 * len(ordered))
    fig, ax = _base_axis((fig_width, 5.8))
    colors = [MODEL_TYPE_COLORS.get(model_type, MODEL_TYPE_COLORS["Unknown"]) for model_type in ordered["model_type"]]
    positions = np.arange(len(ordered))
    ax.bar(positions, ordered[metric], color=colors)
    ax.set_title(title)
    ax.set_ylabel(METRIC_COLUMNS.get(metric, metric))
    _style_model_xaxis(fig, ax, ordered["display_name"])
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def figure_grouped_type_summary(df: pd.DataFrame, metrics: list[str], title: str) -> plt.Figure | None:
    """Build the grouped type summary figure.
    
    Args:
        df: Input dataframe.
        title: Chart title.
        metrics: Metric names to plot or aggregate.
    """
    available_metrics = [metric for metric in metrics if metric in df.columns and df[metric].notna().any()]
    if not available_metrics:
        return None
    subset = df[df["model_type"].isin(["SNN", "DNN", "Classical"])].copy()
    if subset.empty:
        return None
    grouped = subset.groupby("model_type")[available_metrics].mean().reindex(["SNN", "DNN", "Classical"]).dropna(how="all")
    if grouped.empty:
        return None
    fig, ax = _base_axis((11.0, 5.0))
    x = np.arange(len(grouped.index))
    width = 0.8 / max(len(available_metrics), 1)
    for idx, metric in enumerate(available_metrics):
        ax.bar(x + idx * width - (len(available_metrics) - 1) * width / 2, grouped[metric], width=width, label=METRIC_COLUMNS.get(metric, metric))
    ax.set_xticks(x)
    ax.set_xticklabels(grouped.index)
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def figure_best_snn_vs_dnn(df: pd.DataFrame) -> plt.Figure | None:
    """Build the best snn vs dnn figure.
    
    Args:
        df: Input dataframe.
    """
    best_snn = best_row(df[df["model_type"] == "SNN"], "macro_corr")
    best_dnn = best_row(df[df["model_type"] == "DNN"], "macro_corr")
    if best_snn is None or best_dnn is None:
        return None
    compare = pd.DataFrame(
        [
            {"label": f"Best SNN\n{best_snn['display_name']}", "macro_corr": best_snn["macro_corr"], "macro_snr": best_snn.get("macro_snr"), "test_loss": best_snn.get("test_loss")},
            {"label": f"Best DNN\n{best_dnn['display_name']}", "macro_corr": best_dnn["macro_corr"], "macro_snr": best_dnn.get("macro_snr"), "test_loss": best_dnn.get("test_loss")},
        ]
    )
    available_metrics = [metric for metric in ["macro_corr", "macro_snr", "test_loss"] if compare[metric].notna().any()]
    fig, ax = _base_axis((10.5, 4.8))
    x = np.arange(len(compare))
    width = 0.75 / max(len(available_metrics), 1)
    for idx, metric in enumerate(available_metrics):
        ax.bar(x + idx * width - (len(available_metrics) - 1) * width / 2, compare[metric], width=width, label=METRIC_COLUMNS.get(metric, metric))
    _style_model_xaxis(fig, ax, compare["label"], rotation=0, labelsize=9)
    ax.set_title("Best SNN vs Best DNN")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def figure_gap_summary(df: pd.DataFrame) -> plt.Figure | None:
    """Build the gap summary figure.
    
    Args:
        df: Input dataframe.
    """
    best_snn = best_row(df[df["model_type"] == "SNN"], "macro_corr")
    best_dnn = best_row(df[df["model_type"] == "DNN"], "macro_corr")
    if best_snn is None or best_dnn is None:
        return None
    corr_gap = float(best_dnn["macro_corr"]) - float(best_snn["macro_corr"])
    snr_gap = _safe_float(best_dnn.get("macro_snr")) - _safe_float(best_snn.get("macro_snr"))
    fig, ax = _base_axis((8.5, 4.4))
    labels = ["Macro corr gap", "Macro SNR gap"]
    values = [corr_gap, snr_gap]
    ax.bar(labels, values, color=["#9467bd", "#8c564b"])
    ax.set_title("Accuracy gap: best SNN relative to best DNN")
    ax.set_ylabel("Gap (DNN - SNN)")
    ax.grid(axis="y", alpha=0.25)
    for idx, value in enumerate(values):
        ax.text(idx, value, f"{value:.3f}", ha="center", va="bottom")
    fig.tight_layout()
    return fig


def figure_efficiency_gain(df: pd.DataFrame) -> plt.Figure | None:
    """Build the efficiency gain figure.
    
    Args:
        df: Input dataframe.
    """
    best_snn = best_row(df[df["model_type"] == "SNN"], "macro_corr")
    best_dnn = best_row(df[df["model_type"] == "DNN"], "macro_corr")
    if best_snn is None or best_dnn is None:
        return None
    labels = []
    values = []
    for label, numerator, denominator in [
        ("Parameter reduction", _safe_float(best_dnn.get("parameters")), _safe_float(best_snn.get("parameters"))),
        ("Inference speedup", _safe_float(best_dnn.get("inference_time_ms")), _safe_float(best_snn.get("inference_time_ms"))),
        ("Training speedup", _safe_float(best_dnn.get("training_time_s")), _safe_float(best_snn.get("training_time_s"))),
    ]:
        if numerator > 0 and denominator > 0:
            labels.append(label)
            values.append(numerator / denominator)
    if not values:
        return None
    fig, ax = _base_axis((8.8, 4.4))
    ax.bar(labels, values, color="#17becf")
    ax.set_title("Efficiency gain of best SNN relative to best DNN")
    ax.set_ylabel("Factor (DNN / SNN)")
    ax.grid(axis="y", alpha=0.25)
    for idx, value in enumerate(values):
        ax.text(idx, value, f"{value:.2f}x", ha="center", va="bottom")
    fig.tight_layout()
    return fig


def figure_scatter(df: pd.DataFrame, x: str, y: str, title: str) -> plt.Figure | None:
    """Build the scatter figure.
    
    Args:
        df: Input dataframe.
        x: Input tensor.
        title: Chart title.
    """
    subset = df.dropna(subset=[x, y]).copy()
    if subset.empty:
        return None
    fig, ax = _base_axis()
    for model_type, group in subset.groupby("model_type", dropna=False):
        ax.scatter(group[x], group[y], label=model_type, alpha=0.85, s=70, color=MODEL_TYPE_COLORS.get(model_type, MODEL_TYPE_COLORS["Unknown"]))
    ax.set_xlabel(METRIC_COLUMNS.get(x, SIGNAL_CONDITION_COLUMNS.get(x, x)))
    ax.set_ylabel(METRIC_COLUMNS.get(y, SIGNAL_CONDITION_COLUMNS.get(y, y)))
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend()
    for _, row in subset.iterrows():
        if row[y] == subset[y].max():
            ax.annotate(str(row["display_name"]), (row[x], row[y]), xytext=(5, 5), textcoords="offset points", fontsize=8)
    fig.tight_layout()
    return fig


def _format_parameter_tick(value: float, _pos: float) -> str:
    if value <= 0:
        return ""
    if value >= 1_000_000:
        scaled = value / 1_000_000
        return f"{scaled:.0f}M" if abs(scaled - round(scaled)) < 1e-8 else f"{scaled:.1f}M"
    if value >= 1_000:
        scaled = value / 1_000
        return f"{scaled:.0f}k" if abs(scaled - round(scaled)) < 1e-8 else f"{scaled:.1f}k"
    return f"{int(value)}"


def _compute_parameter_pareto(df: pd.DataFrame) -> pd.DataFrame:
    subset = df.dropna(subset=["parameters", "macro_corr"]).copy()
    subset = subset[subset["parameters"] > 0]
    if subset.empty:
        return pd.DataFrame(columns=df.columns)

    keep_indices: list[int] = []
    for idx, row in subset.iterrows():
        dominated = False
        for other_idx, other in subset.iterrows():
            if idx == other_idx:
                continue
            better_metric = float(other["macro_corr"]) >= float(row["macro_corr"])
            lower_params = float(other["parameters"]) <= float(row["parameters"])
            strictly_better = float(other["macro_corr"]) > float(row["macro_corr"]) or float(other["parameters"]) < float(row["parameters"])
            if better_metric and lower_params and strictly_better:
                dominated = True
                break
        if not dominated:
            keep_indices.append(idx)
    return subset.loc[keep_indices].sort_values(["parameters", "macro_corr"], ascending=[True, False])


def _style_parameter_tradeoff_axis(ax: plt.Axes) -> None:
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(LogLocator(base=10.0))
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
    ax.xaxis.set_major_formatter(FuncFormatter(_format_parameter_tick))
    ax.grid(which="major", axis="both", color="#c7cbd1", linewidth=0.8, alpha=0.7)
    ax.grid(which="minor", axis="x", color="#d9dde3", linewidth=0.5, alpha=0.45)
    ax.set_xlabel("Parameter count (log scale)")
    ax.set_ylabel("Macro correlation")


def _annotate_parameter_tradeoff_labels(ax: plt.Axes, label_df: pd.DataFrame) -> None:
    if label_df.empty:
        return
    x_min = float(label_df["parameters"].min())
    x_max = float(label_df["parameters"].max())
    x_span = max(np.log10(x_max) - np.log10(x_min), 1e-6)
    y_min = float(label_df["macro_corr"].min())
    y_max = float(label_df["macro_corr"].max())
    y_span = max(y_max - y_min, 1e-6)
    offset_candidates = [(8, 8), (8, -12), (10, 22), (-92, 8), (-92, -12), (-104, 22), (16, 34), (-112, 34)]
    placed_points: list[tuple[float, float]] = []
    for _, row in label_df.iterrows():
        x = float(row["parameters"])
        y = float(row["macro_corr"])
        nearby_count = sum(
            abs(np.log10(x) - np.log10(prev_x)) <= 0.08 * x_span and abs(y - prev_y) <= 0.12 * y_span
            for prev_x, prev_y in placed_points
        )
        dx, dy = offset_candidates[nearby_count % len(offset_candidates)]
        ax.annotate(
            _paper_label_for_row(row),
            (x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "#d0d0d0", "linewidth": 0.4, "alpha": 0.92},
            arrowprops={"arrowstyle": "-", "color": "#666666", "lw": 0.6, "alpha": 0.6},
        )
        placed_points.append((x, y))


def _parameter_tradeoff_label_subset(subset: pd.DataFrame, pareto: pd.DataFrame) -> pd.DataFrame:
    if subset.empty:
        return subset
    working = subset.copy()
    working["_paper_label"] = working.apply(_paper_label_for_row, axis=1)
    if len(subset) <= 14:
        return working.sort_values(["macro_corr", "parameters"], ascending=[False, True]).drop_duplicates(subset=["_paper_label"], keep="first")

    parts: list[pd.DataFrame] = []
    if not pareto.empty:
        pareto_working = pareto.copy()
        pareto_working["_paper_label"] = pareto_working.apply(_paper_label_for_row, axis=1)
        parts.append(pareto_working)
    parts.append(working.sort_values("macro_corr", ascending=False).head(4))
    parts.append(working.sort_values("parameters", ascending=True).head(3))
    for model_type in ["SNN", "DNN"]:
        group = working[working["model_type"] == model_type]
        if not group.empty:
            parts.append(group.sort_values("macro_corr", ascending=False).head(2))
            parts.append(group.sort_values("parameters", ascending=True).head(1))
    labels = pd.concat(parts, ignore_index=False).drop_duplicates(subset=["_paper_label"], keep="first")
    return labels.sort_values(["macro_corr", "parameters"], ascending=[False, True]).head(12)


def _add_snn_zoom_inset(fig: plt.Figure, ax: plt.Axes, subset: pd.DataFrame, pareto: pd.DataFrame) -> None:
    snn = subset[subset["model_type"] == "SNN"].copy()
    if len(snn) < 2:
        return
    dnn = subset[subset["model_type"] == "DNN"].copy()
    if not dnn.empty and float(dnn["parameters"].min()) <= float(snn["parameters"].max()) * 2.0:
        return

    inset = inset_axes(ax, width="38%", height="42%", loc="lower right", borderpad=1.2)
    for model_type, group in subset.groupby("model_type", dropna=False):
        inset.scatter(
            group["parameters"],
            group["macro_corr"],
            marker=MODEL_TYPE_MARKERS.get(model_type, "o"),
            s=38,
            alpha=0.7,
            color=MODEL_TYPE_COLORS.get(model_type, MODEL_TYPE_COLORS["Unknown"]),
            edgecolors="white",
            linewidths=0.6,
        )
    if not pareto.empty:
        inset.scatter(
            pareto["parameters"],
            pareto["macro_corr"],
            s=54,
            facecolors="none",
            edgecolors="#111111",
            linewidths=1.0,
            zorder=4,
        )
    inset.set_xscale("log")
    inset.set_xlim(float(snn["parameters"].min()) * 0.9, float(snn["parameters"].max()) * 1.15)
    inset.set_ylim(float(snn["macro_corr"].min()) - 0.02, float(snn["macro_corr"].max()) + 0.02)
    inset.xaxis.set_major_locator(LogLocator(base=10.0))
    inset.xaxis.set_major_formatter(FuncFormatter(_format_parameter_tick))
    inset.tick_params(axis="both", labelsize=7)
    inset.grid(which="major", axis="both", color="#d0d4da", linewidth=0.6, alpha=0.7)
    inset.grid(which="minor", axis="x", color="#e2e6eb", linewidth=0.4, alpha=0.4)
    inset.set_title("Low-parameter SNN region", fontsize=8)


def _figure_parameter_corr_tradeoff(
    df: pd.DataFrame,
    *,
    include_labels: bool,
    title: str,
    include_inset: bool = True,
    dedupe_same_parameter_count: bool = False,
) -> plt.Figure | None:
    subset = df.dropna(subset=["parameters", "macro_corr"]).copy()
    if subset.empty:
        return None
    subset = subset[subset["parameters"] > 0].sort_values(["parameters", "macro_corr"], ascending=[True, False])
    if subset.empty:
        return None
    if dedupe_same_parameter_count:
        subset = subset.drop_duplicates(subset=["parameters"], keep="first").reset_index(drop=True)
        if subset.empty:
            return None

    pareto = _compute_parameter_pareto(subset)
    fig, ax = _base_axis((11.4, 5.8))
    for model_type, group in subset.groupby("model_type", dropna=False):
        ax.scatter(
            group["parameters"],
            group["macro_corr"],
            label=model_type,
            marker=MODEL_TYPE_MARKERS.get(model_type, "o"),
            alpha=0.8,
            s=78,
            color=MODEL_TYPE_COLORS.get(model_type, MODEL_TYPE_COLORS["Unknown"]),
            edgecolors="white",
            linewidths=0.8,
            zorder=2,
        )
    if not pareto.empty:
        ax.scatter(
            pareto["parameters"],
            pareto["macro_corr"],
            s=122,
            facecolors="none",
            edgecolors="#111111",
            linewidths=1.3,
            label="Pareto-efficient",
            zorder=4,
        )

    _style_parameter_tradeoff_axis(ax)
    ax.set_title(title)
    legend = ax.legend(title="Model family", frameon=True, loc="lower right")
    legend.get_frame().set_alpha(0.92)
    legend.get_frame().set_edgecolor("#d0d0d0")
    if include_inset:
        _add_snn_zoom_inset(fig, ax, subset, pareto)
    if include_labels:
        _annotate_parameter_tradeoff_labels(ax, _parameter_tradeoff_label_subset(subset, pareto))
    if not pareto.empty:
        snn = subset[subset["model_type"] == "SNN"]
        dnn = subset[subset["model_type"] == "DNN"]
        if not snn.empty and not dnn.empty and float(dnn["parameters"].median()) > float(snn["parameters"].median()):
            ax.text(
                0.02,
                0.98,
                "Several SNN variants remain competitive while using far fewer parameters.",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#d0d0d0", "linewidth": 0.5, "alpha": 0.9},
            )
    fig.tight_layout()
    return fig


def figure_parameter_corr_distribution(df: pd.DataFrame) -> plt.Figure | None:
    """Build the parameter corr distribution figure.
    
    Args:
        df: Input dataframe.
    """
    return _figure_parameter_corr_tradeoff(
        df,
        include_labels=False,
        title="Performance-parameter trade-off: macro correlation vs model size",
        include_inset=True,
    )


def figure_parameter_corr_named(df: pd.DataFrame) -> plt.Figure | None:
    """Build the parameter corr named figure.
    
    Args:
        df: Input dataframe.
    """
    return _figure_parameter_corr_tradeoff(
        df,
        include_labels=True,
        title="Performance-parameter trade-off with labeled models",
        include_inset=False,
        dedupe_same_parameter_count=True,
    )


def figure_bubble(df: pd.DataFrame) -> plt.Figure | None:
    """Build the bubble figure.
    
    Args:
        df: Input dataframe.
    """
    subset = df.dropna(subset=["inference_time_ms", "macro_corr", "parameters"]).copy()
    if subset.empty:
        return None
    fig, ax = _base_axis((10.5, 5.0))
    scale = subset["parameters"].fillna(0).clip(lower=1.0)
    bubble_sizes = 200 * np.sqrt(scale / scale.max())
    for model_type, group in subset.groupby("model_type", dropna=False):
        mask = group.index
        ax.scatter(
            group["inference_time_ms"],
            group["macro_corr"],
            s=bubble_sizes.loc[mask],
            alpha=0.55,
            label=model_type,
            color=MODEL_TYPE_COLORS.get(model_type, MODEL_TYPE_COLORS["Unknown"]),
        )
    ax.set_xlabel(METRIC_COLUMNS["inference_time_ms"])
    ax.set_ylabel(METRIC_COLUMNS["macro_corr"])
    ax.set_title("Accuracy-efficiency bubble plot")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig


def figure_pareto_frontier(df: pd.DataFrame) -> plt.Figure | None:
    """Build the pareto frontier figure.
    
    Args:
        df: Input dataframe.
    """
    subset = df.dropna(subset=["inference_time_ms", "macro_corr", "parameters"]).copy()
    if subset.empty:
        return None
    frontier = compute_pareto_frontier(subset)
    fig, ax = _base_axis((10.5, 5.0))
    for model_type, group in subset.groupby("model_type", dropna=False):
        ax.scatter(
            group["inference_time_ms"],
            group["macro_corr"],
            alpha=0.45,
            s=60,
            label=f"{model_type} models",
            color=MODEL_TYPE_COLORS.get(model_type, MODEL_TYPE_COLORS["Unknown"]),
        )
    if not frontier.empty:
        ordered = frontier.sort_values("inference_time_ms")
        ax.scatter(
            ordered["inference_time_ms"],
            ordered["macro_corr"],
            s=90,
            color="#111111",
            marker="D",
            label="Pareto-optimal models",
            zorder=3,
        )
        x_span = max(float(ordered["inference_time_ms"].max()) - float(ordered["inference_time_ms"].min()), 1e-6)
        y_span = max(float(ordered["macro_corr"].max()) - float(ordered["macro_corr"].min()), 1e-6)
        offset_candidates = [
            (10, 10),
            (10, -14),
            (10, 24),
            (-90, 10),
            (-90, -14),
            (-90, 24),
            (18, 38),
            (-98, 38),
        ]
        placed_points: list[tuple[float, float]] = []
        for _, row in ordered.iterrows():
            x = float(row["inference_time_ms"])
            y = float(row["macro_corr"])
            nearby_count = sum(
                abs(x - prev_x) <= 0.08 * x_span and abs(y - prev_y) <= 0.12 * y_span
                for prev_x, prev_y in placed_points
            )
            dx, dy = offset_candidates[nearby_count % len(offset_candidates)]
            ax.annotate(
                str(row["display_name"]),
                (x, y),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=8,
                bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "none", "alpha": 0.88},
                arrowprops={"arrowstyle": "-", "color": "#666666", "lw": 0.6, "alpha": 0.6},
            )
            placed_points.append((x, y))
    ax.set_xlabel(METRIC_COLUMNS["inference_time_ms"])
    ax.set_ylabel(METRIC_COLUMNS["macro_corr"])
    ax.set_title("Pareto frontier: accuracy vs inference time")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig


def figure_module_impact(df: pd.DataFrame, metric: str, title: str) -> plt.Figure | None:
    """Build the module impact figure.
    
    Args:
        df: Input dataframe.
        metric: Metric column or metric name to use.
        title: Chart title.
    """
    impact = compute_ablation_removal_impact(df, metric)
    if impact.empty:
        return None
    fig, ax = _base_axis((10.5, 4.8))
    ax.barh(impact["module_label"], impact[f"{metric}_drop"], color="#ff7f0e")
    ax.set_title(title)
    ax.set_xlabel("Performance drop")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    return fig


def figure_module_ranking(df: pd.DataFrame, metric: str, title: str) -> plt.Figure | None:
    """Build the module ranking figure.
    
    Args:
        df: Input dataframe.
        metric: Metric column or metric name to use.
        title: Chart title.
    """
    ranking = compute_module_contributions(df, metric)
    if ranking.empty:
        return None
    gain_column = f"{metric}_gain"
    merged_rows: list[dict[str, object]] = []
    for _, row in ranking.iterrows():
        gain_value = float(row[gain_column])
        if merged_rows and math.isclose(float(merged_rows[-1][gain_column]), gain_value, rel_tol=1e-9, abs_tol=1e-12):
            merged_rows[-1]["module_label"] = f"{merged_rows[-1]['module_label']} + {row['module_label']}"
        else:
            merged_rows.append(
                {
                    "module_label": str(row["module_label"]),
                    gain_column: gain_value,
                }
            )
    ranking = pd.DataFrame(merged_rows)
    fig, ax = _base_axis((10.5, 4.8))
    ax.barh(ranking["module_label"], ranking[gain_column], color="#bcbd22")
    ax.set_title(title)
    ax.set_xlabel("Average gain when module is enabled")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    return fig


def figure_heatmap(df: pd.DataFrame, metric: str) -> plt.Figure | None:
    """Build the heatmap figure.
    
    Args:
        df: Input dataframe.
        metric: Metric column or metric name to use.
    """
    module_columns = available_module_columns(df)
    module_columns = [column for column in module_columns if column != "frequency_features"]
    subset = df.dropna(subset=[metric]).copy()
    subset = subset[subset["module_count"] > 0]
    if subset.empty or not module_columns:
        return None
    subset = subset.sort_values(metric, ascending=False)
    label_column = "display_name" if "display_name" in subset.columns else "model_name"
    subset = subset.drop_duplicates(subset=[label_column], keep="first").head(15)
    matrix_frame = subset[module_columns].copy()
    for column in matrix_frame.columns:
        matrix_frame[column] = matrix_frame[column].astype("boolean").fillna(False).astype(float)
    matrix = matrix_frame.to_numpy()
    fig, ax = _base_axis((12.0, max(4.0, 0.45 * len(subset) + 1.5)))
    cmap = plt.cm.get_cmap("Blues", 2)
    image = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=0.0, vmax=1.0, interpolation="nearest")
    ax.set_title(f"Enabled modules vs {METRIC_COLUMNS.get(metric, metric)}")
    ax.set_yticks(np.arange(len(subset)))
    ax.set_yticklabels(
        [
            f"{name} ({value:.3f})"
            for name, value in zip(subset[label_column], subset[metric])
        ]
    )
    ax.set_xticks(np.arange(len(module_columns)))
    ax.set_xticklabels([MODULE_LABELS.get(column, column) for column in module_columns], rotation=45, ha="right")
    colorbar = fig.colorbar(image, ax=ax, shrink=0.8, ticks=[0.0, 1.0])
    colorbar.ax.set_yticklabels(["0", "1"])
    fig.tight_layout()
    return fig


def figure_waterfall(df: pd.DataFrame, metric: str) -> plt.Figure | None:
    """Build the waterfall figure.
    
    Args:
        df: Input dataframe.
        metric: Metric column or metric name to use.
    """
    sequence = build_ablation_sequence(df, metric)
    if sequence.empty:
        return None
    fig, ax = _base_axis((10.5, 4.8))
    running = 0.0
    xs = np.arange(len(sequence))
    for idx, row in sequence.iterrows():
        increment = float(row["increment"])
        ax.bar(xs[idx], increment, bottom=running, color="#4c78a8")
        running += increment
    ax.set_xticks(xs)
    ax.set_xticklabels(sequence["added_modules"], rotation=35, ha="right")
    ax.set_title(f"Cumulative {METRIC_COLUMNS.get(metric, metric)} improvement across module additions")
    ax.set_ylabel(METRIC_COLUMNS.get(metric, metric))
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def figure_signal_condition(df: pd.DataFrame, x_column: str, y_column: str, title: str) -> plt.Figure | None:
    """Build the signal condition figure.
    
    Args:
        df: Input dataframe.
        title: Chart title.
    """
    subset = df.dropna(subset=[x_column, y_column]).copy()
    if subset.empty or subset[x_column].nunique(dropna=True) < 2:
        return None
    fig, ax = _base_axis((10.5, 4.8))
    aggregated = subset.groupby([x_column, "model_type"], dropna=False)[y_column].mean().reset_index()
    for model_type, group in aggregated.groupby("model_type", dropna=False):
        ax.plot(group[x_column], group[y_column], marker="o", linewidth=1.8, label=model_type, color=MODEL_TYPE_COLORS.get(model_type, MODEL_TYPE_COLORS["Unknown"]))
    ax.set_title(title)
    ax.set_xlabel(SIGNAL_CONDITION_COLUMNS.get(x_column, x_column))
    ax.set_ylabel(METRIC_COLUMNS.get(y_column, y_column))
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig


def figure_difficulty_bands(df: pd.DataFrame) -> plt.Figure | None:
    """Build the difficulty bands figure.
    
    Args:
        df: Input dataframe.
    """
    difficulty = compute_difficulty_summary(df)
    if difficulty.empty:
        return None
    fig, ax = _base_axis((10.5, 4.8))
    pivot = difficulty.pivot(index="difficulty_band", columns="model_type", values="macro_corr")
    pivot = pivot.reindex(index=[label for label in ["Easy", "Medium", "Hard"] if label in pivot.index])
    x = np.arange(len(pivot.index))
    width = 0.8 / max(len(pivot.columns), 1)
    for idx, column in enumerate(pivot.columns):
        ax.bar(x + idx * width - (len(pivot.columns) - 1) * width / 2, pivot[column], width=width, label=column, color=MODEL_TYPE_COLORS.get(column, MODEL_TYPE_COLORS["Unknown"]))
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index)
    ax.set_title("SNN vs DNN under easy, medium, and hard signal conditions")
    ax.set_ylabel(METRIC_COLUMNS["macro_corr"])
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig


def figure_best_per_category(df: pd.DataFrame) -> plt.Figure | None:
    """Build the best per category figure.
    
    Args:
        df: Input dataframe.
    """
    rows = []
    for model_type in ["SNN", "DNN", "Classical"]:
        row = best_row(df[df["model_type"] == model_type], "macro_corr")
        if row is not None:
            rows.append({"model_type": model_type, "display_name": row["display_name"], "macro_corr": row["macro_corr"]})
    if not rows:
        return None
    best_df = pd.DataFrame(rows)
    fig, ax = _base_axis((9.0, 4.6))
    ax.bar(best_df["model_type"], best_df["macro_corr"], color=[MODEL_TYPE_COLORS.get(item, MODEL_TYPE_COLORS["Unknown"]) for item in best_df["model_type"]])
    ax.set_title("Best model per category")
    ax.set_ylabel(METRIC_COLUMNS["macro_corr"])
    for idx, row in best_df.iterrows():
        ax.text(idx, row["macro_corr"], row["display_name"], rotation=90, va="bottom", ha="center", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def figure_best_shallow_vs_tasnet(df: pd.DataFrame) -> plt.Figure | None:
    """Build the best shallow vs tasnet figure.
    
    Args:
        df: Input dataframe.
    """
    best_snn = best_row(df[df["model_type"] == "SNN"], "macro_corr")
    tasnet = df[df["model_name"].astype(str).str.lower() == "tasnet"]
    if best_snn is None:
        return None
    comparator = tasnet.iloc[0] if not tasnet.empty else best_row(df[df["model_type"] == "DNN"], "macro_corr")
    if comparator is None:
        return None
    rows = pd.DataFrame(
        [
            {"label": f"Best shallow\n{best_snn['display_name']}", "macro_corr": best_snn["macro_corr"], "parameters": best_snn.get("parameters"), "inference_time_ms": best_snn.get("inference_time_ms")},
            {"label": f"Comparator\n{comparator['display_name']}", "macro_corr": comparator["macro_corr"], "parameters": comparator.get("parameters"), "inference_time_ms": comparator.get("inference_time_ms")},
        ]
    )
    fig, ax = _base_axis((10.5, 4.8))
    x = np.arange(len(rows))
    width = 0.25
    for idx, metric in enumerate(["macro_corr", "parameters", "inference_time_ms"]):
        if rows[metric].notna().any():
            ax.bar(x + idx * width - width, rows[metric], width=width, label=METRIC_COLUMNS.get(metric, metric))
    _style_model_xaxis(fig, ax, rows["label"], rotation=0, labelsize=9)
    ax.set_title("Best shallow model vs TasNet or best DNN")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def figure_speedup_vs_loss(df: pd.DataFrame, cost_column: str, title: str) -> plt.Figure | None:
    """Build the speedup vs loss figure.
    
    Args:
        df: Input dataframe.
        title: Chart title.
        cost_column: Project value for this call.
    """
    best_dnn = best_row(df[df["model_type"] == "DNN"], "macro_corr")
    snn_rows = df[(df["model_type"] == "SNN") & df[cost_column].notna() & df["macro_corr"].notna()]
    if best_dnn is None or snn_rows.empty:
        return None
    comparator_cost = _safe_float(best_dnn.get(cost_column))
    if comparator_cost <= 0:
        return None
    subset = snn_rows.copy()
    subset["performance_loss"] = float(best_dnn["macro_corr"]) - subset["macro_corr"]
    subset["speedup_factor"] = comparator_cost / subset[cost_column].clip(lower=1e-8)
    fig, ax = _base_axis((10.5, 4.8))
    ax.scatter(subset["speedup_factor"], subset["performance_loss"], s=90, alpha=0.85, color=MODEL_TYPE_COLORS["SNN"])
    ax.set_title(title)
    ax.set_xlabel("Speedup / reduction factor relative to best DNN")
    ax.set_ylabel("Macro correlation loss")
    ax.grid(alpha=0.25)
    for _, row in subset.iterrows():
        ax.annotate(str(row["display_name"]), (row["speedup_factor"], row["performance_loss"]), xytext=(4, 4), textcoords="offset points", fontsize=8)
    fig.tight_layout()
    return fig


def build_chart_catalog(df: pd.DataFrame) -> list[ChartSpec]:
    """Build chart catalog.
    
    Args:
        df: Input dataframe.
    """
    df = prepare_chart_generation_dataframe(df)
    charts: list[ChartSpec] = []

    def add(section: str, slug: str, title: str, figure: plt.Figure | None) -> None:
        if figure is not None:
            charts.append(ChartSpec(section=section, slug=slug, title=title, figure=figure))

    for metric, title in [
        ("macro_corr", "Macro correlation per model"),
        ("macro_snr", "Macro SNR per model"),
        ("test_loss", "Test loss per model"),
        ("parameters", "Parameter count per model"),
        ("inference_time_ms", "Inference time per model"),
        ("training_time_s", "Training time per model"),
    ]:
        add("Overall", f"overall_{metric}", title, figure_bar_by_model(df, metric, title))

    add("SNN vs DNN", "snn_dnn_averages", "Average SNN vs DNN performance", figure_grouped_type_summary(df, ["macro_corr", "macro_snr", "test_loss"], "Average SNN vs DNN performance"))
    add("SNN vs DNN", "best_snn_vs_dnn", "Best SNN vs best DNN", figure_best_snn_vs_dnn(df))
    add("SNN vs DNN", "accuracy_gap", "Accuracy gap chart", figure_gap_summary(df))
    add("SNN vs DNN", "efficiency_gain", "Efficiency gain chart", figure_efficiency_gain(df))

    for x_column, y_column, title in [
        ("parameters", "macro_corr", "Macro correlation vs parameter count"),
        ("inference_time_ms", "macro_corr", "Macro correlation vs inference time"),
        ("parameters", "macro_snr", "Macro SNR vs parameter count"),
        ("inference_time_ms", "macro_snr", "Macro SNR vs inference time"),
        ("parameters", "test_loss", "Test loss vs parameter count"),
    ]:
        add("Trade-offs", f"{y_column}_vs_{x_column}", title, figure_scatter(df, x_column, y_column, title))
    add("Trade-offs", "parameter_corr_distribution", "Parameter count vs macro correlation distribution", figure_parameter_corr_distribution(df))
    add("Trade-offs", "parameter_corr_named", "Parameter count vs macro correlation with model names", figure_parameter_corr_named(df))
    add("Trade-offs", "bubble", "Bubble plot", figure_bubble(df))
    add("Trade-offs", "pareto_frontier", "Pareto frontier", figure_pareto_frontier(df))

    add("Ablations", "module_removal_corr", "Performance change when each module is removed", figure_module_impact(df, "macro_corr", "Macro correlation drop when removing each module"))
    add("Ablations", "waterfall_corr", "Cumulative module improvement", figure_waterfall(df, "macro_corr"))
    add("Ablations", "heatmap_corr", "Module heatmap", figure_heatmap(df, "macro_corr"))
    add("Ablations", "module_rank_corr", "Macro correlation module ranking", figure_module_ranking(df, "macro_corr", "Module contribution ranking by macro correlation"))
    add("Ablations", "module_rank_snr", "Macro SNR module ranking", figure_module_ranking(df, "macro_snr", "Module contribution ranking by macro SNR"))

    for condition_column in ["noise_level", "overlap_level", "num_components"]:
        for metric in ["macro_corr", "macro_snr", "test_loss"]:
            add(
                "Signal Conditions",
                f"{metric}_vs_{condition_column}",
                f"{METRIC_COLUMNS.get(metric, metric)} vs {SIGNAL_CONDITION_COLUMNS.get(condition_column, condition_column)}",
                figure_signal_condition(df, condition_column, metric, f"{METRIC_COLUMNS.get(metric, metric)} vs {SIGNAL_CONDITION_COLUMNS.get(condition_column, condition_column)}"),
            )
    add("Signal Conditions", "difficulty_bands", "SNN vs DNN by difficulty band", figure_difficulty_bands(df))

    add("Thesis Summary", "best_per_category", "Best model per category", figure_best_per_category(df))
    add("Thesis Summary", "best_shallow_vs_tasnet", "Best shallow vs TasNet or best DNN", figure_best_shallow_vs_tasnet(df))
    add("Thesis Summary", "tradeoff_summary", "Accuracy-efficiency trade-off summary", figure_pareto_frontier(df))
    add("Thesis Summary", "parameter_reduction_vs_loss", "Parameter reduction vs performance loss", figure_speedup_vs_loss(df, "parameters", "Parameter reduction vs performance loss"))
    add("Thesis Summary", "inference_speedup_vs_loss", "Inference speedup vs performance loss", figure_speedup_vs_loss(df, "inference_time_ms", "Inference speedup vs performance loss"))
    add("Thesis Summary", "training_speedup_vs_loss", "Training speedup vs performance loss", figure_speedup_vs_loss(df, "training_time_s", "Training speedup vs performance loss"))

    return charts
