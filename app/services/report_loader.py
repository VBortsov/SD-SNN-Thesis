from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch

from app.services.paths import REPO_ROOT, SAVED_MODELS_DIR
from app.services.inference_service import load_model, run_inference, count_parameters


@dataclass
class ReportLoadResult:
    """Result data returned by a workflow."""
    dataframe: pd.DataFrame
    warnings: list[str]


def _safe_float(value, default=float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def discover_report_files() -> list[Path]:
    """Discover report files."""
    return sorted(SAVED_MODELS_DIR.rglob("final_test_report.json"))


def infer_checkpoint(model_dir: Path) -> str:
    """Infer checkpoint.
    
    Args:
        model_dir: File-system location.
    """
    for name in ["*_best.pt", "*_last.pt"]:
        found = sorted(model_dir.glob(name))
        if found:
            return str(found[0].relative_to(REPO_ROOT))
    any_pt = sorted(model_dir.glob("*.pt"))
    if any_pt:
        return str(any_pt[0].relative_to(REPO_ROOT))
    return ""


def _load_single_report(path: Path) -> tuple[dict, str | None]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {}, f"Failed to parse {path}: {exc}"

    model_dir = path.parent
    summary = payload.get("test_summary", {}) if isinstance(payload, dict) else {}
    macro = summary.get("macro_average", {}) if isinstance(summary, dict) else {}
    clean_sum = summary.get("clean_sum_metrics") or summary.get("mixture_metrics", {})
    observed = summary.get("observed_mixture_metrics", {}) if isinstance(summary, dict) else {}
    metadata = payload.get("training_metadata", {}) if isinstance(payload, dict) else {}
    model_name = str(payload.get("model_name") or metadata.get("model_name") or model_dir.name)
    row = {
        "model_name": model_name,
        "test_loss": _safe_float(payload.get("test_loss")),
        "macro_corr": _safe_float(macro.get("corr")),
        "macro_snr_db": _safe_float(macro.get("snr_db")),
        "clean_sum_mse": _safe_float(clean_sum.get("mse") if isinstance(clean_sum, dict) else None),
        "observed_mixture_mse": _safe_float(observed.get("mse") if isinstance(observed, dict) else None),
        "report_path": str(path.relative_to(REPO_ROOT)),
        "checkpoint_path": infer_checkpoint(model_dir),
        "training_date": payload.get("timestamp", ""),
        "training_time_sec": _safe_float(metadata.get("elapsed_seconds") if isinstance(metadata, dict) else None),
        "parameter_count": _safe_float(metadata.get("parameter_count") if isinstance(metadata, dict) else None),
        "inference_time_ms": _safe_float(metadata.get("inference_time_ms") if isinstance(metadata, dict) else None),
        "train_samples": metadata.get("train_samples") if isinstance(metadata, dict) else None,
        "val_samples": metadata.get("val_samples") if isinstance(metadata, dict) else None,
        "test_samples": metadata.get("test_samples") if isinstance(metadata, dict) else None,
        "total_samples": metadata.get("total_samples") if isinstance(metadata, dict) else None,
        "epochs": metadata.get("epochs") if isinstance(metadata, dict) else None,
    }
    return row, None


def load_reports_dataframe(registry: list[dict]) -> ReportLoadResult:
    """Load reports dataframe.
    
    Args:
        registry: Model registry entries.
    """
    warnings: list[str] = []
    rows: list[dict] = []
    for report_path in discover_report_files():
        row, err = _load_single_report(report_path)
        if err:
            warnings.append(err)
            continue
        rows.append(row)

    ordered = [
        "model_name",
        "display_name",
        "family",
        "depth_label",
        "test_loss",
        "macro_corr",
        "macro_snr_db",
        "clean_sum_mse",
        "observed_mixture_mse",
        "parameter_count",
        "inference_time_ms",
        "training_date",
        "training_time_sec",
        "train_samples",
        "val_samples",
        "test_samples",
        "total_samples",
        "epochs",
        "checkpoint_path",
        "report_path",
    ]

    if not rows:
        df = pd.DataFrame(columns=ordered)
    else:
        df = pd.DataFrame(rows)

    meta = pd.DataFrame(registry) if registry else pd.DataFrame()
    if not meta.empty:
        meta = meta.rename(columns={"key": "model_name"}).copy()
        meta_subset = meta[
            [
                "model_name",
                "display_name",
                "family",
                "depth_label",
                "default_checkpoint",
                "enabled",
            ]
        ]
        if df.empty:
            df = meta_subset.copy()
        else:
            df = df.merge(
                meta_subset,
                on="model_name",
                how="left",
            )
        df["checkpoint_path"] = df.get("checkpoint_path", pd.Series(dtype="object")).replace("", pd.NA).fillna(df["default_checkpoint"])

        known_model_names = set(df["model_name"].astype(str).tolist())
        missing_registry_rows = meta_subset[~meta_subset["model_name"].isin(known_model_names)].copy()
        if not missing_registry_rows.empty:
            for col in [
                "test_loss",
                "macro_corr",
                "macro_snr_db",
                "clean_sum_mse",
                "observed_mixture_mse",
                "parameter_count",
                "inference_time_ms",
                "training_date",
                "training_time_sec",
                "train_samples",
                "val_samples",
                "test_samples",
                "total_samples",
                "epochs",
                "report_path",
                "checkpoint_path",
            ]:
                if col not in missing_registry_rows.columns:
                    missing_registry_rows[col] = pd.NA
            missing_registry_rows["checkpoint_path"] = missing_registry_rows["default_checkpoint"]
            missing_registry_rows = missing_registry_rows[df.columns.intersection(missing_registry_rows.columns).tolist() + [col for col in missing_registry_rows.columns if col not in df.columns]]
            df = pd.concat([df, missing_registry_rows], ignore_index=True, sort=False)
    else:
        df["display_name"] = df["model_name"]
        df["family"] = "unknown"
        df["depth_label"] = "unknown"

    if "parameter_count" not in df.columns:
        df["parameter_count"] = pd.NA
    if "inference_time_ms" not in df.columns:
        df["inference_time_ms"] = pd.NA
    df = enrich_with_model_stats(df)
    for col in ordered:
        if col not in df.columns:
            df[col] = pd.NA
    return ReportLoadResult(dataframe=df[ordered], warnings=warnings)


def enrich_with_model_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Enrich with model stats.
    
    Args:
        df: Input dataframe.
    """
    if df.empty:
        return df
    enriched = df.copy()
    for idx, row in enriched.iterrows():
        if pd.notna(row.get("parameter_count")) and pd.notna(row.get("inference_time_ms")):
            continue
        model_key = str(row.get("model_key") or row.get("model_name", "")).strip()
        if not model_key:
            continue
        try:
            model, _ = load_model(model_key=model_key, out_channels=3, checkpoint_path=None)
            enriched.at[idx, "parameter_count"] = count_parameters(model)
            dummy = torch.zeros(1024, dtype=torch.float32).numpy()
            _, infer_ms = run_inference(model, dummy)
            enriched.at[idx, "inference_time_ms"] = float(infer_ms)
        except Exception:
            continue
    return enriched
