from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from app.services.inference_service import evaluate_prediction, load_model, run_inference
from app.services.paths import REPO_ROOT, SAVED_MODELS_DIR
from app.services.signal_service import PURE_MODE, SignalConfig, generate_signal


COMPARISON_SCRIPT = REPO_ROOT / "evaluation" / "model_eval" / "6compare_all_evals.py"
COMPARISON_ARTIFACTS_DIR = REPO_ROOT / "evaluation" / "model_eval" / "artifacts"


def comparison_script_path() -> Path:
    """Return the comparison script path."""
    return COMPARISON_SCRIPT


def build_comparison_command(skip_run: bool = False) -> list[str]:
    """Build comparison command.
    
    Args:
        skip_run: Skip evaluation subprocesses when true.
    """
    cmd = [sys.executable, str(COMPARISON_SCRIPT)]
    if skip_run:
        cmd.append("--skip-run")
    return cmd


def run_model_comparison(skip_run: bool = False):
    """Run model comparison.
    
    Args:
        skip_run: Skip evaluation subprocesses when true.
    """
    if not COMPARISON_SCRIPT.exists():
        raise FileNotFoundError(f"Comparison script not found: {COMPARISON_SCRIPT}")

    cmd = build_comparison_command(skip_run=skip_run)
    process = subprocess.Popen(
        cmd,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    return process, cmd


def comparison_table_path() -> Path:
    """Return the comparison table path."""
    return COMPARISON_ARTIFACTS_DIR / "model_comparison_table.md"


def _resolve_candidate_checkpoint(path_str: str | None) -> Path | None:
    if not path_str:
        return None
    path = Path(path_str)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def _read_train_samples(run_dir: Path) -> int | None:
    config_path = run_dir / "train_config.json"
    if not config_path.exists():
        return None
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    value = payload.get("train_samples") if isinstance(payload, dict) else None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _select_checkpoint_from_run_dir(run_dir: Path) -> Path | None:
    for pattern in ["*_best.pt", "*_last.pt", "*_epoch_*.pt", "*.pt"]:
        matches = [item for item in sorted(run_dir.glob(pattern)) if not item.name.endswith("_weights_only.pt")]
        if matches:
            return matches[0]
    return None


def resolve_model_checkpoint(model_spec: dict) -> str:
    """Resolve model checkpoint.
    
    Args:
        model_spec: Model registry entry.
    """
    configured = _resolve_candidate_checkpoint(str(model_spec.get("default_checkpoint", "")).strip())
    if configured is not None and configured.exists():
        return str(configured.relative_to(REPO_ROOT))

    model_key = str(model_spec.get("key", model_spec.get("model_key", ""))).strip()
    if not model_key or not SAVED_MODELS_DIR.exists():
        return ""

    candidates: list[tuple[int, int, int, str, Path]] = []
    for run_dir in SAVED_MODELS_DIR.iterdir():
        if not run_dir.is_dir():
            continue
        if run_dir.name != model_key and not run_dir.name.startswith(f"{model_key}_"):
            continue
        checkpoint = _select_checkpoint_from_run_dir(run_dir)
        if checkpoint is None:
            continue
        train_samples = _read_train_samples(run_dir)
        if train_samples == 8000:
            priority = 0
        elif train_samples is not None and 3500 <= train_samples <= 5000:
            priority = 1
        else:
            priority = 2
        distance_to_4000 = abs((train_samples or 10**9) - 4000) if priority == 1 else 0
        candidates.append((priority, distance_to_4000, -(train_samples or -1), run_dir.name, checkpoint))

    if not candidates:
        return ""
    candidates.sort()
    return str(candidates[0][4].relative_to(REPO_ROOT))


def _clean_metric_samples(values) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    return arr[np.isfinite(arr)]


def _bootstrap_mean_ci(values, confidence: float = 0.95, n_resamples: int = 2000, seed: int = 12345) -> tuple[float, float]:
    arr = _clean_metric_samples(values)
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        return float(arr[0]), float(arr[0])
    rng = np.random.default_rng(seed)
    draws = rng.choice(arr, size=(n_resamples, arr.size), replace=True).mean(axis=1)
    alpha = 1.0 - confidence
    low = float(np.quantile(draws, alpha / 2.0))
    high = float(np.quantile(draws, 1.0 - alpha / 2.0))
    return low, high


def _paired_permutation_pvalue(reference, candidate, *, lower_is_better: bool, n_resamples: int = 20000, seed: int = 12345) -> tuple[float, float]:
    ref = np.asarray(reference, dtype=float).reshape(-1)
    cand = np.asarray(candidate, dtype=float).reshape(-1)
    if ref.size == 0 or cand.size == 0:
        return float("nan"), float("nan")
    shared = min(ref.size, cand.size)
    if shared <= 0:
        return float("nan"), float("nan")
    ref = ref[:shared]
    cand = cand[:shared]
    mask = np.isfinite(ref) & np.isfinite(cand)
    ref = ref[mask]
    cand = cand[mask]
    if ref.size == 0:
        return float("nan"), float("nan")

    diff = (cand - ref) if lower_is_better else (ref - cand)
    observed = float(np.mean(diff))
    if ref.size == 1:
        return observed, 1.0

    centered = diff - np.mean(diff)
    rng = np.random.default_rng(seed)
    if centered.size <= 15:
        sign_patterns = 1 << centered.size
        flips = ((np.arange(sign_patterns)[:, None] >> np.arange(centered.size)) & 1) * 2 - 1
        permuted = np.mean(flips * centered, axis=1)
    else:
        flips = rng.choice(np.array([-1.0, 1.0]), size=(n_resamples, centered.size), replace=True)
        permuted = np.mean(flips * centered, axis=1)
    p_value = float((np.sum(np.abs(permuted) >= abs(observed)) + 1) / (permuted.size + 1))
    return observed, p_value


def add_paired_statistics(
    comparison_df: pd.DataFrame,
    *,
    metric: str,
    reference_model: str,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Add paired statistics.
    
    Args:
        metric: Metric column or metric name to use.
        comparison_df: Input dataframe.
    """
    if comparison_df.empty or metric not in comparison_df.columns:
        return comparison_df

    metric_samples_col = f"{metric}_samples"
    if metric_samples_col not in comparison_df.columns:
        return comparison_df

    enriched = comparison_df.copy()
    lower_is_better = metric in {"observed_mse", "test_loss", "clean_sum_mse", "observed_mixture_mse"}
    enriched[f"{metric}_ci_low"] = np.nan
    enriched[f"{metric}_ci_high"] = np.nan
    enriched[f"{metric}_ci"] = ""
    enriched["paired_reference_model"] = reference_model
    enriched["paired_mean_diff_vs_reference"] = np.nan
    enriched["paired_p_value"] = np.nan
    enriched["paired_significant_95"] = False

    reference_rows = enriched[enriched["display_name"] == reference_model]
    if reference_rows.empty:
        return enriched
    reference_samples = reference_rows.iloc[0][metric_samples_col]

    for idx, row in enriched.iterrows():
        samples = row.get(metric_samples_col, [])
        ci_low, ci_high = _bootstrap_mean_ci(samples, confidence=confidence)
        enriched.at[idx, f"{metric}_ci_low"] = ci_low
        enriched.at[idx, f"{metric}_ci_high"] = ci_high
        if np.isfinite(ci_low) and np.isfinite(ci_high):
            enriched.at[idx, f"{metric}_ci"] = f"[{ci_low:.4f}, {ci_high:.4f}]"

        mean_diff, p_value = _paired_permutation_pvalue(
            reference_samples,
            samples,
            lower_is_better=lower_is_better,
        )
        enriched.at[idx, "paired_mean_diff_vs_reference"] = mean_diff
        enriched.at[idx, "paired_p_value"] = p_value
        enriched.at[idx, "paired_significant_95"] = bool(np.isfinite(p_value) and p_value < 0.05)

    return enriched


def evaluate_registry_on_components(
    registry: list[dict],
    component_types: list[str],
    *,
    fs: int,
    duration: float,
    noise_level: float,
    seed: int,
    num_samples: int,
    permutation_invariant: bool = True,
) -> list[dict]:
    """Evaluate registry on components.
    
    Args:
        registry: Model registry entries.
        seed: Random seed for reproducible output.
    """
    rows: list[dict] = []
    component_count = len(component_types)
    for model_spec in [item for item in registry if item.get("enabled", True)]:
        model_key = model_spec.get("model_key", model_spec.get("key", ""))
        display_name = model_spec.get("display_name", model_key)
        checkpoint = resolve_model_checkpoint(model_spec)
        row = {
            "model_name": model_spec.get("key", model_key),
            "display_name": display_name,
            "family": model_spec.get("family", ""),
            "depth_label": model_spec.get("depth_label", ""),
            "component_set": ", ".join(component_types),
            "n_components": component_count,
            "checkpoint_path": checkpoint,
            "status": "ok",
        }
        try:
            if not checkpoint:
                raise FileNotFoundError(f"No checkpoint found for model '{display_name}'.")
            model, _ = load_model(model_key=model_key, out_channels=component_count, checkpoint_path=checkpoint)
            macro_corr = []
            macro_snr = []
            observed_mse = []
            for sample_idx in range(num_samples):
                config = SignalConfig(
                    signal_type="mixed",
                    n_components=component_count,
                    duration=duration,
                    fs=fs,
                    noise_level=noise_level,
                    seed=seed + sample_idx,
                    generation_mode=PURE_MODE,
                    selected_component_types=component_types,
                )
                generated = generate_signal(config)
                y_pred, _ = run_inference(model, generated.mixture)
                if y_pred.shape != generated.components.shape:
                    raise ValueError(f"prediction shape {y_pred.shape} does not match target shape {generated.components.shape}")
                report = evaluate_prediction(
                    generated.components,
                    y_pred,
                    generated.mixture,
                    permutation_invariant=permutation_invariant,
                )
                macro = report.get("macro_average", {})
                observed = report.get("observed_mixture_metrics", {})
                macro_corr.append(float(macro.get("corr", np.nan)))
                macro_snr.append(float(macro.get("snr_db", np.nan)))
                observed_mse.append(float(observed.get("mse", np.nan)))
            row.update(
                {
                    "macro_corr": float(np.nanmean(macro_corr)),
                    "macro_snr_db": float(np.nanmean(macro_snr)),
                    "observed_mse": float(np.nanmean(observed_mse)),
                    "macro_corr_samples": macro_corr,
                    "macro_snr_db_samples": macro_snr,
                    "observed_mse_samples": observed_mse,
                }
            )
        except Exception as exc:
            row.update(
                {
                    "macro_corr": np.nan,
                    "macro_snr_db": np.nan,
                    "observed_mse": np.nan,
                    "macro_corr_samples": [],
                    "macro_snr_db_samples": [],
                    "observed_mse_samples": [],
                    "status": f"failed: {exc}",
                }
            )
        rows.append(row)
    return rows
