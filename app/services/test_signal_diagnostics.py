from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from textwrap import fill
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from app.services.paths import APP_EXPORTS_DIR, REPO_ROOT


EPS = 1e-12


@dataclass
class FailureThresholds:
    """Thresholds used to label weak reconstructions."""
    corr_shape_failure: float = 0.45
    cosine_shape_failure: float = 0.50
    corr_amplitude_ok: float = 0.75
    mse_amplitude_mismatch: float = 0.08
    mae_high_error: float = 0.18
    rmse_high_error: float = 0.28
    max_abs_error_high: float = 0.70
    energy_ratio_low: float = 0.60
    energy_ratio_high: float = 1.50
    snr_poor: float = 0.0
    si_sdr_poor: float = 0.0
    explained_variance_bad: float = 0.0
    spectral_convergence_high: float = 0.60
    log_spectral_distance_high: float = 1.00
    fft_magnitude_l1_high: float = 3.0
    dominant_error_ratio: float = 1.20


def thresholds_to_dict(thresholds: FailureThresholds) -> dict[str, float]:
    """Thresholds to dict.
    
    Args:
        thresholds: Failure threshold settings.
    """
    return asdict(thresholds)


def build_thresholds(overrides: dict[str, Any] | None = None) -> FailureThresholds:
    """Build thresholds.
    
    Args:
        overrides: Optional threshold overrides.
    """
    values = thresholds_to_dict(FailureThresholds())
    for key, value in (overrides or {}).items():
        if key in values:
            values[key] = float(value)
    return FailureThresholds(**values)


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


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


def _spectral_bandwidth(x: np.ndarray, fs: int | None) -> float:
    x = np.asarray(x, dtype=float)
    if x.size < 4:
        return float("nan")
    mag = np.abs(np.fft.rfft(x - np.mean(x)))
    d = 1.0 / fs if fs and fs > 0 else 1.0
    freq = np.fft.rfftfreq(x.size, d=d)
    centroid = _spectral_centroid(x, fs)
    return float(np.sqrt(np.sum(((freq - centroid) ** 2) * mag) / (np.sum(mag) + EPS)))


def _spectral_overlap(a: np.ndarray, b: np.ndarray) -> float:
    spec_a = np.abs(np.fft.rfft(a - np.mean(a)))
    spec_b = np.abs(np.fft.rfft(b - np.mean(b)))
    denom = min(np.sum(spec_a), np.sum(spec_b)) + EPS
    return float(np.sum(np.minimum(spec_a, spec_b)) / denom)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) < EPS or np.std(b) < EPS:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _window_centroids(x: np.ndarray, fs: int | None, windows: int = 8) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.size < windows * 4:
        return np.array([], dtype=float)
    chunks = np.array_split(x, windows)
    values = []
    for chunk in chunks:
        if chunk.size < 4:
            continue
        values.append(_spectral_centroid(chunk, fs))
    return np.asarray(values, dtype=float)


def _chirp_likeness_score(x: np.ndarray, fs: int | None) -> float:
    values = _window_centroids(x, fs)
    if values.size < 3 or not np.all(np.isfinite(values)):
        return float("nan")
    x_axis = np.linspace(0.0, 1.0, values.size)
    slope, intercept = np.polyfit(x_axis, values, deg=1)
    residual = values - (slope * x_axis + intercept)
    return float((abs(slope) + np.std(residual)) / (np.mean(np.abs(values)) + EPS))


def _phase_shift_estimate(y_true: np.ndarray, y_pred: np.ndarray, fs: int | None) -> float:
    true_centered = np.asarray(y_true, dtype=float) - np.mean(y_true)
    pred_centered = np.asarray(y_pred, dtype=float) - np.mean(y_pred)
    if true_centered.size != pred_centered.size or true_centered.size < 4:
        return float("nan")
    corr = np.correlate(pred_centered, true_centered, mode="full")
    lag = int(np.argmax(corr) - (true_centered.size - 1))
    if fs and fs > 0:
        return float(lag / fs)
    return float(lag)


def _infer_failure_labels(
    metrics: dict[str, Any],
    *,
    thresholds: FailureThresholds,
    dominant_component: bool,
) -> list[str]:
    labels: list[str] = []
    corr = _safe_float(metrics.get("corr"))
    cosine = _safe_float(metrics.get("cosine_similarity"))
    mse = _safe_float(metrics.get("mse"))
    mae = _safe_float(metrics.get("mae"))
    rmse = _safe_float(metrics.get("rmse"))
    snr = _safe_float(metrics.get("snr_db"))
    si_sdr = _safe_float(metrics.get("si_sdr_db"))
    explained_variance = _safe_float(metrics.get("explained_variance"))
    energy_ratio = _safe_float(metrics.get("energy_ratio"))
    spectral_convergence = _safe_float(metrics.get("spectral_convergence"))
    lsd = _safe_float(metrics.get("log_spectral_distance"))
    fft_l1 = _safe_float(metrics.get("fft_magnitude_l1"))
    max_abs_error = _safe_float(metrics.get("max_abs_error"))

    if corr < thresholds.corr_shape_failure or cosine < thresholds.cosine_shape_failure:
        labels.append("shape_failure")
    if corr >= thresholds.corr_amplitude_ok and mse >= thresholds.mse_amplitude_mismatch:
        labels.append("amplitude_mismatch")
    if mae >= thresholds.mae_high_error or rmse >= thresholds.rmse_high_error or max_abs_error >= thresholds.max_abs_error_high:
        labels.append("high_pointwise_error")
    if energy_ratio < thresholds.energy_ratio_low:
        labels.append("component_suppression")
    if energy_ratio > thresholds.energy_ratio_high:
        labels.append("component_overprediction")
    if snr < thresholds.snr_poor:
        labels.append("poor_reconstruction_quality")
    if si_sdr < thresholds.si_sdr_poor:
        labels.append("poor_source_separation")
    if (
        spectral_convergence > thresholds.spectral_convergence_high
        or lsd > thresholds.log_spectral_distance_high
        or fft_l1 > thresholds.fft_magnitude_l1_high
    ):
        labels.append("spectral_mismatch")
    if explained_variance < thresholds.explained_variance_bad:
        labels.append("worse_than_mean_prediction")
    if dominant_component:
        labels.append("dominant_failure_component")
    return labels


def _primary_failure_mode(labels: list[str]) -> str:
    if "shape_failure" in labels:
        return "shape-related"
    if "spectral_mismatch" in labels:
        return "spectral"
    if "amplitude_mismatch" in labels or "component_suppression" in labels or "component_overprediction" in labels:
        return "amplitude/energy"
    if "poor_source_separation" in labels:
        return "source-separation"
    if "high_pointwise_error" in labels:
        return "pointwise"
    return "mixed"


def compute_component_diagnostics(
    report: dict[str, Any],
    component_names: list[str] | None,
    *,
    y_true: np.ndarray | None = None,
    y_pred: np.ndarray | None = None,
    fs: int | None = None,
    thresholds: FailureThresholds | None = None,
) -> pd.DataFrame:
    """Compute component diagnostics.
    
    Args:
        y_true: Target component signals.
        y_pred: Predicted component signals.
    """
    thresholds = thresholds or FailureThresholds()
    component_metrics = list(report.get("component_metrics", []))
    names = component_names or [f"component_{index + 1}" for index in range(len(component_metrics))]

    dominant_index = None
    if component_metrics:
        nmse_values = np.asarray([_safe_float(item.get("nmse")) for item in component_metrics], dtype=float)
        if np.isfinite(nmse_values).any():
            dominant_index = int(np.nanargmax(nmse_values))

    rows: list[dict[str, Any]] = []
    for index, metrics in enumerate(component_metrics):
        true_component = None if y_true is None or index >= y_true.shape[0] else np.asarray(y_true[index], dtype=float)
        pred_component = None if y_pred is None or index >= y_pred.shape[0] else np.asarray(y_pred[index], dtype=float)
        true_energy = float(np.sum(true_component**2)) if true_component is not None else float("nan")
        pred_energy = float(np.sum(pred_component**2)) if pred_component is not None else float("nan")
        residual = true_component - pred_component if true_component is not None and pred_component is not None else None
        residual_energy = float(np.sum(residual**2)) if residual is not None else float("nan")
        labels = _infer_failure_labels(
            metrics,
            thresholds=thresholds,
            dominant_component=(dominant_index == index),
        )
        rows.append(
            {
                "component": names[index] if index < len(names) else f"component_{index + 1}",
                **metrics,
                "true_energy": true_energy,
                "pred_energy": pred_energy,
                "residual_energy": residual_energy,
                "residual_energy_ratio": residual_energy / (true_energy + EPS) if np.isfinite(true_energy) else float("nan"),
                "dominant_frequency_true": _dominant_frequency(true_component, fs) if true_component is not None else float("nan"),
                "dominant_frequency_pred": _dominant_frequency(pred_component, fs) if pred_component is not None else float("nan"),
                "spectral_centroid_true": _spectral_centroid(true_component, fs) if true_component is not None else float("nan"),
                "spectral_centroid_pred": _spectral_centroid(pred_component, fs) if pred_component is not None else float("nan"),
                "bandwidth_true": _spectral_bandwidth(true_component, fs) if true_component is not None else float("nan"),
                "bandwidth_pred": _spectral_bandwidth(pred_component, fs) if pred_component is not None else float("nan"),
                "chirp_likeness_true": _chirp_likeness_score(true_component, fs) if true_component is not None else float("nan"),
                "chirp_likeness_pred": _chirp_likeness_score(pred_component, fs) if pred_component is not None else float("nan"),
                "phase_shift_estimate": _phase_shift_estimate(true_component, pred_component, fs)
                if true_component is not None and pred_component is not None
                else float("nan"),
                "failure_labels": labels,
                "failure_label_text": ", ".join(labels),
                "primary_failure_mode": _primary_failure_mode(labels),
            }
        )
    return pd.DataFrame(rows)


def compute_signal_difficulty_descriptors(
    mixture: np.ndarray,
    components: np.ndarray | None,
    predictions: np.ndarray | None,
    component_names: list[str] | None,
    *,
    fs: int | None,
) -> dict[str, Any]:
    """Compute signal difficulty descriptors.
    
    Args:
        mixture: Observed mixed signal.
    """
    mixture = np.asarray(mixture, dtype=float).reshape(-1)
    names = component_names or []
    components = None if components is None else np.asarray(components, dtype=float)
    predictions = None if predictions is None else np.asarray(predictions, dtype=float)

    summary: dict[str, Any] = {
        "mixture_energy": float(np.sum(mixture**2)),
    }
    if components is None or components.ndim != 2:
        return summary

    true_energies = np.sum(components**2, axis=1)
    clean_sum = np.sum(components, axis=0)
    noise = mixture - clean_sum if mixture.shape == clean_sum.shape else None
    residual = None
    if predictions is not None and predictions.shape == components.shape:
        residual = components - predictions
        predicted_energies = np.sum(predictions**2, axis=1)
        residual_energies = np.sum(residual**2, axis=1)
    else:
        predicted_energies = np.full_like(true_energies, np.nan, dtype=float)
        residual_energies = np.full_like(true_energies, np.nan, dtype=float)

    overlap_rows = []
    for left in range(components.shape[0]):
        for right in range(left + 1, components.shape[0]):
            overlap_rows.append(
                {
                    "component_a": names[left] if left < len(names) else f"component_{left + 1}",
                    "component_b": names[right] if right < len(names) else f"component_{right + 1}",
                    "spectral_overlap": _spectral_overlap(components[left], components[right]),
                    "correlation": _corr(components[left], components[right]),
                }
            )

    true_corr_matrix = np.corrcoef(components) if components.shape[0] > 1 else np.array([[1.0]])
    summary.update(
        {
            "estimated_noise_power": float(np.mean(noise**2)) if noise is not None else float("nan"),
            "estimated_input_snr_db": float(
                10.0 * np.log10((np.mean(clean_sum**2) + EPS) / (np.mean(noise**2) + EPS))
            )
            if noise is not None
            else float("nan"),
            "weakest_component_ratio": float(np.min(true_energies) / (np.max(true_energies) + EPS)) if true_energies.size else float("nan"),
            "amplitude_ratio_strongest_to_weakest": float(np.max(true_energies) / (np.min(true_energies) + EPS)) if true_energies.size else float("nan"),
            "mean_pairwise_spectral_overlap": float(np.mean([item["spectral_overlap"] for item in overlap_rows])) if overlap_rows else float("nan"),
            "max_pairwise_spectral_overlap": float(np.max([item["spectral_overlap"] for item in overlap_rows])) if overlap_rows else float("nan"),
            "mean_abs_component_correlation": float(np.mean(np.abs(true_corr_matrix[np.triu_indices_from(true_corr_matrix, k=1)])))
            if true_corr_matrix.shape[0] > 1
            else 0.0,
            "max_abs_component_correlation": float(np.max(np.abs(true_corr_matrix[np.triu_indices_from(true_corr_matrix, k=1)])))
            if true_corr_matrix.shape[0] > 1
            else 0.0,
            "component_correlation_matrix": true_corr_matrix.tolist(),
            "pairwise_spectral_overlap": overlap_rows,
        }
    )

    component_rows = []
    for index in range(components.shape[0]):
        true_component = components[index]
        pred_component = predictions[index] if predictions is not None and index < predictions.shape[0] else None
        component_rows.append(
            {
                "component": names[index] if index < len(names) else f"component_{index + 1}",
                "true_energy": float(true_energies[index]),
                "pred_energy": float(predicted_energies[index]),
                "residual_energy": float(residual_energies[index]),
                "residual_energy_ratio": float(residual_energies[index] / (true_energies[index] + EPS)),
                "dominant_frequency": _dominant_frequency(true_component, fs),
                "spectral_centroid": _spectral_centroid(true_component, fs),
                "bandwidth": _spectral_bandwidth(true_component, fs),
                "chirp_likeness": _chirp_likeness_score(true_component, fs),
            }
        )
    summary["component_descriptors"] = component_rows
    return summary


def build_explanation_text(
    component_df: pd.DataFrame,
    *,
    macro_metrics: dict[str, Any] | None = None,
    clean_metrics: dict[str, Any] | None = None,
    difficulty_summary: dict[str, Any] | None = None,
) -> str:
    """Build explanation text.
    
    Args:
        component_df: Input dataframe.
    """
    if component_df.empty:
        return "No per-component metrics were available for interpretation."

    ranking = component_df.sort_values(["nmse", "residual_energy_ratio"], ascending=False, na_position="last")
    dominant = ranking.iloc[0]
    labels = list(dominant.get("failure_labels", []))
    parts = []
    parts.append(
        f"The {dominant['component']} component is the main failure case. "
        f"It shows {dominant.get('primary_failure_mode', 'mixed')} problems"
    )

    reasons = []
    corr = _safe_float(dominant.get("corr"))
    snr = _safe_float(dominant.get("snr_db"))
    energy_ratio = _safe_float(dominant.get("energy_ratio"))
    spectral_convergence = _safe_float(dominant.get("spectral_convergence"))
    if np.isfinite(corr):
        reasons.append(f"correlation {corr:.2f}")
    if np.isfinite(snr):
        reasons.append(f"SNR {snr:.2f} dB")
    if np.isfinite(spectral_convergence):
        reasons.append(f"spectral convergence {spectral_convergence:.2f}")
    if np.isfinite(energy_ratio):
        reasons.append(f"energy ratio {energy_ratio:.2f}")
    if reasons:
        parts[-1] += " with " + ", ".join(reasons)
    parts[-1] += "."

    suppressed = component_df[component_df["failure_label_text"].str.contains("component_suppression", na=False)]["component"].tolist()
    overpredicted = component_df[component_df["failure_label_text"].str.contains("component_overprediction", na=False)]["component"].tolist()
    spectral = component_df[component_df["failure_label_text"].str.contains("spectral_mismatch", na=False)]["component"].tolist()
    shape = component_df[component_df["failure_label_text"].str.contains("shape_failure", na=False)]["component"].tolist()

    if suppressed:
        parts.append(f"Suppressed components: {', '.join(suppressed)}.")
    if overpredicted:
        parts.append(f"Overpredicted components: {', '.join(overpredicted)}.")
    if spectral:
        parts.append(f"Spectral mismatch appears strongest in {', '.join(spectral)}.")
    if shape:
        parts.append(f"Shape mismatch is visible in {', '.join(shape)}.")

    if macro_metrics:
        macro_corr = _safe_float(macro_metrics.get("corr"))
        macro_snr = _safe_float(macro_metrics.get("snr_db"))
        if np.isfinite(macro_corr) and np.isfinite(macro_snr):
            parts.append(f"Overall macro correlation is {macro_corr:.3f} with macro SNR {macro_snr:.3f} dB.")

    if clean_metrics:
        clean_corr = _safe_float(clean_metrics.get("corr"))
        clean_nmse = _safe_float(clean_metrics.get("nmse"))
        if np.isfinite(clean_corr) and np.isfinite(clean_nmse):
            parts.append(f"Clean-sum reconstruction has correlation {clean_corr:.3f} and NMSE {clean_nmse:.3f}.")

    if difficulty_summary:
        overlap = _safe_float(difficulty_summary.get("max_pairwise_spectral_overlap"))
        weakest = _safe_float(difficulty_summary.get("weakest_component_ratio"))
        if np.isfinite(overlap) and overlap > 0.55:
            parts.append("The signal itself has high spectral overlap, which increases separation difficulty.")
        if np.isfinite(weakest) and weakest < 0.20:
            parts.append("One component is much weaker than the others, making suppression errors more likely.")

    if "poor_source_separation" in labels or any("spectral_mismatch" in text for text in component_df["failure_label_text"].tolist()):
        parts.append("This pattern suggests component mixing or insufficient model capacity for the signal structure.")

    return " ".join(parts)


def build_signal_difficulty_text(summary: dict[str, Any]) -> str:
    """Build signal difficulty text.
    
    Args:
        summary: Summary payload.
    """
    if not summary:
        return "No signal difficulty descriptors were available."
    statements = []
    overlap = _safe_float(summary.get("max_pairwise_spectral_overlap"))
    snr = _safe_float(summary.get("estimated_input_snr_db"))
    weakest = _safe_float(summary.get("weakest_component_ratio"))
    amplitude_ratio = _safe_float(summary.get("amplitude_ratio_strongest_to_weakest"))

    if np.isfinite(overlap):
        if overlap > 0.60:
            statements.append("The signal has strong spectral overlap between components.")
        elif overlap > 0.35:
            statements.append("The signal has moderate spectral overlap.")
        else:
            statements.append("The components are relatively well separated spectrally.")
    if np.isfinite(snr):
        if snr < 5.0:
            statements.append("The observed mixture is noisy.")
        elif snr < 12.0:
            statements.append("The mixture has moderate noise contamination.")
        else:
            statements.append("The mixture noise level is relatively mild.")
    if np.isfinite(weakest) and np.isfinite(amplitude_ratio):
        if weakest < 0.20:
            statements.append(f"The weakest component is small relative to the strongest one (strength ratio {amplitude_ratio:.2f}).")
    return " ".join(statements) if statements else "No strong difficulty cues were detected from the signal descriptors."


def compare_model_results(model_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare model results.
    
    Args:
        model_results: Model result records to compare.
    """
    snn = [item for item in model_results if item.get("model_type") == "SNN"]
    dnn = [item for item in model_results if item.get("model_type") == "DNN"]
    best_snn = max(snn, key=lambda item: _safe_float(item.get("macro_corr")), default=None)
    best_dnn = max(dnn, key=lambda item: _safe_float(item.get("macro_corr")), default=None)
    comparison: dict[str, Any] = {
        "best_snn": best_snn,
        "best_dnn": best_dnn,
        "component_comparison": pd.DataFrame(),
    }
    if best_snn is None or best_dnn is None:
        return comparison

    snn_df = best_snn.get("component_table", pd.DataFrame())
    dnn_df = best_dnn.get("component_table", pd.DataFrame())
    if snn_df.empty or dnn_df.empty:
        return comparison

    merged = snn_df.merge(dnn_df, on="component", suffixes=("_snn", "_dnn"))
    rows = []
    for _, row in merged.iterrows():
        rows.append(
            {
                "component": row["component"],
                "corr_snn": row.get("corr_snn"),
                "corr_dnn": row.get("corr_dnn"),
                "corr_gap": _safe_float(row.get("corr_dnn")) - _safe_float(row.get("corr_snn")),
                "snr_snn": row.get("snr_db_snn"),
                "snr_dnn": row.get("snr_db_dnn"),
                "snr_gap": _safe_float(row.get("snr_db_dnn")) - _safe_float(row.get("snr_db_snn")),
                "residual_energy_ratio_snn": row.get("residual_energy_ratio_snn"),
                "residual_energy_ratio_dnn": row.get("residual_energy_ratio_dnn"),
                "spectral_error_snn": row.get("spectral_convergence_snn"),
                "spectral_error_dnn": row.get("spectral_convergence_dnn"),
                "better_model": best_dnn["display_name"]
                if _safe_float(row.get("corr_dnn")) >= _safe_float(row.get("corr_snn"))
                else best_snn["display_name"],
                "failure_labels_snn": row.get("failure_label_text_snn", ""),
                "failure_labels_dnn": row.get("failure_label_text_dnn", ""),
            }
        )
    comparison["component_comparison"] = pd.DataFrame(rows)
    comparison["macro_corr_gap"] = _safe_float(best_dnn.get("macro_corr")) - _safe_float(best_snn.get("macro_corr"))
    comparison["macro_snr_gap"] = _safe_float(best_dnn.get("macro_snr")) - _safe_float(best_snn.get("macro_snr"))
    return comparison


def _base_axis(figsize: tuple[float, float] = (9.5, 4.6)) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def figure_prediction_preview(
    time_axis: np.ndarray,
    mixture: np.ndarray,
    components: np.ndarray | None,
    prediction: np.ndarray,
    component_names: list[str] | None,
    display_name: str,
) -> plt.Figure:
    """Build the prediction preview figure.
    
    Args:
        mixture: Observed mixed signal.
    """
    names = component_names or [f"component_{index + 1}" for index in range(prediction.shape[0])]
    rows = prediction.shape[0] + 2
    fig, axes = plt.subplots(rows, 1, figsize=(12.0, 2.2 * rows), sharex=True)
    clean_sum_pred = np.sum(prediction, axis=0)
    axes[0].plot(time_axis, mixture, color="black", linewidth=1.0, label="Mixture")
    axes[0].set_title(f"Signal preview: {display_name}")
    axes[0].legend(loc="upper right")

    axes[1].plot(time_axis, clean_sum_pred, linewidth=1.0, label="Predicted reconstruction")
    if components is not None:
        clean_sum_true = np.sum(components, axis=0)
        axes[1].plot(time_axis, clean_sum_true, linewidth=1.0, alpha=0.85, label="True reconstruction")
        axes[1].plot(time_axis, clean_sum_true - clean_sum_pred, linewidth=0.8, alpha=0.7, label="Residual")
    axes[1].legend(loc="upper right")
    axes[1].set_title("Reconstruction")

    for index in range(prediction.shape[0]):
        axes[index + 2].plot(time_axis, prediction[index], linewidth=0.95, label=f"Pred {names[index]}")
        if components is not None and index < components.shape[0]:
            axes[index + 2].plot(time_axis, components[index], linewidth=0.95, alpha=0.85, label=f"True {names[index]}")
        axes[index + 2].legend(loc="upper right")
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    return fig


def figure_component_metric_comparison(
    comparison_df: pd.DataFrame,
    *,
    metric_snn: str,
    metric_dnn: str,
    title: str,
    ylabel: str,
) -> plt.Figure | None:
    """Build the component metric comparison figure.
    
    Args:
        title: Chart title.
        comparison_df: Input dataframe.
    """
    if comparison_df.empty:
        return None
    fig, ax = _base_axis()
    x = np.arange(len(comparison_df))
    width = 0.38
    ax.bar(x - width / 2, comparison_df[metric_snn], width=width, label="Best SNN", color="#1f77b4")
    ax.bar(x + width / 2, comparison_df[metric_dnn], width=width, label="Best DNN", color="#d62728")
    ax.set_xticks(x)
    ax.set_xticklabels([fill(str(value), width=14) for value in comparison_df["component"]], rotation=20, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig


def figure_failure_label_heatmap(comparison_df: pd.DataFrame) -> plt.Figure | None:
    """Build the failure label heatmap figure.
    
    Args:
        comparison_df: Input dataframe.
    """
    if comparison_df.empty:
        return None
    labels = sorted(
        {
            item
            for column in ["failure_labels_snn", "failure_labels_dnn"]
            for value in comparison_df[column].fillna("")
            for item in [part.strip() for part in str(value).split(",") if part.strip()]
        }
    )
    if not labels:
        return None
    matrix = np.zeros((len(comparison_df), len(labels)), dtype=float)
    for row_index, row in comparison_df.iterrows():
        label_text = f"{row.get('failure_labels_snn', '')}, {row.get('failure_labels_dnn', '')}"
        active = {item.strip() for item in label_text.split(",") if item.strip()}
        for col_index, label in enumerate(labels):
            matrix[row_index, col_index] = 1.0 if label in active else 0.0
    fig, ax = _base_axis((11.0, max(4.2, 0.55 * len(comparison_df) + 1.2)))
    image = ax.imshow(matrix, aspect="auto", cmap="Reds", vmin=0.0, vmax=1.0)
    ax.set_yticks(np.arange(len(comparison_df)))
    ax.set_yticklabels([fill(str(value), width=14) for value in comparison_df["component"]])
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels([fill(label, width=14) for label in labels], rotation=35, ha="right")
    ax.set_title("Failure label comparison across best SNN and best DNN")
    fig.colorbar(image, ax=ax, shrink=0.8)
    fig.tight_layout()
    return fig


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, pd.DataFrame):
        return value.to_dict("records")
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return value.item()
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save_diagnostic_artifact(
    *,
    base_name: str,
    time_axis: np.ndarray,
    mixture: np.ndarray,
    components: np.ndarray | None,
    model_result: dict[str, Any],
    signal_settings: dict[str, Any],
    comparison_summary: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    """Save diagnostic artifact.
    
    Args:
        mixture: Observed mixed signal.
    """
    out_dir = APP_EXPORTS_DIR / "test_signal_diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_name = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in base_name).strip("_").lower()
    metadata_path = out_dir / f"{safe_name}.json"
    arrays_path = out_dir / f"{safe_name}.npz"

    prediction = np.asarray(model_result.get("prediction"), dtype=float)
    residuals = None
    if components is not None and prediction.shape == components.shape:
        residuals = components - prediction
    np.savez_compressed(
        arrays_path,
        time_axis=np.asarray(time_axis, dtype=float),
        mixture=np.asarray(mixture, dtype=float),
        true_components=np.asarray(components, dtype=float) if components is not None else np.array([], dtype=float),
        predicted_components=prediction,
        residuals=np.asarray(residuals, dtype=float) if residuals is not None else np.array([], dtype=float),
    )

    payload = {
        "model_name": model_result.get("model_name"),
        "display_name": model_result.get("display_name"),
        "model_type": model_result.get("model_type"),
        "checkpoint": model_result.get("checkpoint"),
        "load_message": model_result.get("load_msg"),
        "signal_settings": signal_settings,
        "component_names": model_result.get("component_names", []),
        "macro_metrics": model_result.get("macro_metrics", {}),
        "clean_sum_metrics": model_result.get("clean_sum_metrics", {}),
        "per_component_metrics": model_result.get("component_table", pd.DataFrame()).to_dict("records"),
        "diagnostic_descriptors": model_result.get("difficulty_summary", {}),
        "failure_labels": model_result.get("component_table", pd.DataFrame())[["component", "failure_label_text"]].to_dict("records")
        if isinstance(model_result.get("component_table"), pd.DataFrame) and not model_result["component_table"].empty
        else [],
        "automatic_explanation": model_result.get("explanation", ""),
        "comparison_summary": comparison_summary or {},
        "arrays_file": str(arrays_path.relative_to(REPO_ROOT)),
    }
    metadata_path.write_text(json.dumps(_json_ready(payload), indent=2), encoding="utf-8")
    return metadata_path, arrays_path
