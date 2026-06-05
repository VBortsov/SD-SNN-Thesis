from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def reconstruction_overview_figure(
    t: np.ndarray,
    mixture: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    component_names: list[str] | None = None,
    t_min: float | None = None,
    t_max: float | None = None,
):
    """Reconstruction overview figure.
    
    Args:
        mixture: Observed mixed signal.
        y_true: Target component signals.
        y_pred: Predicted component signals.
    """
    mask = np.ones_like(t, dtype=bool)
    if t_min is not None and t_max is not None:
        mask = (t >= t_min) & (t <= t_max)
    tt = t[mask]
    mixture_view = mixture[mask]
    y_true_view = y_true[:, mask]
    y_pred_view = y_pred[:, mask]

    n_components = y_true.shape[0]
    labels = component_names or [f"C{idx + 1}" for idx in range(n_components)]
    fig, axes = plt.subplots(n_components + 2, 1, figsize=(12, 2.3 * (n_components + 2)), sharex=True)
    axes[0].plot(tt, mixture_view, color="black", linewidth=1.2, label="Observed mixture")
    axes[0].set_title("Observed Mixture")
    axes[0].legend(loc="upper right")

    clean_true = np.sum(y_true_view, axis=0)
    clean_pred = np.sum(y_pred_view, axis=0)
    axes[1].plot(tt, clean_true, label="Clean-sum true", linewidth=1.2)
    axes[1].plot(tt, clean_pred, label="Clean-sum pred", linewidth=1.2, alpha=0.8)
    axes[1].set_title("Clean-Sum Reconstruction")
    axes[1].legend(loc="upper right")

    for idx in range(n_components):
        label = labels[idx] if idx < len(labels) else f"C{idx + 1}"
        axes[idx + 2].plot(tt, y_true_view[idx], label=f"True {label}", linewidth=1.1)
        axes[idx + 2].plot(tt, y_pred_view[idx], label=f"Pred {label}", linewidth=1.0, alpha=0.85)
        axes[idx + 2].plot(tt, y_true_view[idx] - y_pred_view[idx], label="Residual", linewidth=0.9, alpha=0.6)
        axes[idx + 2].legend(loc="upper right")
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    return fig


def reconstruction_comparison_figure(
    t: np.ndarray,
    mixture: np.ndarray,
    y_true: np.ndarray,
    predictions: list[tuple[str, np.ndarray]],
    t_min: float | None = None,
    t_max: float | None = None,
):
    """Reconstruction comparison figure.
    
    Args:
        mixture: Observed mixed signal.
        y_true: Target component signals.
    """
    mask = np.ones_like(t, dtype=bool)
    if t_min is not None and t_max is not None:
        mask = (t >= t_min) & (t <= t_max)
    tt = t[mask]
    mixture_view = mixture[mask]
    clean_true = np.sum(y_true[:, mask], axis=0)

    fig, axes = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True)
    axes[0].plot(tt, mixture_view, color="black", linewidth=1.2, label="Observed mixture")
    axes[0].set_title("Observed Mixture")
    axes[0].legend(loc="upper right")

    axes[1].plot(tt, clean_true, label="Clean-sum true", linewidth=1.4, color="black")
    for label, y_pred in predictions:
        axes[1].plot(tt, np.sum(y_pred[:, mask], axis=0), label=label, linewidth=1.1, alpha=0.85)
    axes[1].set_title("Clean-Sum Model Comparison")
    axes[1].set_xlabel("Time (s)")
    axes[1].legend(loc="upper right")

    fig.tight_layout()
    return fig


def fft_comparison_figure(freq: np.ndarray, true_mag: np.ndarray, pred_mag: np.ndarray):
    """Build FFT comparison figure.
    
    Args:
        freq: Frequency bins.
        true_mag: Reference FFT magnitude.
        pred_mag: Predicted FFT magnitude.
    """
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(freq, true_mag, label="True FFT magnitude")
    ax.plot(freq, pred_mag, label="Pred FFT magnitude", alpha=0.8)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    ax.legend(loc="upper right")
    ax.set_title("FFT Magnitude Comparison")
    fig.tight_layout()
    return fig
