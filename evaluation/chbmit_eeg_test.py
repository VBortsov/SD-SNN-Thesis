#!/usr/bin/env python3
"""
chbmit_eeg_test.py
==================
Tests trained decomposition models on real EEG data from the CHB-MIT
Scalp EEG Corpus (https://physionet.org/content/chbmit/1.0.0/).

CHB-MIT records at exactly 256 Hz with a bipolar 23-channel montage, which
is a near-perfect match for models trained on synthetic signals at 256 Hz
with components in the 2–40 Hz range (harmonic, AM-FM, chirp).

TWO EVALUATION MODES
--------------------
1. semi_synthetic  [default, quantitative]
   For each 4-second EEG window:
     - Generate synthetic harmonic + AM-FM + chirp components (known ground truth)
     - Mix them with a scaled copy of the EEG window as realistic background noise
     - Feed the combined signal to the model
     - Evaluate component recovery with the same metrics used in thesis (macro
       correlation, macro SNR, test loss)
   This gives real numbers directly comparable to the synthetic benchmark.

2. direct_eeg  [qualitative]
   Feed raw EEG windows directly as the mixture input (no ground truth).
   The model's three output components are saved as figures for visual/spectral
   inspection.  Useful for showing real-world behaviour in the thesis appendix.

USAGE
-----
# 1. Download CHB-MIT EDF files from PhysioNet and place them anywhere, e.g.:
#       data/chbmit/chb01/chb01_01.edf  ...  chb01_14.edf

# 2. Install dependencies (if not already present):
#       pip install mne matplotlib

# 3. Run (from project root):
#
#   Semi-synthetic on two models:
#       python evaluation/chbmit_eeg_test.py \\
#           --edf_files data/chbmit/chb01/chb01_01.edf data/chbmit/chb01/chb01_02.edf \\
#           --model_key tasnet \\
#           --checkpoint decomposers/ML_methods/NN_based/saved_models/tasnet/tasnet_best.pt \\
#           --mode semi_synthetic --eeg_alpha 0.05 --n_windows 200
#
#   Alpha sweep (tests robustness across noise levels):
#       python evaluation/chbmit_eeg_test.py \\
#           --edf_files data/chbmit/chb01/*.edf \\
#           --model_key tasnet \\
#           --checkpoint decomposers/ML_methods/NN_based/saved_models/tasnet/tasnet_best.pt \\
#           --mode semi_synthetic --alpha_sweep --n_windows 200
#
#   Direct EEG qualitative run:
#       python evaluation/chbmit_eeg_test.py \\
#           --edf_files data/chbmit/chb01/chb01_01.edf \\
#           --model_key tasnet \\
#           --checkpoint decomposers/ML_methods/NN_based/saved_models/tasnet/tasnet_best.pt \\
#           --mode direct_eeg --n_examples 10
"""

import sys
import os

# Ensure project root is on the path when run directly
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import argparse
import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe on any machine
import matplotlib.pyplot as plt

try:
    import mne
    mne.set_log_level("WARNING")
except ImportError:
    raise ImportError(
        "MNE-Python is required.  Install it with:\n"
        "    pip install mne"
    )

from decomposers.ML_methods.NN_based.datasets.synthetic import SyntheticSignalDataset
from decomposers.ML_methods.NN_based.datasets.utils import normalize_signal
from app.services.inference_service import load_model, run_inference
from evaluation.decomposition import evaluate_decomposition

# ──────────────────────────────────────────────────────────────────────────────
# Constants — must match training configuration exactly
# ──────────────────────────────────────────────────────────────────────────────

SIGNAL_LENGTH = 1024       # samples per window
FS = 256                   # Hz  (CHB-MIT native rate — no resampling needed)
DURATION = SIGNAL_LENGTH / FS   # 4 seconds
COMPONENT_TYPES = ["harmonic", "amfm", "chirp"]

# Default bandpass limits matching the training component frequency ranges:
#   harmonic  2–40 Hz
#   AM-FM     ~2–14 Hz
#   chirp     3–15 Hz (starting freq) up to higher instantaneous frequencies
FMIN = 2.0
FMAX = 40.0

# Preferred channel (midline, robust across subjects)
DEFAULT_CHANNEL = "FZ-CZ"


# ──────────────────────────────────────────────────────────────────────────────
# EDF loading helpers
# ──────────────────────────────────────────────────────────────────────────────

def _pick_channel(raw: "mne.io.BaseRaw", preferred: str) -> str:
    """Return the best available channel name, falling back gracefully."""
    available = raw.ch_names
    # exact match
    if preferred in available:
        return preferred
    # case-insensitive match
    upper_map = {c.upper(): c for c in available}
    if preferred.upper() in upper_map:
        return upper_map[preferred.upper()]
    # drop any EKG/ECG channels and use the first EEG channel
    eeg_candidates = [c for c in available if "EKG" not in c.upper() and "ECG" not in c.upper()]
    if eeg_candidates:
        chosen = eeg_candidates[0]
        print(f"  [warn] Channel '{preferred}' not found — using '{chosen}'")
        return chosen
    raise ValueError(f"No usable EEG channel found in {raw.filenames[0]}")


def load_edf_channel(
    edf_path: Path,
    channel: str = DEFAULT_CHANNEL,
    fmin: float = FMIN,
    fmax: float = FMAX,
) -> np.ndarray:
    """
    Load one channel from an EDF file, bandpass-filter it, and return the
    raw signal as a 1-D float64 array at 256 Hz.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)

    ch = _pick_channel(raw, channel)
    raw.pick_channels([ch])

    sfreq = raw.info["sfreq"]
    if abs(sfreq - FS) > 0.5:
        print(f"  [info] Resampling {edf_path.name}: {sfreq:.1f} Hz → {FS} Hz")
        raw.resample(FS, verbose=False)

    raw.filter(fmin, fmax, method="iir", verbose=False)

    data = raw.get_data()[0]   # (n_samples,)
    return data.astype(np.float64)


def extract_windows(signal: np.ndarray, window_size: int = SIGNAL_LENGTH) -> np.ndarray:
    """
    Split a long signal into non-overlapping windows of exactly `window_size`
    samples.  Any trailing samples that don't fill a full window are discarded.
    """
    n_windows = len(signal) // window_size
    trimmed = signal[: n_windows * window_size]
    return trimmed.reshape(n_windows, window_size)


# ──────────────────────────────────────────────────────────────────────────────
# Semi-synthetic dataset construction
# ──────────────────────────────────────────────────────────────────────────────

def build_semi_synthetic_batch(
    eeg_windows: np.ndarray,
    eeg_alpha: float = 0.05,
    seed: int = 42,
) -> tuple:
    """
    For each EEG window build one semi-synthetic test sample:

        mixture = normalize( synthetic_mixture  +  eeg_alpha * normalize(eeg_window) )

    where synthetic_mixture is generated exactly as during training (same
    generator, same normalization, same random weights) but with zero Gaussian
    noise — the EEG background provides the real-world noise instead.

    Parameters
    ----------
    eeg_windows : (N, 1024) array
    eeg_alpha   : amplitude scaling of the EEG background
                  0.00 → identical to synthetic training (no real-world noise)
                  0.05 → comparable to the training noise level (std ≈ 0.02–0.05)
                  0.10 → mild EEG interference
                  0.20 → moderate EEG interference (challenging for the model)
    seed        : base random seed for reproducibility

    Returns
    -------
    mixtures    : (N, 1024) float32  — model input
    components  : (N, 3, 1024) float32  — ground-truth components
    eeg_norms   : (N, 1024) float32  — normalized EEG windows (for reference)
    """
    n = len(eeg_windows)

    # Build a SyntheticSignalDataset with no Gaussian noise;
    # the EEG background acts as the noise source instead.
    dataset = SyntheticSignalDataset(
        num_samples=n,
        signal_length=SIGNAL_LENGTH,
        fs=FS,
        use_weights=True,
        noise_range=(0.0, 0.0),
        component_types=COMPONENT_TYPES,
    )

    mixtures = np.zeros((n, SIGNAL_LENGTH), dtype=np.float32)
    components = np.zeros((n, len(COMPONENT_TYPES), SIGNAL_LENGTH), dtype=np.float32)
    eeg_norms = np.zeros((n, SIGNAL_LENGTH), dtype=np.float32)

    rng_state = np.random.get_state()   # save global state

    for i in range(n):
        # Use a deterministic seed per sample so results are reproducible
        np.random.seed(seed + i)

        mix_tensor, comp_tensor = dataset[i]
        clean_mixture = mix_tensor.numpy()[0]          # (1024,) — already normalized
        comps = comp_tensor.numpy()                    # (3, 1024)

        eeg_norm = normalize_signal(eeg_windows[i]).astype(np.float32)

        combined = clean_mixture + eeg_alpha * eeg_norm
        combined = normalize_signal(combined).astype(np.float32)

        mixtures[i] = combined
        components[i] = comps
        eeg_norms[i] = eeg_norm

    np.random.set_state(rng_state)   # restore global state
    return mixtures, components, eeg_norms


# ──────────────────────────────────────────────────────────────────────────────
# Quantitative evaluation (semi-synthetic)
# ──────────────────────────────────────────────────────────────────────────────

def run_semi_synthetic_evaluation(
    model,
    mixtures: np.ndarray,
    components: np.ndarray,
    verbose: bool = True,
) -> dict:
    """
    Run model inference on all windows and compute the same aggregate metrics
    used in the thesis (macro correlation, macro SNR, test loss / MSE).

    Returns a dict that can be written directly to a JSON report.
    """
    n = len(mixtures)
    per_corr = []
    per_snr = []
    per_mse = []

    for i in range(n):
        y_pred, _ = run_inference(model, mixtures[i])          # (3, 1024)
        report = evaluate_decomposition(
            y_true=components[i],
            y_pred=y_pred,
            observed_mixture=mixtures[i],
            permutation_invariant=True,
        )
        macro = report["macro_average"]
        per_corr.append(macro["corr"])
        per_snr.append(macro["snr_db"])
        per_mse.append(macro["mse"])

        if verbose and (i + 1) % 50 == 0:
            print(f"    [{i+1}/{n}]  corr={np.mean(per_corr):.4f}  "
                  f"snr={np.mean(per_snr):.2f} dB  mse={np.mean(per_mse):.6f}")

    return {
        "n_windows": n,
        "macro_corr_mean": float(np.mean(per_corr)),
        "macro_corr_std":  float(np.std(per_corr)),
        "macro_snr_mean":  float(np.mean(per_snr)),
        "macro_snr_std":   float(np.std(per_snr)),
        "test_loss_mean":  float(np.mean(per_mse)),
        "test_loss_std":   float(np.std(per_mse)),
        "per_window_corr": [float(x) for x in per_corr],
        "per_window_snr":  [float(x) for x in per_snr],
        "per_window_mse":  [float(x) for x in per_mse],
    }


# ──────────────────────────────────────────────────────────────────────────────
# Alpha sweep
# ──────────────────────────────────────────────────────────────────────────────

ALPHA_SWEEP_VALUES = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]


def run_alpha_sweep(
    model,
    eeg_windows: np.ndarray,
    n_windows: int,
    seed: int,
    out_dir: Path,
    model_key: str,
) -> dict:
    """
    Evaluate the model at multiple eeg_alpha levels.
    Produces a table + figure showing how macro correlation and SNR degrade as
    real-world EEG interference increases.
    """
    eeg_subset = eeg_windows[:n_windows]
    sweep_results = {}

    print("\n  Alpha sweep:")
    print(f"  {'alpha':>6}  {'macro_corr':>12}  {'macro_snr_dB':>13}  {'test_loss':>10}")
    print("  " + "-" * 48)

    for alpha in ALPHA_SWEEP_VALUES:
        mixtures, components, _ = build_semi_synthetic_batch(eeg_subset, eeg_alpha=alpha, seed=seed)
        result = run_semi_synthetic_evaluation(model, mixtures, components, verbose=False)
        sweep_results[str(alpha)] = result
        print(f"  {alpha:>6.2f}  {result['macro_corr_mean']:>12.4f}  "
              f"{result['macro_snr_mean']:>13.2f}  {result['test_loss_mean']:>10.6f}")

    # ── Figure ────────────────────────────────────────────────────────────────
    alphas = [float(k) for k in sweep_results]
    corrs  = [sweep_results[k]["macro_corr_mean"] for k in sweep_results]
    snrs   = [sweep_results[k]["macro_snr_mean"]  for k in sweep_results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(alphas, corrs, "o-", color="#2980b9", lw=1.8, ms=6)
    ax1.axhline(0.937, color="#e74c3c", ls="--", lw=1.2, label="TasNet synthetic (0.937)")
    ax1.set_xlabel("EEG background amplitude (α)")
    ax1.set_ylabel("Macro Correlation")
    ax1.set_title("Macro Correlation vs EEG Noise Level")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.plot(alphas, snrs, "s-", color="#27ae60", lw=1.8, ms=6)
    ax2.axhline(9.39, color="#e74c3c", ls="--", lw=1.2, label="TasNet synthetic (9.39 dB)")
    ax2.set_xlabel("EEG background amplitude (α)")
    ax2.set_ylabel("Macro SNR (dB)")
    ax2.set_title("Macro SNR vs EEG Noise Level")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f"CHB-MIT EEG robustness — {model_key}", fontsize=11)
    plt.tight_layout()
    fig_path = out_dir / f"alpha_sweep_{model_key}.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Sweep figure saved: {fig_path}")

    return sweep_results


# ──────────────────────────────────────────────────────────────────────────────
# Direct EEG inference (qualitative)
# ──────────────────────────────────────────────────────────────────────────────

def run_direct_eeg(
    model,
    eeg_windows: np.ndarray,
    out_dir: Path,
    model_key: str,
    n_examples: int = 10,
):
    """
    Feed normalized EEG windows directly to the model and save time-domain
    and spectral figures for visual inspection.

    No ground truth is available; the outputs are labelled as "predicted
    component N" and can be compared qualitatively to expected EEG rhythms.
    """
    fig_dir = out_dir / "direct_eeg_figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    t = np.linspace(0, DURATION, SIGNAL_LENGTH)
    freqs = np.fft.rfftfreq(SIGNAL_LENGTH, d=1.0 / FS)
    indices = np.linspace(0, len(eeg_windows) - 1, n_examples, dtype=int)

    comp_colors = ["#e74c3c", "#2980b9", "#27ae60"]
    comp_labels = [f"Component {i} (predicted)" for i in range(len(COMPONENT_TYPES))]

    for window_idx in indices:
        eeg = normalize_signal(eeg_windows[window_idx]).astype(np.float32)
        y_pred, inference_ms = run_inference(model, eeg)

        n_rows = 1 + len(COMPONENT_TYPES)
        fig, axes = plt.subplots(n_rows, 2, figsize=(14, 3 * n_rows))

        # ── Input row ───────────────────────────────────────────────────────
        axes[0, 0].plot(t, eeg, color="k", lw=0.7)
        axes[0, 0].set_title("Input EEG  (normalized, 2–40 Hz bandpass)")
        axes[0, 0].set_ylabel("Amplitude")

        spec_in = np.abs(np.fft.rfft(eeg))
        axes[0, 1].plot(freqs, spec_in, color="k", lw=0.7)
        axes[0, 1].set_title("Input — FFT magnitude")
        axes[0, 1].set_ylabel("|FFT|")

        # ── Component rows ───────────────────────────────────────────────────
        for j, (color, label) in enumerate(zip(comp_colors, comp_labels)):
            ax_t = axes[j + 1, 0]
            ax_f = axes[j + 1, 1]

            ax_t.plot(t, y_pred[j], color=color, lw=0.7)
            ax_t.set_title(label)
            ax_t.set_ylabel("Amplitude")

            spec_c = np.abs(np.fft.rfft(y_pred[j]))
            ax_f.plot(freqs, spec_c, color=color, lw=0.7)
            ax_f.set_title(f"{label} — FFT magnitude")
            ax_f.set_ylabel("|FFT|")

        for ax in axes[-1]:
            ax.set_xlabel("Time (s)" if ax in axes[:, 0] else "Frequency (Hz)")

        fig.suptitle(
            f"Direct EEG inference | {model_key} | window {window_idx} | "
            f"inference {inference_ms:.1f} ms",
            fontsize=10,
        )
        plt.tight_layout()

        fig_path = fig_dir / f"window_{window_idx:05d}.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"  {n_examples} figures saved to: {fig_dir}")


# ──────────────────────────────────────────────────────────────────────────────
# Summary figure for semi-synthetic results
# ──────────────────────────────────────────────────────────────────────────────

def save_distribution_figure(result: dict, out_dir: Path, model_key: str, eeg_alpha: float):
    """
    Plot per-window distributions of macro correlation and SNR and compare
    to the synthetic benchmark values.
    """
    corrs = result["per_window_corr"]
    snrs  = result["per_window_snr"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.hist(corrs, bins=30, color="#2980b9", edgecolor="white", alpha=0.85)
    ax1.axvline(np.mean(corrs), color="#e74c3c", ls="--", lw=1.5,
                label=f"Mean = {np.mean(corrs):.4f}")
    ax1.set_xlabel("Macro Correlation")
    ax1.set_ylabel("Count")
    ax1.set_title(f"Macro Correlation (α={eeg_alpha})")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.hist(snrs, bins=30, color="#27ae60", edgecolor="white", alpha=0.85)
    ax2.axvline(np.mean(snrs), color="#e74c3c", ls="--", lw=1.5,
                label=f"Mean = {np.mean(snrs):.2f} dB")
    ax2.set_xlabel("Macro SNR (dB)")
    ax2.set_ylabel("Count")
    ax2.set_title(f"Macro SNR (α={eeg_alpha})")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(
        f"CHB-MIT semi-synthetic | {model_key} | {result['n_windows']} windows",
        fontsize=11,
    )
    plt.tight_layout()
    fig_path = out_dir / f"distribution_{model_key}_alpha{eeg_alpha:.2f}.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Distribution figure saved: {fig_path}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Test decomposition models on CHB-MIT real EEG data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--edf_files", nargs="+", required=True,
        help="One or more .edf file paths (e.g. data/chbmit/chb01/chb01_01.edf)",
    )
    p.add_argument(
        "--model_key", default="tasnet",
        help="Model key matching the registry (default: tasnet)",
    )
    p.add_argument(
        "--checkpoint", required=True,
        help="Path to the .pt checkpoint file",
    )
    p.add_argument(
        "--mode", choices=["semi_synthetic", "direct_eeg", "both"],
        default="semi_synthetic",
        help="Evaluation mode (default: semi_synthetic)",
    )
    p.add_argument(
        "--channel", default=DEFAULT_CHANNEL,
        help=f"EEG channel to extract (default: {DEFAULT_CHANNEL})",
    )
    p.add_argument(
        "--eeg_alpha", type=float, default=0.05,
        help=(
            "EEG background amplitude for semi_synthetic mode (default: 0.05). "
            "0.0 = no real EEG noise (identical to synthetic training), "
            "0.05 = mild (matches training noise level), "
            "0.20 = moderate interference."
        ),
    )
    p.add_argument(
        "--alpha_sweep", action="store_true",
        help="Run semi_synthetic at multiple alpha levels instead of a single one.",
    )
    p.add_argument(
        "--n_windows", type=int, default=200,
        help="Maximum number of 4-second EEG windows to evaluate (default: 200)",
    )
    p.add_argument(
        "--n_examples", type=int, default=10,
        help="Number of example figures for direct_eeg mode (default: 10)",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    p.add_argument(
        "--out_dir", default="evaluation/chbmit_results",
        help="Output directory for reports and figures (default: evaluation/chbmit_results)",
    )
    return p.parse_args()


def main():
    """Run the command-line entry point."""
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Model   : {args.model_key}")
    print(f"  Checkpoint: {args.checkpoint}")
    model, msg = load_model(
        args.model_key,
        out_channels=len(COMPONENT_TYPES),
        checkpoint_path=args.checkpoint,
    )
    model.eval()
    print(f"  {msg}")

    # ── Load and preprocess EEG files ─────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Loading EDF files (channel={args.channel}, bandpass={FMIN}–{FMAX} Hz)")
    all_eeg_segments = []
    for edf_path_str in args.edf_files:
        edf_path = Path(edf_path_str)
        if not edf_path.exists():
            print(f"  [skip] File not found: {edf_path}")
            continue
        try:
            seg = load_edf_channel(edf_path, channel=args.channel)
            all_eeg_segments.append(seg)
            print(f"  Loaded: {edf_path.name}  ({len(seg)/FS:.1f} s)")
        except Exception as exc:
            print(f"  [error] {edf_path.name}: {exc}")

    if not all_eeg_segments:
        print("\nNo EEG data could be loaded. Exiting.")
        sys.exit(1)

    combined_eeg = np.concatenate(all_eeg_segments)
    eeg_windows  = extract_windows(combined_eeg, window_size=SIGNAL_LENGTH)
    print(f"\n  Total EEG loaded : {len(combined_eeg)/FS:.1f} s")
    print(f"  4-second windows : {len(eeg_windows)}")

    # Clamp to requested limit
    eeg_windows = eeg_windows[: args.n_windows]
    print(f"  Windows to use   : {len(eeg_windows)}")

    # ── Build result dict ─────────────────────────────────────────────────────
    results = {
        "timestamp":           datetime.now().isoformat(),
        "model_key":           args.model_key,
        "checkpoint":          str(args.checkpoint),
        "channel":             args.channel,
        "edf_files":           [str(f) for f in args.edf_files],
        "n_eeg_windows_used":  len(eeg_windows),
        "mode":                args.mode,
        "eeg_alpha":           args.eeg_alpha,
        "component_types":     COMPONENT_TYPES,
        "signal_length":       SIGNAL_LENGTH,
        "fs":                  FS,
        "fmin":                FMIN,
        "fmax":                FMAX,
        "seed":                args.seed,
    }

    # ── Semi-synthetic evaluation ─────────────────────────────────────────────
    if args.mode in ("semi_synthetic", "both"):
        print(f"\n{'='*60}")

        if args.alpha_sweep:
            print("  [semi_synthetic] Alpha sweep ...")
            sweep = run_alpha_sweep(
                model, eeg_windows, len(eeg_windows),
                seed=args.seed, out_dir=out_dir, model_key=args.model_key,
            )
            results["alpha_sweep"] = sweep

        else:
            alpha = args.eeg_alpha
            print(f"  [semi_synthetic] Building batch  eeg_alpha={alpha} ...")
            mixtures, components, _ = build_semi_synthetic_batch(
                eeg_windows, eeg_alpha=alpha, seed=args.seed,
            )
            print(f"  Evaluating {len(mixtures)} windows ...")
            semi_res = run_semi_synthetic_evaluation(model, mixtures, components)
            results["semi_synthetic"] = semi_res

            print(f"\n  ── Results ─────────────────────────────────────────────")
            print(f"  Windows evaluated  : {semi_res['n_windows']}")
            print(f"  Macro Correlation  : {semi_res['macro_corr_mean']:.4f} "
                  f"± {semi_res['macro_corr_std']:.4f}")
            print(f"  Macro SNR (dB)     : {semi_res['macro_snr_mean']:.2f} "
                  f"± {semi_res['macro_snr_std']:.2f}")
            print(f"  Test Loss (MSE)    : {semi_res['test_loss_mean']:.6f} "
                  f"± {semi_res['test_loss_std']:.6f}")

            save_distribution_figure(semi_res, out_dir, args.model_key, alpha)

    # ── Direct EEG qualitative run ────────────────────────────────────────────
    if args.mode in ("direct_eeg", "both"):
        print(f"\n{'='*60}")
        print(f"  [direct_eeg] Running qualitative inference ({args.n_examples} examples) ...")
        run_direct_eeg(
            model, eeg_windows, out_dir,
            model_key=args.model_key, n_examples=args.n_examples,
        )

    # ── Save JSON report ──────────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = out_dir / f"chbmit_report_{args.model_key}_{ts}.json"
    report_path.write_text(json.dumps(results, indent=2, default=str))

    print(f"\n{'='*60}")
    print(f"  Report saved: {report_path}")
    print(f"  Done.")


if __name__ == "__main__":
    main()
