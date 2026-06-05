"""Template evaluation script for shallow NN decomposer experiments.

This file provides a reusable baseline template that can be copied/adapted
for new model-specific evaluation scripts.
"""

import sys
from itertools import permutations
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from decomposers.ML_methods.NN_based.datasets.synthetic import SyntheticSignalDataset
from decomposers.ML_methods.NN_based.models_shallow.mlp.singlehead import SingleHeadMLPDecomposer


CHECKPOINT_PATH = "decomposers/ML_methods/NN_based/saved_models/mlp_singlehead/mlp_singlehead_best.pt"
BATCH_SIZE = 32
SIGNAL_LENGTH = 1024
FS = 256
NUM_TEST_SAMPLES = 300
OUT_CHANNELS = 3
COMPONENT_NAMES = ["harmonic", "amfm", "chirp"]
PERMUTATION_INVARIANT = True


def find_candidate_checkpoints(root: Path):
    """Find candidate checkpoints.
    
    Args:
        root: Directory to search from.
    """
    exts = {".pt", ".pth", ".ckpt"}
    candidates = []

    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in exts:
            candidates.append(path)

    return sorted(candidates)


def resolve_checkpoint():
    """Resolve checkpoint."""
    if CHECKPOINT_PATH is not None:
        ckpt = Path(CHECKPOINT_PATH)
        if ckpt.is_absolute():
            if ckpt.exists():
                return ckpt
        else:
            ckpt = PROJECT_ROOT / ckpt
            if ckpt.exists():
                return ckpt

        raise FileNotFoundError(f"Configured checkpoint does not exist:\n{ckpt}")

    candidates = find_candidate_checkpoints(PROJECT_ROOT)

    if len(candidates) == 0:
        raise FileNotFoundError(
            "No checkpoint file was found anywhere in your project.\n\n"
            "You need to either:\n"
            "1. train the model and save it, or\n"
            "2. set CHECKPOINT_PATH in this script to the correct .pt/.pth/.ckpt file."
        )

    if len(candidates) == 1:
        return candidates[0]

    print("Multiple checkpoint candidates found:\n")
    for i, path in enumerate(candidates, start=1):
        print(f"{i:2d}. {path}")

    raise RuntimeError(
        "\nMultiple checkpoint files were found. "
        "Set CHECKPOINT_PATH explicitly in this eval script."
    )


def mse(y_true, y_pred):
    """Compute MSE.
    
    Args:
        y_true: Target component signals.
        y_pred: Predicted component signals.
    """
    return float(np.mean((y_true - y_pred) ** 2))


def rmse(y_true, y_pred):
    """Compute RMSE.
    
    Args:
        y_true: Target component signals.
        y_pred: Predicted component signals.
    """
    return float(np.sqrt(mse(y_true, y_pred)))


def mae(y_true, y_pred):
    """Compute MAE.
    
    Args:
        y_true: Target component signals.
        y_pred: Predicted component signals.
    """
    return float(np.mean(np.abs(y_true - y_pred)))


def nmse(y_true, y_pred, eps=1e-12):
    """Compute normalized MSE.
    
    Args:
        y_true: Target component signals.
        y_pred: Predicted component signals.
        eps: Small value used to avoid division by zero.
    """
    denom = np.mean(y_true ** 2) + eps
    return float(np.mean((y_true - y_pred) ** 2) / denom)


def relative_l2(y_true, y_pred, eps=1e-12):
    """Compute relative L2 error.
    
    Args:
        y_true: Target component signals.
        y_pred: Predicted component signals.
        eps: Small value used to avoid division by zero.
    """
    num = np.linalg.norm(y_true - y_pred)
    den = np.linalg.norm(y_true) + eps
    return float(num / den)


def max_abs_error(y_true, y_pred):
    """Compute max abs error.
    
    Args:
        y_true: Target component signals.
        y_pred: Predicted component signals.
    """
    return float(np.max(np.abs(y_true - y_pred)))


def correlation(y_true, y_pred, eps=1e-12):
    """Compute correlation.
    
    Args:
        y_true: Target component signals.
        y_pred: Predicted component signals.
        eps: Small value used to avoid division by zero.
    """
    yt = y_true - np.mean(y_true)
    yp = y_pred - np.mean(y_pred)
    denom = (np.linalg.norm(yt) * np.linalg.norm(yp)) + eps
    return float(np.sum(yt * yp) / denom)


def cosine_similarity(y_true, y_pred, eps=1e-12):
    """Compute cosine similarity.
    
    Args:
        y_true: Target component signals.
        y_pred: Predicted component signals.
        eps: Small value used to avoid division by zero.
    """
    denom = (np.linalg.norm(y_true) * np.linalg.norm(y_pred)) + eps
    return float(np.sum(y_true * y_pred) / denom)


def explained_variance(y_true, y_pred, eps=1e-12):
    """Compute explained variance.
    
    Args:
        y_true: Target component signals.
        y_pred: Predicted component signals.
        eps: Small value used to avoid division by zero.
    """
    var_true = np.var(y_true)
    var_err = np.var(y_true - y_pred)
    return float(1.0 - var_err / (var_true + eps))


def snr_db(y_true, y_pred, eps=1e-12):
    """Compute SNR in dB.
    
    Args:
        y_true: Target component signals.
        y_pred: Predicted component signals.
        eps: Small value used to avoid division by zero.
    """
    signal_power = np.mean(y_true ** 2)
    noise_power = np.mean((y_true - y_pred) ** 2)
    return float(10.0 * np.log10((signal_power + eps) / (noise_power + eps)))


def si_sdr_db(y_true, y_pred, eps=1e-12):
    """Compute SI-SDR in dB.
    
    Args:
        y_true: Target component signals.
        y_pred: Predicted component signals.
        eps: Small value used to avoid division by zero.
    """
    y_true = y_true.astype(np.float64)
    y_pred = y_pred.astype(np.float64)

    alpha = np.dot(y_pred, y_true) / (np.dot(y_true, y_true) + eps)
    target = alpha * y_true
    noise = y_pred - target

    target_energy = np.sum(target ** 2)
    noise_energy = np.sum(noise ** 2)

    return float(10.0 * np.log10((target_energy + eps) / (noise_energy + eps)))


def energy_ratio(y_true, y_pred, eps=1e-12):
    """Compute energy ratio.
    
    Args:
        y_true: Target component signals.
        y_pred: Predicted component signals.
        eps: Small value used to avoid division by zero.
    """
    return float((np.sum(y_pred ** 2) + eps) / (np.sum(y_true ** 2) + eps))


def _fft_mag(x):
    return np.abs(np.fft.rfft(x))


def spectral_convergence(y_true, y_pred, eps=1e-12):
    """Spectral convergence.
    
    Args:
        y_true: Target component signals.
        y_pred: Predicted component signals.
        eps: Small value used to avoid division by zero.
    """
    s_true = _fft_mag(y_true)
    s_pred = _fft_mag(y_pred)
    num = np.linalg.norm(s_true - s_pred)
    den = np.linalg.norm(s_true) + eps
    return float(num / den)


def log_spectral_distance(y_true, y_pred, eps=1e-12):
    """Log spectral distance.
    
    Args:
        y_true: Target component signals.
        y_pred: Predicted component signals.
        eps: Small value used to avoid division by zero.
    """
    s_true = _fft_mag(y_true)
    s_pred = _fft_mag(y_pred)
    log_true = np.log(s_true + eps)
    log_pred = np.log(s_pred + eps)
    return float(np.sqrt(np.mean((log_true - log_pred) ** 2)))


def fft_magnitude_l1(y_true, y_pred):
    """Compute FFT magnitude L1 error.
    
    Args:
        y_true: Target component signals.
        y_pred: Predicted component signals.
    """
    s_true = _fft_mag(y_true)
    s_pred = _fft_mag(y_pred)
    return float(np.mean(np.abs(s_true - s_pred)))


def best_permutation_match(true_components, pred_components):
    """Best permutation match.
    
    Args:
        true_components: Project value for this call.
        pred_components: Project value for this call.
    """
    num_components = true_components.shape[0]
    best_perm = None
    best_score = -np.inf

    for perm in permutations(range(num_components)):
        score = 0.0
        for i, j in enumerate(perm):
            score += correlation(true_components[i], pred_components[j])
        if score > best_score:
            best_score = score
            best_perm = perm

    aligned_pred = pred_components[list(best_perm)]
    return aligned_pred, best_perm, float(best_score)


def evaluate_component(y_true, y_pred):
    """Evaluate component.
    
    Args:
        y_true: Target component signals.
        y_pred: Predicted component signals.
    """
    return {
        "mse": mse(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "nmse": nmse(y_true, y_pred),
        "relative_l2": relative_l2(y_true, y_pred),
        "max_abs_error": max_abs_error(y_true, y_pred),
        "corr": correlation(y_true, y_pred),
        "cosine_similarity": cosine_similarity(y_true, y_pred),
        "explained_variance": explained_variance(y_true, y_pred),
        "snr_db": snr_db(y_true, y_pred),
        "si_sdr_db": si_sdr_db(y_true, y_pred),
        "energy_ratio": energy_ratio(y_true, y_pred),
        "spectral_convergence": spectral_convergence(y_true, y_pred),
        "log_spectral_distance": log_spectral_distance(y_true, y_pred),
        "fft_magnitude_l1": fft_magnitude_l1(y_true, y_pred),
    }


def evaluate_sample(mixture, true_components, pred_components, permutation_invariant=True):
    """Evaluate sample.
    
    Args:
        mixture: Observed mixed signal.
    """
    if mixture.ndim == 2:
        mixture = mixture.squeeze(0)

    if permutation_invariant:
        pred_components, perm, perm_score = best_permutation_match(true_components, pred_components)
    else:
        perm = tuple(range(true_components.shape[0]))
        perm_score = None

    component_metrics = []
    for i in range(true_components.shape[0]):
        component_metrics.append(evaluate_component(true_components[i], pred_components[i]))

    clean_sum_true = np.sum(true_components, axis=0)
    clean_sum_pred = np.sum(pred_components, axis=0)
    clean_sum_metrics = evaluate_component(clean_sum_true, clean_sum_pred)
    observed_mixture_metrics = evaluate_component(mixture, clean_sum_pred)

    return {
        "permutation": perm,
        "permutation_score": perm_score,
        "components": component_metrics,
        "clean_sum": clean_sum_metrics,
        "observed_mixture": observed_mixture_metrics,
        # Backward compatibility shim for older consumers.
        "mixture": clean_sum_metrics,
    }


def aggregate_results(results, component_names=None):
    """Aggregate results.
    
    Args:
        results: Project value for this call.
        component_names: Names for predicted or target components.
    """
    if component_names is None:
        component_names = [f"component_{i}" for i in range(len(results[0]["components"]))]

    output = {
        "components": {},
        "clean_sum": {},
        "observed_mixture": {},
        "mixture": {},
        "global_average": {},
    }

    metric_names = list(results[0]["components"][0].keys())

    for comp_idx, comp_name in enumerate(component_names):
        output["components"][comp_name] = {}
        for metric in metric_names:
            values = [r["components"][comp_idx][metric] for r in results]
            output["components"][comp_name][metric] = float(np.mean(values))

    for metric in metric_names:
        clean_sum_vals = [r["clean_sum"][metric] for r in results]
        observed_vals = [r["observed_mixture"][metric] for r in results]
        output["clean_sum"][metric] = float(np.mean(clean_sum_vals))
        output["observed_mixture"][metric] = float(np.mean(observed_vals))
    # Backward compatibility shim for older consumers.
    output["mixture"] = output["clean_sum"]

    for metric in metric_names:
        vals = [output["components"][name][metric] for name in component_names]
        output["global_average"][metric] = float(np.mean(vals))

    return output


def print_summary(summary):
    """Print summary.
    
    Args:
        summary: Summary payload.
    """
    print("\n" + "=" * 90)
    print("SINGLE-HEAD MLP EVALUATION SUMMARY")
    print("=" * 90)

    print("\n[Global average over components]")
    for k, v in summary["global_average"].items():
        print(f"{k:>24}: {v: .6f}")

    print("\n[Clean-sum reconstruction: sum(true_components) vs sum(pred_components)]")
    for k, v in summary["clean_sum"].items():
        print(f"{k:>24}: {v: .6f}")

    print("\n[Observed-mixture reconstruction: observed input vs sum(pred_components)]")
    for k, v in summary["observed_mixture"].items():
        print(f"{k:>24}: {v: .6f}")

    print("\n[Per-component metrics]")
    for comp_name, metrics_dict in summary["components"].items():
        print(f"\n--- {comp_name} ---")
        for k, v in metrics_dict.items():
            print(f"{k:>24}: {v: .6f}")


@torch.no_grad()
def evaluate_model(model, loader, device, component_names=None, permutation_invariant=True):
    """Evaluate model."""
    model.eval()
    all_results = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        y_hat = model(x)

        x_np = x.cpu().numpy()
        y_np = y.cpu().numpy()
        y_hat_np = y_hat.cpu().numpy()

        batch_size = x_np.shape[0]
        for b in range(batch_size):
            result = evaluate_sample(
                mixture=x_np[b],
                true_components=y_np[b],
                pred_components=y_hat_np[b],
                permutation_invariant=permutation_invariant,
            )
            all_results.append(result)

    summary = aggregate_results(all_results, component_names=component_names)
    return summary, all_results


def main() -> None:
    """Run the command-line entry point."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Project root: {PROJECT_ROOT}")

    checkpoint_path = resolve_checkpoint()
    print(f"Using checkpoint: {checkpoint_path}")

    test_ds = SyntheticSignalDataset(
        num_samples=NUM_TEST_SAMPLES,
        signal_length=SIGNAL_LENGTH,
        fs=FS,
    )
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = SingleHeadMLPDecomposer(in_channels=1, out_channels=OUT_CHANNELS).to(device)

    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)

    criterion = nn.MSELoss()
    model.eval()
    total_loss = 0.0
    total_count = 0

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            y_hat = model(x)
            loss = criterion(y_hat, y)

            total_loss += loss.item() * x.size(0)
            total_count += x.size(0)

    avg_loss = total_loss / max(total_count, 1)
    print(f"\nTest MSE loss: {avg_loss:.6f}")

    summary, _ = evaluate_model(
        model=model,
        loader=test_loader,
        device=device,
        component_names=COMPONENT_NAMES,
        permutation_invariant=PERMUTATION_INVARIANT,
    )

    print_summary(summary)


if __name__ == "__main__":
    main()
