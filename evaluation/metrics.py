import math
from typing import Dict, Tuple

import numpy as np


EPS = 1e-12


def _to_1d_float(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("Expected a 1D signal.")
    return x


def mse(y_true, y_pred) -> float:
    """Compute MSE.
    
    Args:
        y_true: Target component signals.
        y_pred: Predicted component signals.
    """
    y_true = _to_1d_float(y_true)
    y_pred = _to_1d_float(y_pred)
    return float(np.mean((y_true - y_pred) ** 2))



def rmse(y_true, y_pred) -> float:
    """Compute RMSE.
    
    Args:
        y_true: Target component signals.
        y_pred: Predicted component signals.
    """
    return float(math.sqrt(mse(y_true, y_pred)))



def mae(y_true, y_pred) -> float:
    """Compute MAE.
    
    Args:
        y_true: Target component signals.
        y_pred: Predicted component signals.
    """
    y_true = _to_1d_float(y_true)
    y_pred = _to_1d_float(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))



def normalized_mse(y_true, y_pred) -> float:
    """Compute normalized MSE.
    
    Args:
        y_true: Target component signals.
        y_pred: Predicted component signals.
    """
    y_true = _to_1d_float(y_true)
    denom = np.mean(y_true ** 2) + EPS
    return float(mse(y_true, y_pred) / denom)



def relative_l2_error(y_true, y_pred) -> float:
    """Compute relative L2 error.
    
    Args:
        y_true: Target component signals.
        y_pred: Predicted component signals.
    """
    y_true = _to_1d_float(y_true)
    y_pred = _to_1d_float(y_pred)
    return float(np.linalg.norm(y_true - y_pred) / (np.linalg.norm(y_true) + EPS))



def snr_db(y_true, y_pred) -> float:
    """Compute SNR in dB.
    
    Args:
        y_true: Target component signals.
        y_pred: Predicted component signals.
    """
    y_true = _to_1d_float(y_true)
    y_pred = _to_1d_float(y_pred)
    signal_power = np.mean(y_true ** 2)
    noise_power = np.mean((y_true - y_pred) ** 2)
    return float(10.0 * np.log10((signal_power + EPS) / (noise_power + EPS)))



def psnr_db(y_true, y_pred, data_range: float = None) -> float:
    """Compute PSNR in dB.
    
    Args:
        y_true: Target component signals.
        y_pred: Predicted component signals.
        data_range: Expected signal value range.
    """
    y_true = _to_1d_float(y_true)
    y_pred = _to_1d_float(y_pred)
    err = mse(y_true, y_pred)
    if data_range is None:
        data_range = float(np.max(y_true) - np.min(y_true))
        if data_range < EPS:
            data_range = 1.0
    return float(20.0 * np.log10(data_range / (math.sqrt(err) + EPS)))



def correlation_coefficient(y_true, y_pred) -> float:
    """Correlation coefficient.
    
    Args:
        y_true: Target component signals.
        y_pred: Predicted component signals.
    """
    y_true = _to_1d_float(y_true)
    y_pred = _to_1d_float(y_pred)
    if np.std(y_true) < EPS or np.std(y_pred) < EPS:
        return 0.0
    return float(np.corrcoef(y_true, y_pred)[0, 1])



def cosine_similarity(y_true, y_pred) -> float:
    """Compute cosine similarity.
    
    Args:
        y_true: Target component signals.
        y_pred: Predicted component signals.
    """
    y_true = _to_1d_float(y_true)
    y_pred = _to_1d_float(y_pred)
    num = np.dot(y_true, y_pred)
    denom = (np.linalg.norm(y_true) * np.linalg.norm(y_pred)) + EPS
    return float(num / denom)



def scale_invariant_sdr_db(y_true, y_pred) -> float:
    """Compute scale-invariant SDR in dB.
    
    Args:
        y_true: Target component signals.
        y_pred: Predicted component signals.
    """
    y_true = _to_1d_float(y_true)
    y_pred = _to_1d_float(y_pred)
    alpha = np.dot(y_pred, y_true) / (np.dot(y_true, y_true) + EPS)
    target = alpha * y_true
    noise = y_pred - target
    return float(10.0 * np.log10((np.sum(target ** 2) + EPS) / (np.sum(noise ** 2) + EPS)))



def spectral_convergence(y_true, y_pred) -> float:
    """Spectral convergence.
    
    Args:
        y_true: Target component signals.
        y_pred: Predicted component signals.
    """
    y_true = _to_1d_float(y_true)
    y_pred = _to_1d_float(y_pred)
    spec_true = np.abs(np.fft.rfft(y_true))
    spec_pred = np.abs(np.fft.rfft(y_pred))
    return float(np.linalg.norm(spec_true - spec_pred) / (np.linalg.norm(spec_true) + EPS))



def log_spectral_distance(y_true, y_pred) -> float:
    """Log spectral distance.
    
    Args:
        y_true: Target component signals.
        y_pred: Predicted component signals.
    """
    y_true = _to_1d_float(y_true)
    y_pred = _to_1d_float(y_pred)
    spec_true = np.abs(np.fft.rfft(y_true)) + EPS
    spec_pred = np.abs(np.fft.rfft(y_pred)) + EPS
    diff = np.log(spec_true) - np.log(spec_pred)
    return float(math.sqrt(np.mean(diff ** 2)))



def fft_magnitude_l1(y_true, y_pred) -> float:
    """Compute FFT magnitude L1 error.
    
    Args:
        y_true: Target component signals.
        y_pred: Predicted component signals.
    """
    y_true = _to_1d_float(y_true)
    y_pred = _to_1d_float(y_pred)
    spec_true = np.abs(np.fft.rfft(y_true))
    spec_pred = np.abs(np.fft.rfft(y_pred))
    return float(np.mean(np.abs(spec_true - spec_pred)))



def explained_variance_score(y_true, y_pred) -> float:
    """Explained variance score.
    
    Args:
        y_true: Target component signals.
        y_pred: Predicted component signals.
    """
    y_true = _to_1d_float(y_true)
    y_pred = _to_1d_float(y_pred)
    var_true = np.var(y_true)
    if var_true < EPS:
        return 0.0
    return float(1.0 - np.var(y_true - y_pred) / (var_true + EPS))



def energy_ratio(y_true, y_pred) -> float:
    """Compute energy ratio.
    
    Args:
        y_true: Target component signals.
        y_pred: Predicted component signals.
    """
    y_true = _to_1d_float(y_true)
    y_pred = _to_1d_float(y_pred)
    return float((np.sum(y_pred ** 2) + EPS) / (np.sum(y_true ** 2) + EPS))



def max_abs_error(y_true, y_pred) -> float:
    """Compute max abs error.
    
    Args:
        y_true: Target component signals.
        y_pred: Predicted component signals.
    """
    y_true = _to_1d_float(y_true)
    y_pred = _to_1d_float(y_pred)
    return float(np.max(np.abs(y_true - y_pred)))



def evaluate_signal(y_true, y_pred) -> Dict[str, float]:
    """Evaluate signal.
    
    Args:
        y_true: Target component signals.
        y_pred: Predicted component signals.
    """
    return {
        "mse": mse(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "nmse": normalized_mse(y_true, y_pred),
        "relative_l2": relative_l2_error(y_true, y_pred),
        "snr_db": snr_db(y_true, y_pred),
        "psnr_db": psnr_db(y_true, y_pred),
        "corr": correlation_coefficient(y_true, y_pred),
        "cosine_similarity": cosine_similarity(y_true, y_pred),
        "si_sdr_db": scale_invariant_sdr_db(y_true, y_pred),
        "spectral_convergence": spectral_convergence(y_true, y_pred),
        "log_spectral_distance": log_spectral_distance(y_true, y_pred),
        "fft_magnitude_l1": fft_magnitude_l1(y_true, y_pred),
        "explained_variance": explained_variance_score(y_true, y_pred),
        "energy_ratio": energy_ratio(y_true, y_pred),
        "max_abs_error": max_abs_error(y_true, y_pred),
    }
