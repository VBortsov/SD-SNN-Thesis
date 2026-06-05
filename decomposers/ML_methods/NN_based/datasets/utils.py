import numpy as np


def normalize_signal(x, eps=1e-8):
    """Normalize signal.
    
    Args:
        x: Input tensor.
        eps: Small value used to avoid division by zero.
    """
    return x / (np.max(np.abs(x)) + eps)


def random_weights(n, low=0.3, high=1.2):
    """Random weights.
    
    Args:
        n: Number of values to generate.
        low: Lower sampling bound.
        high: Upper sampling bound.
    """
    return np.random.uniform(low, high, size=n)


def add_noise(x, noise_std):
    """Add noise.
    
    Args:
        x: Input tensor.
        noise_std: Standard deviation of additive noise.
    """
    return x + np.random.normal(0.0, noise_std, size=x.shape)