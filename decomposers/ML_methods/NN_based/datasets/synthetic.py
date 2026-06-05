import os

import numpy as np
import torch
from torch.utils.data import Dataset

from generators.basic.basicSignalGenerator import basicSignalGenerator
from decomposers.ML_methods.NN_based.datasets.utils import normalize_signal, random_weights, add_noise

SUPPORTED_COMPONENT_TYPES = ["harmonic", "amfm", "chirp", "trend", "transient"]
DEFAULT_COMPONENT_TYPES = ["harmonic", "amfm", "chirp"]


def _component_types_from_env():
    raw = os.environ.get("NN_COMPONENT_TYPES", "").strip()
    if not raw:
        return None
    return [item.strip() for item in raw.split(",") if item.strip()]


class SyntheticSignalDataset(Dataset):
    """
    Dataset for signal decomposition.

    By default it preserves the original 3-source setup:
    harmonic, amfm, chirp. Passing component_types, or setting the
    NN_COMPONENT_TYPES environment variable, selects an explicit source set.
    """

    def __init__(
        self,
        num_samples: int,
        signal_length: int = 1024,
        fs: int = 256,
        use_weights: bool = True,
        noise_range=(0.0, 0.05),
        component_types=None,
    ):
        """Initialize layers and settings."""
        self.num_samples = num_samples
        self.signal_length = signal_length
        self.fs = fs
        self.duration = signal_length / fs
        self.t = np.linspace(0, self.duration, signal_length, endpoint=False)

        self.use_weights = use_weights
        self.noise_range = noise_range
        selected = component_types or _component_types_from_env() or DEFAULT_COMPONENT_TYPES
        invalid = [name for name in selected if name not in SUPPORTED_COMPONENT_TYPES]
        if invalid:
            raise ValueError(f"Unsupported component types: {', '.join(invalid)}")
        if not selected:
            raise ValueError("At least one component type must be selected.")
        self.component_types = list(selected)

    def __len__(self):
        return self.num_samples

    # -------------------------
    # Component generators
    # -------------------------

    def _harmonic(self):
        n = np.random.randint(1, 4)
        return basicSignalGenerator.harmonic_mixture_generator(
            self.t,
            amplitudes=np.random.uniform(0.2, 1.0, size=n),
            frequencies=np.random.uniform(2.0, 40.0, size=n),
            phases=np.random.uniform(0.0, 2 * np.pi, size=n)
        )

    def _amfm(self):
        return basicSignalGenerator.am_fm_mode_generator(
            self.t,
            amplitude_envelope=lambda tt: 1.0 + 0.3 * np.sin(2 * np.pi * np.random.uniform(0.1, 1.0) * tt),
            instantaneous_frequency=lambda tt: 8 + 6 * np.sin(2 * np.pi * np.random.uniform(0.05, 0.5) * tt),
            phase0=np.random.uniform(0.0, 2 * np.pi)
        )

    def _chirp(self):
        order = np.random.choice([1, 2, 3])

        coeffs = [np.random.uniform(3.0, 15.0)]

        if order >= 1:
            coeffs.append(np.random.uniform(2.0, 25.0))
        if order >= 2:
            coeffs.append(np.random.uniform(-8.0, 8.0))
        if order >= 3:
            coeffs.append(np.random.uniform(-3.0, 3.0))

        return basicSignalGenerator.chirp_signal_generator(
            self.t,
            coefficients=coeffs,
            amplitude=np.random.uniform(0.5, 1.0),
            phase0=np.random.uniform(0.0, 2 * np.pi)
        )

    def _trend(self):
        x = np.linspace(-1.0, 1.0, self.signal_length)
        coeffs = np.random.uniform(-0.35, 0.35, size=3)
        trend = coeffs[0] * x + coeffs[1] * (x**2 - np.mean(x**2)) + coeffs[2] * (x**3)
        trend = trend - np.mean(trend)
        max_abs = float(np.max(np.abs(trend))) if trend.size else 0.0
        if max_abs > 0:
            trend = trend / max_abs * float(np.random.uniform(0.25, 0.8))
        return trend

    def _transient(self):
        duration = max(float(self.t[-1] - self.t[0]), 1e-6)
        center = float(np.random.uniform(self.t[0] + 0.15 * duration, self.t[0] + 0.85 * duration))
        width = float(np.random.uniform(0.025 * duration, 0.09 * duration))
        amplitude = float(np.random.uniform(0.4, 1.0))
        return amplitude * np.exp(-0.5 * ((self.t - center) / width) ** 2)

    def _component(self, component_type):
        builders = {
            "harmonic": self._harmonic,
            "amfm": self._amfm,
            "chirp": self._chirp,
            "trend": self._trend,
            "transient": self._transient,
        }
        return builders[component_type]()

    # -------------------------
    # Main logic
    # -------------------------

    def __getitem__(self, idx):
        components = np.stack(
            [normalize_signal(self._component(component_type)) for component_type in self.component_types],
            axis=0,
        )

        if self.use_weights:
            w = random_weights(len(self.component_types))
            components = components * w[:, None]

        mixture = np.sum(components, axis=0)
        mixture = normalize_signal(mixture)

        noise_std = np.random.uniform(*self.noise_range)
        mixture = add_noise(mixture, noise_std)

        mixture = mixture.astype(np.float32)[None, :]
        components = components.astype(np.float32)

        return torch.from_numpy(mixture), torch.from_numpy(components)
