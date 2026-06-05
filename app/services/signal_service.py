from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from generators.basic.basicSignalGenerator import basicSignalGenerator

PRESET_MODE = "preset"
PURE_MODE = "pure"
SUPPORTED_PRESETS = ["mixed", "harmonic", "amfm", "chirp"]
SUPPORTED_COMPONENT_TYPES = ["harmonic", "amfm", "chirp", "trend", "transient"]


@dataclass
class SignalConfig:
    """Configuration container."""
    signal_type: str
    n_components: int
    duration: float
    fs: int
    noise_level: float
    seed: int
    generation_mode: str = PRESET_MODE
    selected_component_types: list[str] | None = None


@dataclass
class GeneratedSignal:
    """Generated mixture, components, time axis, and labels."""
    t: np.ndarray
    components: np.ndarray
    mixture: np.ndarray
    component_names: list[str]


def build_time_axis(duration: float, fs: int) -> np.ndarray:
    """Build the sample time axis.
    
    Args:
        duration: Signal duration in seconds.
        fs: Sampling rate in Hz.
    """
    n = max(4, int(duration * fs))
    return np.linspace(0.0, duration, n, endpoint=False)


def _harmonic_component(t: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = int(rng.integers(1, 4))
    return basicSignalGenerator.harmonic_mixture_generator(
        t=t,
        amplitudes=rng.uniform(0.2, 1.0, size=n),
        frequencies=rng.uniform(2.0, 40.0, size=n),
        phases=rng.uniform(0.0, 2 * np.pi, size=n),
    )


def _amfm_component(t: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    amp_freq = float(rng.uniform(0.1, 1.0))
    inst_freq = float(rng.uniform(0.05, 0.5))
    return basicSignalGenerator.am_fm_mode_generator(
        t=t,
        amplitude_envelope=lambda tt: 1.0 + 0.3 * np.sin(2 * np.pi * amp_freq * tt),
        instantaneous_frequency=lambda tt: 8 + 6 * np.sin(2 * np.pi * inst_freq * tt),
        phase0=float(rng.uniform(0.0, 2 * np.pi)),
    )


def _chirp_component(t: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    coeffs = [
        float(rng.uniform(3.0, 15.0)),
        float(rng.uniform(-6.0, 18.0)),
        float(rng.uniform(-4.0, 4.0)),
    ]
    return basicSignalGenerator.chirp_signal_generator(
        t=t,
        coefficients=coeffs,
        amplitude=float(rng.uniform(0.4, 1.0)),
        phase0=float(rng.uniform(0.0, 2 * np.pi)),
    )


def _trend_component(t: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Generate a bounded slow baseline drift with no fast oscillation."""
    if t.size == 0:
        return np.array([], dtype=float)
    x = np.linspace(-1.0, 1.0, t.size)
    coeffs = rng.uniform(-0.35, 0.35, size=3)
    trend = coeffs[0] * x + coeffs[1] * (x**2 - np.mean(x**2)) + coeffs[2] * (x**3)
    trend = trend - np.mean(trend)
    max_abs = float(np.max(np.abs(trend))) if trend.size else 0.0
    if max_abs > 0:
        trend = trend / max_abs * float(rng.uniform(0.25, 0.8))
    return trend.astype(float)


def _transient_component(t: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Generate a localized non-oscillatory Gaussian pulse."""
    if t.size == 0:
        return np.array([], dtype=float)
    duration = max(float(t[-1] - t[0]), 1e-6)
    center = float(rng.uniform(t[0] + 0.15 * duration, t[0] + 0.85 * duration))
    width = float(rng.uniform(0.025 * duration, 0.09 * duration))
    amplitude = float(rng.uniform(0.4, 1.0))
    pulse = amplitude * np.exp(-0.5 * ((t - center) / width) ** 2)
    return pulse.astype(float)


def component_builders() -> dict[str, Callable[[np.ndarray, np.random.Generator], np.ndarray]]:
    """Component builders."""
    return {
        "harmonic": _harmonic_component,
        "amfm": _amfm_component,
        "chirp": _chirp_component,
        "trend": _trend_component,
        "transient": _transient_component,
    }


def make_unique_component_names(names: list[str]) -> list[str]:
    """Make unique component names.
    
    Args:
        names: Names to normalize or deduplicate.
    """
    counts: dict[str, int] = {}
    total = {name: names.count(name) for name in names}
    unique = []
    for name in names:
        counts[name] = counts.get(name, 0) + 1
        unique.append(f"{name}_{counts[name]}" if total[name] > 1 else name)
    return unique


def validate_pure_component_types(selected: list[str] | None) -> tuple[bool, str]:
    """Validate pure component types.
    
    Args:
        selected: Selected component or module names.
    """
    if not selected:
        return False, "Select at least one pure component family."
    invalid = [item for item in selected if item not in SUPPORTED_COMPONENT_TYPES]
    if invalid:
        return False, f"Unsupported component family: {', '.join(invalid)}."
    return True, ""


def infer_component_sequence(config: SignalConfig) -> list[str]:
    """Infer component sequence.
    
    Args:
        config: Configuration object for the operation.
    """
    mode = config.generation_mode or PRESET_MODE
    if mode == PURE_MODE:
        selected = list(config.selected_component_types or [])
        valid, msg = validate_pure_component_types(selected)
        if not valid:
            raise ValueError(msg)
        return selected

    count = max(1, int(config.n_components))
    if config.signal_type == "mixed":
        sequence = ["harmonic", "amfm", "chirp"]
        return [sequence[idx % len(sequence)] for idx in range(count)]
    kind = config.signal_type if config.signal_type in SUPPORTED_COMPONENT_TYPES else "harmonic"
    return [kind for _ in range(count)]


def generate_signal(config: SignalConfig) -> GeneratedSignal:
    """Generate a synthetic signal sample.
    
    Args:
        config: Configuration object for the operation.
    """
    rng = np.random.default_rng(config.seed)
    t = build_time_axis(config.duration, config.fs)
    builders = component_builders()
    component_types = infer_component_sequence(config)
    components = [builders[kind](t, rng) for kind in component_types]

    stacked = np.stack(components, axis=0).astype(float)
    mixture = np.sum(stacked, axis=0)
    if config.noise_level > 0:
        mixture = mixture + rng.normal(0.0, config.noise_level, size=mixture.shape)
    return GeneratedSignal(
        t=t,
        components=stacked,
        mixture=mixture,
        component_names=make_unique_component_names(component_types),
    )


def generate_components(config: SignalConfig) -> tuple[np.ndarray, np.ndarray]:
    """Backward-compatible API returning only time and clean components."""
    generated = generate_signal(config)
    return generated.t, generated.components


def fft_magnitude(x: np.ndarray, fs: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute fft magnitude.
    
    Args:
        x: Input tensor.
        fs: Sampling rate in Hz.
    """
    spec = np.abs(np.fft.rfft(x))
    freq = np.fft.rfftfreq(x.size, d=1.0 / fs)
    return freq, spec
