import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from generators.basic.basicSignalGenerator import basicSignalGenerator
from evaluation.decomposition import evaluate_decomposition, format_report


def main():
    """Run the command-line entry point."""
    fs = 1000
    duration = 2.0
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)

    harmonic = basicSignalGenerator.harmonic_mixture_generator(
        t,
        amplitudes=[1.0, 0.4, 0.2],
        frequencies=[5.0, 15.0, 30.0],
        phases=[0.0, np.pi / 6, np.pi / 3],
    )
    am_fm = basicSignalGenerator.am_fm_mode_generator(
        t,
        amplitude_envelope=lambda tt: 1.0 + 0.25 * np.sin(2 * np.pi * 0.5 * tt),
        instantaneous_frequency=lambda tt: 10.0 + 3.0 * np.sin(2 * np.pi * 0.25 * tt),
        phase0=0.2,
    )
    chirp = basicSignalGenerator.chirp_signal_generator(
        t,
        coefficients=[5.0, 12.0, -2.5],
        amplitude=0.8,
        phase0=0.1,
    )

    y_true = np.stack([harmonic, am_fm, chirp], axis=0)

    rng = np.random.default_rng(42)
    y_pred = np.stack([
        chirp + 0.05 * rng.normal(size=t.shape),
        harmonic + 0.03 * rng.normal(size=t.shape),
        am_fm + 0.04 * rng.normal(size=t.shape),
    ], axis=0)

    report = evaluate_decomposition(y_true, y_pred, permutation_invariant=True)
    print(format_report(report, component_names=["harmonic", "am_fm", "chirp"]))


if __name__ == "__main__":
    main()
