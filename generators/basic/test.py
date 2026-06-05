import numpy as np
import matplotlib.pyplot as plt

from generators.basic.basicSignalGenerator import basicSignalGenerator

fs = 1000
duration = 2.0
t = np.linspace(0, duration, int(fs * duration), endpoint=False)

# 1) Harmonic mixture
harmonic_signal = basicSignalGenerator.harmonic_mixture_generator(
    t,
    amplitudes=[1.0, 0.5, 0.3],
    frequencies=[5, 15, 30],
    phases=[0.0, np.pi / 4, np.pi / 2]
)

# 2) AM-FM mode
am_fm_signal = basicSignalGenerator.am_fm_mode_generator(
    t,
    amplitude_envelope=lambda tt: 1.0 + 0.4 * np.sin(2 * np.pi * 0.5 * tt),
    instantaneous_frequency=lambda tt: 10 + 5 * np.sin(2 * np.pi * 0.25 * tt),
    phase0=0.0
)

# 3) General polynomial chirp
# Instantaneous frequency:
# f(tau) = 5 + 37.5 * tau
# Over 2 seconds, this goes roughly from 5 Hz to 80 Hz
chirp_signal = basicSignalGenerator.chirp_signal_generator(
    t,
    coefficients=[5.0, 37.5],
    amplitude=1.0,
    phase0=0.0
)

plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(t, harmonic_signal)
plt.title("Harmonic Mixture Generator")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(t, am_fm_signal)
plt.title("AM-FM Mode Generator")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(t, chirp_signal)
plt.title("Polynomial Chirp Signal Generator")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.grid(True)

plt.tight_layout()
plt.show()