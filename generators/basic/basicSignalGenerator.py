import numpy as np


class basicSignalGenerator:
    """Static helpers for basic synthetic signal generation."""

    @staticmethod
    def harmonic_mixture_generator(
        t: np.ndarray,
        amplitudes,
        frequencies,
        phases=None
    ) -> np.ndarray:
        """
        Generate a harmonic mixture signal:
            x(t) = sum_k A_k * cos(2*pi*f_k*t + phi_k)

        Args:
            t: Time vector, shape (N,)
            amplitudes: List or array of amplitudes
            frequencies: List or array of frequencies in Hz
            phases: List or array of phases in radians; defaults to zeros

        Returns:
            Signal array of shape (N,)
        """
        amplitudes = np.asarray(amplitudes, dtype=float)
        frequencies = np.asarray(frequencies, dtype=float)

        if phases is None:
            phases = np.zeros_like(amplitudes)
        phases = np.asarray(phases, dtype=float)

        if not (len(amplitudes) == len(frequencies) == len(phases)):
            raise ValueError("amplitudes, frequencies, and phases must have the same length")

        x = np.zeros_like(t, dtype=float)
        for A, f, phi in zip(amplitudes, frequencies, phases):
            x += A * np.cos(2 * np.pi * f * t + phi)

        return x

    @staticmethod
    def am_fm_mode_generator(
        t: np.ndarray,
        amplitude_envelope,
        instantaneous_frequency,
        phase0: float = 0.0
    ) -> np.ndarray:
        """
        Generate an AM-FM mode signal:
            x(t) = a(t) * cos(phi(t))
        where:
            d(phi)/dt = 2*pi*f(t)

        Args:
            t: Time vector, shape (N,)
            amplitude_envelope: Scalar, array, or callable a(t)
            instantaneous_frequency: Scalar, array, or callable f(t) in Hz
            phase0: Initial phase in radians

        Returns:
            Signal array of shape (N,)
        """
        t = np.asarray(t, dtype=float)

        if callable(amplitude_envelope):
            a_t = np.asarray(amplitude_envelope(t), dtype=float)
        else:
            a_t = np.asarray(amplitude_envelope, dtype=float)
            if a_t.ndim == 0:
                a_t = np.full_like(t, a_t)

        if callable(instantaneous_frequency):
            f_t = np.asarray(instantaneous_frequency(t), dtype=float)
        else:
            f_t = np.asarray(instantaneous_frequency, dtype=float)
            if f_t.ndim == 0:
                f_t = np.full_like(t, f_t)

        if len(a_t) != len(t) or len(f_t) != len(t):
            raise ValueError("amplitude_envelope and instantaneous_frequency must match t in length")

        dt = np.mean(np.diff(t))
        phase = phase0 + 2 * np.pi * np.cumsum(f_t) * dt

        return a_t * np.cos(phase)

    @staticmethod
    def chirp_signal_generator(
        t: np.ndarray,
        coefficients,
        amplitude: float = 1.0,
        phase0: float = 0.0
    ) -> np.ndarray:
        """
        Generate a general polynomial chirp signal.

        Instantaneous frequency is defined as:
            f(tau) = c0 + c1*tau + c2*tau^2 + ... + cn*tau^n

        where:
            tau = t - t[0]

        The phase is obtained analytically:
            phi(tau) = phase0 + 2*pi * integral(f(tau) dtau)
                     = phase0 + 2*pi * sum_k [c_k * tau^(k+1)/(k+1)]

        Args:
            t: Time vector, shape (N,)
            coefficients: Sequence [c0, c1, c2, ..., cn] defining
                          the polynomial instantaneous frequency in Hz
            amplitude: Signal amplitude
            phase0: Initial phase in radians

        Returns:
            Signal array of shape (N,)
        """
        t = np.asarray(t, dtype=float)
        coefficients = np.asarray(coefficients, dtype=float)

        if t.ndim != 1:
            raise ValueError("t must be a 1D array")

        if len(t) < 2:
            raise ValueError("t must contain at least two time samples")

        if coefficients.ndim != 1 or len(coefficients) == 0:
            raise ValueError("coefficients must be a non-empty 1D array or list")

        tau = t - t[0]

        phase_integral = np.zeros_like(tau, dtype=float)
        for k, c_k in enumerate(coefficients):
            phase_integral += c_k * tau ** (k + 1) / (k + 1)

        phase = phase0 + 2 * np.pi * phase_integral

        return amplitude * np.cos(phase)
