from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from app.services.signal_service import (
    PURE_MODE,
    SUPPORTED_COMPONENT_TYPES,
    SignalConfig,
    fft_magnitude,
    generate_signal,
    validate_pure_component_types,
)


def render() -> None:
    """Render the Streamlit view."""
    st.title("Signal Generator Lab")
    generation_label = st.radio("Component mode", ["Non-pure (preset mode)", "Pure (manual components)"])
    generation_mode = PURE_MODE if generation_label.startswith("Pure") else "preset"
    signal_type = "mixed"
    n_components = 3
    selected_component_types: list[str] = []

    if generation_mode == PURE_MODE:
        cols = st.columns(len(SUPPORTED_COMPONENT_TYPES))
        for idx, component_type in enumerate(SUPPORTED_COMPONENT_TYPES):
            if cols[idx].checkbox(component_type, value=component_type in ["harmonic", "chirp"], key=f"lab_component_{component_type}"):
                selected_component_types.append(component_type)
        n_components = len(selected_component_types)
    else:
        signal_type = st.selectbox("Signal mode", ["harmonic", "amfm", "chirp", "mixed"])
        n_components = st.slider("Number of components", min_value=1, max_value=6, value=3, step=1)

    fs = st.number_input("Sampling rate", min_value=64, max_value=4096, value=256, step=64)
    duration = st.slider("Duration (s)", min_value=0.5, max_value=8.0, value=4.0, step=0.1)
    noise_level = st.slider("Noise level", min_value=0.0, max_value=0.2, value=0.02, step=0.005)
    seed = st.number_input("Seed", min_value=0, max_value=999999, value=123, step=1)

    if generation_mode == PURE_MODE:
        valid, msg = validate_pure_component_types(selected_component_types)
        if not valid:
            st.warning(msg)
            return

    config = SignalConfig(
        signal_type=signal_type,
        n_components=int(n_components),
        duration=float(duration),
        fs=int(fs),
        noise_level=float(noise_level),
        seed=int(seed),
        generation_mode=generation_mode,
        selected_component_types=selected_component_types,
    )
    generated = generate_signal(config)
    t = generated.t
    stacked = generated.components
    mixture_noisy = generated.mixture
    component_names = generated.component_names
    st.caption(
        f"Generation mode: {'Pure' if generation_mode == PURE_MODE else 'Non-pure preset'} | "
        f"Components: {', '.join(component_names)} | Count: {len(component_names)}"
    )

    st.subheader("Time-Domain Signals")
    st.line_chart(pd.DataFrame({"mixture": mixture_noisy}))
    st.line_chart(pd.DataFrame(stacked.T, columns=component_names))

    freq, mag = fft_magnitude(mixture_noisy, int(fs))
    st.subheader("FFT Magnitude")
    st.line_chart(pd.DataFrame({"frequency_hz": freq, "magnitude": mag}).set_index("frequency_hz"))

    if st.checkbox("Show spectrogram", value=False):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.specgram(mixture_noisy, Fs=int(fs))
        ax.set_xlabel("Time")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

    metadata = {
        "generation_mode": generation_mode,
        "signal_type": signal_type,
        "n_components": len(component_names),
        "selected_component_types": selected_component_types if generation_mode == PURE_MODE else [],
        "component_names": component_names,
        "duration": float(duration),
        "fs": int(fs),
        "noise_level": float(noise_level),
        "seed": int(seed),
    }
    data_payload = {"time": t.tolist(), "components": stacked.tolist(), "mixture": mixture_noisy.tolist(), "metadata": metadata}
    st.session_state["last_generated_signal"] = data_payload
    st.download_button("Download generated sample JSON", data=json.dumps(data_payload), file_name="generated_signal_sample.json", mime="application/json")
    st.download_button("Download metadata JSON", data=json.dumps(metadata, indent=2), file_name="generated_signal_metadata.json", mime="application/json")
