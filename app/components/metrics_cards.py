from __future__ import annotations

import streamlit as st


def render_primary_metrics(macro_corr: float, macro_snr: float, observed_mse: float) -> None:
    """Render primary metrics.
    
    Args:
        macro_corr: Macro-averaged correlation.
        macro_snr: Macro-averaged SNR in dB.
        observed_mse: MSE against the observed mixture.
    """
    col1, col2, col3 = st.columns(3)
    col1.metric("Macro Corr", f"{macro_corr:.4f}")
    col2.metric("Macro SNR (dB)", f"{macro_snr:.4f}")
    col3.metric("Observed Mixture MSE", f"{observed_mse:.6f}")

