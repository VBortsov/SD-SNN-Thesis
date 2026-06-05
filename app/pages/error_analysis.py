from __future__ import annotations

import pandas as pd
import streamlit as st

from app.services.run_store import load_runs, load_sample_analysis


def render(reports_df: pd.DataFrame) -> None:
    """Render the Streamlit view.
    
    Args:
        reports_df: Evaluation reports table.
    """
    st.title("Error Analysis")
    runs = load_runs()
    samples = load_sample_analysis()

    if reports_df.empty:
        st.warning("No report-level metrics available.")
    else:
        st.subheader("Best/Worst Models by Macro Corr")
        best = reports_df.sort_values("macro_corr", ascending=False).head(5)
        worst = reports_df.sort_values("macro_corr", ascending=True).head(5)
        c1, c2 = st.columns(2)
        c1.dataframe(best[["display_name", "depth_label", "macro_corr", "macro_snr_db", "test_loss"]], hide_index=True, use_container_width=True)
        c2.dataframe(worst[["display_name", "depth_label", "macro_corr", "macro_snr_db", "test_loss"]], hide_index=True, use_container_width=True)

        st.subheader("Shallow vs Deep Across Conditions")
        grouped = reports_df.groupby("depth_label")[["macro_corr", "macro_snr_db", "test_loss"]].mean().reset_index()
        st.dataframe(grouped, hide_index=True, use_container_width=True)

    if runs:
        runs_df = pd.DataFrame(runs)
        if "data_settings" in runs_df.columns:
            expanded = pd.json_normalize(runs_df["data_settings"])
            run_metrics = pd.concat([runs_df[["run_id", "model", "depth_label", "status"]], expanded], axis=1)
            st.subheader("Performance by Run Conditions")
            cols = [
                c
                for c in ["run_id", "model", "depth_label", "generation_mode", "signal_type", "selected_component_types", "noise_level", "n_components", "duration", "fs"]
                if c in run_metrics.columns
            ]
            st.dataframe(run_metrics[cols], hide_index=True, use_container_width=True)
    else:
        st.info("No run history yet.")

    st.subheader("Per-Sample Saved Analysis")
    if not samples:
        st.info("No per-sample artifacts saved yet. Use Reconstruction Inspector -> Save sample for Error Analysis.")
    else:
        sdf = pd.DataFrame(samples)
        st.dataframe(sdf.sort_values("macro_corr", ascending=False), hide_index=True, use_container_width=True)
        best_case = sdf.sort_values("macro_corr", ascending=False).iloc[0].to_dict()
        worst_case = sdf.sort_values("macro_corr", ascending=True).iloc[0].to_dict()
        st.write("Best-case sample:", best_case)
        st.write("Worst-case sample:", worst_case)
