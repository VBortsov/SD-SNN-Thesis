from __future__ import annotations

import pandas as pd
import streamlit as st

from app.services.app_export_service import export_complete_app_bundle
from app.services.export_service import export_dataframe, export_json
from app.services.paths import REPO_ROOT
from app.services.run_store import load_runs


def render(reports_df: pd.DataFrame, registry: list[dict], load_warnings: list[str] | None = None) -> None:
    """Render the Streamlit view.
    
    Args:
        reports_df: Evaluation reports table.
        registry: Model registry entries.
        load_warnings: Warnings from loading data.
    """
    st.title("Export / Thesis Assets")
    st.caption("Generate thesis-ready tables, summaries, selected run exports, or one complete documented app bundle.")

    st.subheader("Full App Bundle")
    st.caption("Exports all currently available charts, CSV tables, JSON summaries, and a manifest with a short description for every exported resource.")
    if st.button("Export complete app bundle"):
        bundle_dir = export_complete_app_bundle(
            reports_df=reports_df,
            registry=registry,
            load_warnings=load_warnings or [],
            session_state=st.session_state,
        )
        st.success(f"Saved: `{bundle_dir.relative_to(REPO_ROOT)}`")

    if not reports_df.empty:
        if st.button("Export full comparison table"):
            path = export_dataframe(reports_df, "thesis_model_comparison")
            st.success(f"Saved: `{path.relative_to(REPO_ROOT)}`")

        summary = reports_df.groupby("depth_label")[["macro_corr", "macro_snr_db", "test_loss"]].mean().reset_index()
        st.dataframe(summary, hide_index=True, use_container_width=True)
        if st.button("Export shallow-vs-deep summary"):
            path = export_dataframe(summary, "thesis_shallow_vs_deep_summary")
            st.success(f"Saved: `{path.relative_to(REPO_ROOT)}`")
            json_path = export_json({"shallow_vs_deep_summary": summary.to_dict(orient="records")}, "thesis_shallow_vs_deep_summary")
            st.success(f"Saved: `{json_path.relative_to(REPO_ROOT)}`")

    last_reconstruction = st.session_state.get("last_reconstruction")
    if last_reconstruction:
        st.subheader("Last Reconstruction Metrics")
        st.json(last_reconstruction["metrics"].get("macro_average", {}))
        if st.button("Export last reconstruction metric summary"):
            path = export_json(last_reconstruction, "thesis_reconstruction_summary")
            st.success(f"Saved: `{path.relative_to(REPO_ROOT)}`")
    else:
        st.info("No reconstruction run stored in session yet.")

    runs = load_runs()
    if runs:
        run_df = pd.DataFrame(runs)
        ids = run_df["run_id"].tolist()
        selected = st.multiselect("Select run IDs to export", ids)
        if selected:
            chosen = run_df[run_df["run_id"].isin(selected)]
            st.dataframe(chosen, hide_index=True, use_container_width=True)
            if st.button("Export selected run summaries"):
                path = export_dataframe(chosen, "thesis_selected_runs")
                st.success(f"Saved: `{path.relative_to(REPO_ROOT)}`")
    else:
        st.info("No runs available for export.")
