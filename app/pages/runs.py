from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from app.services.paths import REPO_ROOT
from app.services.run_store import delete_run, load_runs, toggle_favorite


def _run_to_row(run: dict) -> dict:
    data = run.get("data_settings", {})
    return {
        "run_id": run.get("run_id"),
        "timestamp": run.get("timestamp"),
        "model": run.get("model"),
        "depth_label": run.get("depth_label"),
        "status": run.get("status", ""),
        "favorite": bool(run.get("favorite", False)),
        "best_validation_metric": run.get("best_validation_metric"),
        "checkpoint_path": run.get("checkpoint_path", ""),
        "generation_mode": data.get("generation_mode", "preset"),
        "signal_type": data.get("signal_type", ""),
        "selected_component_types": ", ".join(data.get("selected_component_types", []) or []),
        "noise_level": data.get("noise_level"),
        "n_components": data.get("n_components"),
    }


def render() -> None:
    """Render the Streamlit view."""
    st.title("Runs History")
    runs = load_runs()
    if not runs:
        st.info("No runs recorded yet.")
        return

    df = pd.DataFrame([_run_to_row(r) for r in runs])
    model_options = sorted(df["model"].dropna().unique().tolist())
    depth_options = sorted(df["depth_label"].dropna().unique().tolist())
    model_filter = st.multiselect("Filter by model", model_options, default=model_options)
    depth_filter = st.multiselect("Filter by group", depth_options, default=depth_options)
    favorites_only = st.checkbox("Favorites only", value=False)
    date_from = st.date_input("From date", value=date(2026, 1, 1))

    filtered = df.copy()
    if model_filter:
        filtered = filtered[filtered["model"].isin(model_filter)]
    if depth_filter:
        filtered = filtered[filtered["depth_label"].isin(depth_filter)]
    if favorites_only:
        filtered = filtered[filtered["favorite"]]
    filtered = filtered[pd.to_datetime(filtered["timestamp"], errors="coerce").dt.date >= date_from]

    st.dataframe(filtered, use_container_width=True, hide_index=True)
    selected = st.multiselect("Compare selected run IDs", filtered["run_id"].tolist())
    if selected:
        comp = filtered[filtered["run_id"].isin(selected)]
        st.subheader("Selected Run Comparison")
        st.dataframe(comp, use_container_width=True, hide_index=True)

    st.subheader("Run Actions")
    target_run = st.selectbox("Target run ID", filtered["run_id"].tolist())
    c1, c2, c3 = st.columns(3)
    if c1.button("Toggle favorite"):
        if toggle_favorite(target_run):
            st.success("Updated favorite flag.")
            st.rerun()
    if c2.button("Delete run entry"):
        if delete_run(target_run):
            st.success("Deleted run entry.")
            st.rerun()
    if c3.button("Check checkpoint path"):
        row = filtered[filtered["run_id"] == target_run].iloc[0].to_dict()
        ckpt = str(row.get("checkpoint_path", "")).strip()
        if not ckpt:
            st.warning("No checkpoint path in this run.")
        else:
            full = REPO_ROOT / ckpt
            if full.exists():
                st.success(f"Checkpoint exists: `{ckpt}`")
            else:
                st.warning(f"Checkpoint missing: `{ckpt}`")
