from __future__ import annotations

from pathlib import Path
import json

import pandas as pd
import streamlit as st

from app.components.tables import render_dataframe
from app.services.paths import REPO_ROOT, SAVED_MODELS_DIR
from app.services.run_store import load_runs


def _latest_checkpoint() -> str:
    checkpoints = sorted(SAVED_MODELS_DIR.rglob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not checkpoints:
        return "No checkpoints found."
    latest = checkpoints[0]
    return str(latest.relative_to(REPO_ROOT))


def render(reports_df: pd.DataFrame, report_warnings: list[str]) -> None:
    """Render the Streamlit view.
    
    Args:
        reports_df: Evaluation reports table.
        report_warnings: Warnings from report loading.
    """
    st.title("Dashboard")
    if report_warnings:
        with st.expander("Report warnings", expanded=False):
            for warning in report_warnings:
                st.warning(warning)

    runs = load_runs()
    latest_run = runs[-1] if runs else None

    if reports_df.empty:
        st.warning("No evaluation reports found. Run model evaluation scripts first.")
    else:
        by_corr = reports_df.sort_values("macro_corr", ascending=False).iloc[0]
        by_snr = reports_df.sort_values("macro_snr_db", ascending=False).iloc[0]
        col1, col2, col3 = st.columns(3)
        col1.metric("Top Macro Corr", f"{by_corr['display_name']} ({by_corr['macro_corr']:.4f})")
        col2.metric("Top Macro SNR", f"{by_snr['display_name']} ({by_snr['macro_snr_db']:.3f} dB)")
        col3.metric("Models With Reports", str(len(reports_df)))

        group_df = (
            reports_df.dropna(subset=["depth_label"])
            .groupby("depth_label")[["macro_corr", "macro_snr_db", "test_loss"]]
            .mean()
            .reset_index()
        )
        st.subheader("Shallow vs Deep Aggregate")
        if group_df.empty:
            st.info("Depth labels are missing in the model registry.")
        else:
            st.dataframe(group_df, use_container_width=True, hide_index=True)

        c1, c2 = st.columns(2)
        with c1:
            st.caption("Top models by macro correlation")
            top_corr = reports_df.sort_values("macro_corr", ascending=False).head(8)
            st.bar_chart(top_corr.set_index("display_name")[["macro_corr"]])
        with c2:
            st.caption("Top models by macro SNR")
            top_snr = reports_df.sort_values("macro_snr_db", ascending=False).head(8)
            st.bar_chart(top_snr.set_index("display_name")[["macro_snr_db"]])

        family_counts = reports_df["family"].fillna("unknown").value_counts().rename_axis("family").reset_index(name="count")
        st.caption("Model counts by family")
        st.bar_chart(family_counts.set_index("family")[["count"]])
        with st.expander("Report table"):
            render_dataframe(reports_df)

    st.subheader("Latest Experiment Run")
    if latest_run is None:
        st.info("No training runs tracked yet.")
    else:
        st.json(latest_run)
        ckpt_path = str(latest_run.get("checkpoint_path", "")).strip()
        if ckpt_path:
            history_path = (REPO_ROOT / ckpt_path).parent / "training_history.json"
            if history_path.exists():
                try:
                    payload = json.loads(history_path.read_text(encoding="utf-8"))
                    history = payload.get("history", [])
                    hist_df = pd.DataFrame(history)
                    if {"epoch", "train_loss", "val_loss"}.issubset(hist_df.columns):
                        st.caption("Recent run train/val loss trend")
                        st.line_chart(hist_df.set_index("epoch")[["train_loss", "val_loss"]])
                except Exception:
                    st.warning("Could not parse training history for latest run.")

    st.subheader("Latest Checkpoint Summary")
    latest_ckpt = _latest_checkpoint()
    if latest_ckpt.startswith("No checkpoints"):
        st.warning(latest_ckpt)
    else:
        st.success(f"Latest checkpoint: `{latest_ckpt}`")

    if reports_df.empty:
        st.warning("Missing reports: Model comparison and dashboard metrics are limited.")
