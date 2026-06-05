from __future__ import annotations

import pandas as pd
import streamlit as st

from app.services.registry_service import save_registry, validate_registry


REGISTRY_COLUMNS = [
    "key",
    "model_key",
    "display_name",
    "family",
    "depth_label",
    "num_layers",
    "notes",
    "default_checkpoint",
    "enabled",
]


def render(registry: list[dict]) -> None:
    """Render the Streamlit view.
    
    Args:
        registry: Model registry entries.
    """
    st.title("Model Registry")
    st.caption("Define model metadata and shallow/deep grouping used across the dashboard.")

    base_df = pd.DataFrame(registry)
    for col in REGISTRY_COLUMNS:
        if col not in base_df.columns:
            base_df[col] = ""
    base_df = base_df[REGISTRY_COLUMNS]

    edited = st.data_editor(
        base_df,
        use_container_width=True,
        num_rows="dynamic",
        hide_index=True,
        key="model_registry_editor",
    )

    if st.button("Save registry", type="primary"):
        records = edited.fillna("").to_dict(orient="records")
        for row in records:
            if row.get("num_layers", "") in ("", None):
                row["num_layers"] = None
        errors = validate_registry(records)
        if errors:
            for err in errors:
                st.error(err)
            return
        save_registry(records)
        st.success("Model registry saved.")

