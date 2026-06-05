from __future__ import annotations

import pandas as pd
import streamlit as st


def render_dataframe(df: pd.DataFrame, height: int = 360) -> None:
    """Render dataframe.
    
    Args:
        df: Input dataframe.
        height: Rendered table height.
    """
    st.dataframe(df, use_container_width=True, hide_index=True, height=height)

