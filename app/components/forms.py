from __future__ import annotations

import streamlit as st


def bool_filter(label: str, default: bool = False) -> bool:
    """Bool filter.
    
    Args:
        label: Model or axis label.
        default: Default control value.
    """
    return st.checkbox(label, value=default)

