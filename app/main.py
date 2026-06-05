from __future__ import annotations

import streamlit as st

from app.pages import (
    comparison,
    cost_data,
    dashboard,
    error_analysis,
    export_page,
    generator_lab,
    hard_signal_mining,
    real_data,
    research,
    reconstruction,
    registry,
    runs,
    training,
)
from app.services.paths import ensure_app_dirs
from app.services.registry_service import ensure_registry_file, load_registry
from app.services.report_loader import load_reports_dataframe


PAGES = [
    "Dashboard",
    "Reconstruction Inspector",
    "Real Data Reconstruction",
    "Test Signal",
    "Model Comparison",
    "Research Comparison",
    "Training Cost and Data Availability",
    "Model Registry",
    "Training",
    "Runs History",
    "Signal Generator",
    "Error Analysis",
    "Export Assets",
]


def main() -> None:
    """Run the command-line entry point."""
    st.set_page_config(page_title="Thesis NN Decomposer Workbench", layout="wide")
    ensure_app_dirs()
    ensure_registry_file()

    registry_data = load_registry()
    load_result = load_reports_dataframe(registry_data)
    reports_df = load_result.dataframe

    st.sidebar.title("NN Decomposer Workbench")
    selected_page = st.sidebar.radio("Page", PAGES, index=PAGES.index(st.session_state.get("selected_page", "Dashboard")))
    st.session_state["selected_page"] = selected_page

    if selected_page == "Dashboard":
        dashboard.render(reports_df, load_result.warnings)
    elif selected_page == "Reconstruction Inspector":
        reconstruction.render(registry_data)
    elif selected_page == "Real Data Reconstruction":
        real_data.render(registry_data)
    elif selected_page == "Test Signal":
        hard_signal_mining.render(registry_data)
    elif selected_page == "Model Comparison":
        comparison.render(reports_df, registry_data)
    elif selected_page == "Research Comparison":
        research.render(reports_df, registry_data, load_result.warnings)
    elif selected_page == "Training Cost and Data Availability":
        cost_data.render(reports_df, registry_data, load_result.warnings)
    elif selected_page == "Model Registry":
        registry.render(registry_data)
    elif selected_page == "Training":
        training.render(registry_data)
    elif selected_page == "Runs History":
        runs.render()
    elif selected_page == "Signal Generator":
        generator_lab.render()
    elif selected_page == "Error Analysis":
        error_analysis.render(reports_df)
    elif selected_page == "Export Assets":
        export_page.render(reports_df, registry_data, load_result.warnings)


if __name__ == "__main__":
    main()
