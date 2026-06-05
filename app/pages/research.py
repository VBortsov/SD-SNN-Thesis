from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import streamlit as st

from app.components.tables import render_dataframe
from app.services.export_service import (
    create_export_bundle_dir,
    export_dataframe,
    export_dataframe_to_dir,
    export_figure_to_dir,
    export_json,
    export_json_to_dir,
)
from app.services.paths import REPO_ROOT
from app.services.research_service import (
    DEFAULT_EXPORT_FORMATS,
    METRIC_COLUMNS,
    RESEARCH_EXPORT_SUBDIR,
    SIGNAL_CONDITION_COLUMNS,
    ResearchDataset,
    apply_ablation_scope,
    available_module_columns,
    build_chart_catalog,
    build_results_tables,
    compute_ablation_removal_impact,
    compute_module_contributions,
    compute_summary_statistics,
    figure_heatmap,
    figure_module_impact,
    figure_module_ranking,
    figure_waterfall,
    module_display_names,
    prepare_chart_generation_dataframe,
    prepare_research_dataset,
)
from app.services.training_service import TrainingRequest, build_command
from decomposers.ML_methods.NN_based.experiment_catalog import get_experiment_spec


def _format_metric(value, precision: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float) and math.isnan(value):
        return "n/a"
    return f"{value:.{precision}f}" if isinstance(value, (float, int)) else str(value)


def _warning_for_missing_coverage(df: pd.DataFrame, module_columns: list[str]) -> list[str]:
    warnings: list[str] = []
    for column, label in [
        ("training_time_s", "training-time charts"),
        ("inference_time_ms", "inference-time charts"),
        ("parameters", "parameter-efficiency charts"),
    ]:
        if column not in df.columns or not df[column].notna().any():
            warnings.append(f"Missing `{column}`; skipping {label}.")
    if not module_columns:
        warnings.append("No module flags were available or inferable; skipping module ablation charts.")
    if not any(column in df.columns and df[column].notna().any() for column in SIGNAL_CONDITION_COLUMNS):
        warnings.append("No signal-condition columns were available; skipping condition and robustness charts.")
    if "model_type" not in df.columns or not df["model_type"].isin(["SNN", "DNN"]).any():
        warnings.append("SNN/DNN labels are incomplete; some cross-group comparisons may be empty.")
    return warnings


def _filtered_dataset(dataset: ResearchDataset) -> ResearchDataset:
    df = dataset.dataframe.copy()
    if df.empty:
        return dataset

    st.sidebar.subheader("Research Filters")
    model_types = sorted([value for value in df["model_type"].dropna().unique().tolist() if str(value).strip()])
    selected_types = st.sidebar.multiselect("Model type", model_types, default=model_types)
    families = sorted([value for value in df["family"].dropna().unique().tolist() if str(value).strip()])
    selected_families = st.sidebar.multiselect("Architecture family", families, default=families)
    search = st.sidebar.text_input("Search model")
    config_search = st.sidebar.text_input("Filter config")
    snn_only_ablation = st.sidebar.checkbox("Ablation focus: SNN only", value=True)

    if selected_types:
        df = df[df["model_type"].isin(selected_types)]
    if selected_families:
        df = df[df["family"].isin(selected_families)]
    if search:
        mask = df["display_name"].str.contains(search, case=False, na=False) | df["model_name"].str.contains(search, case=False, na=False)
        df = df[mask]
    if config_search and "config_name" in df.columns:
        df = df[df["config_name"].astype(str).str.contains(config_search, case=False, na=False)]

    ablation_source_df = dataset.dataframe[dataset.dataframe["model_type"] == "SNN"].copy() if snn_only_ablation else dataset.dataframe.copy()
    ablation_model_options = sorted(ablation_source_df["display_name"].dropna().astype(str).unique().tolist())
    default_ablation_models = ablation_model_options
    selected_ablation_models = st.sidebar.multiselect(
        "Ablation models",
        ablation_model_options,
        default=default_ablation_models,
        help="Choose which models participate in the ablation study comparison.",
    )
    ablation_module_keys = available_module_columns(ablation_source_df)
    ablation_module_labels = module_display_names(ablation_module_keys)
    selected_ablation_modules = st.sidebar.multiselect(
        "Ablation modifications",
        ablation_module_labels,
        default=ablation_module_labels,
        help="Choose which modifications/modules to compare in the ablation analysis.",
    )
    require_selected_module_presence = st.sidebar.checkbox(
        "Only rows using selected modifications",
        value=False,
        help="When enabled, the ablation study keeps only experiment rows where at least one selected modification is present.",
    )
    st.session_state["research_ablation_base_df"] = apply_ablation_scope(
        ablation_source_df,
        selected_models=selected_ablation_models,
        selected_modules=selected_ablation_modules,
        require_selected_module_presence=require_selected_module_presence,
    )
    st.session_state["research_ablation_selected_models"] = selected_ablation_models
    st.session_state["research_ablation_selected_modules"] = selected_ablation_modules

    return ResearchDataset(
        dataframe=df.reset_index(drop=True),
        warnings=dataset.warnings,
        module_columns=dataset.module_columns,
        source_labels=dataset.source_labels,
    )


def _render_chart_exports(chart, default_formats: tuple[str, ...]) -> None:
    formats = st.multiselect(
        f"Formats for {chart.slug}",
        list(DEFAULT_EXPORT_FORMATS),
        default=list(default_formats),
        key=f"formats_{chart.slug}",
        label_visibility="collapsed",
    )
    if st.button(f"Export {chart.title}", key=f"export_{chart.slug}"):
        bundle_dir = create_export_bundle_dir(f"{RESEARCH_EXPORT_SUBDIR}_{chart.slug}")
        saved = []
        for fmt in formats or default_formats:
            path = export_figure_to_dir(chart.figure, bundle_dir / f"{chart.slug}.{fmt}")
            saved.append(str(path.relative_to(REPO_ROOT)))
        st.success("Saved:\n" + "\n".join(saved))


def _render_chart_section(title: str, charts: list) -> None:
    if not charts:
        st.info(f"No {title.lower()} charts were available for the current data.")
        return
    for chart in charts:
        st.subheader(chart.title)
        st.pyplot(chart.figure, clear_figure=False, use_container_width=True)
        _render_chart_exports(chart, DEFAULT_EXPORT_FORMATS)


def _render_single_figure(fig, slug: str, title: str) -> None:
    st.subheader(title)
    st.pyplot(fig, clear_figure=False, use_container_width=True)
    chart_like = type("ChartLike", (), {"figure": fig, "slug": slug, "title": title})
    _render_chart_exports(chart_like, DEFAULT_EXPORT_FORMATS)


def _render_table_with_exports(name: str, df: pd.DataFrame) -> None:
    st.subheader(name.replace("_", " ").title())
    if df.empty:
        st.info("No rows available.")
        return
    render_dataframe(df, height=320)
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        f"Download {name} CSV",
        csv,
        file_name=f"{name}.csv",
        mime="text/csv",
        key=f"download_{name}",
    )
    if st.button(f"Save {name} to app/exports", key=f"save_{name}"):
        path = export_dataframe(df, name)
        st.success(f"Saved: `{path.relative_to(REPO_ROOT)}`")


def _render_inference_optimization_option() -> None:
    st.subheader("Inference Optimization Option")
    st.caption("Use a separate TCN variant with safe latency-oriented defaults so your current TCN checkpoints stay unchanged.")
    if st.button("Show inference-optimized TCN option", key="show_inference_optimized_tcn_option"):
        spec = get_experiment_spec("attention_stem_multi_head_multiscale_tcn_inference_optimized")
        if spec is None:
            st.error("The inference-optimized TCN preset is not registered.")
            return
        req = TrainingRequest(
            model_name=spec.key,
            epochs=20,
            batch_size=32,
            learning_rate=1e-3,
            weight_decay=1e-5,
            seed=42,
            fs=256,
            duration=4.0,
            train_samples=8000,
            val_samples=400,
            test_samples=400,
            noise_level=0.03,
            device="auto",
            output_dir=spec.saved_model_dir,
            component_types=["harmonic", "amfm", "chirp"],
            experiment_name="research_inference_optimized",
        )
        st.info(
            "This preset keeps the same model family but uses built-in same-length padding, "
            "smaller channel widths, fewer TCN blocks, ReLU, BatchNorm, and zero dropout."
        )
        st.code(" ".join(build_command(req)), language="bash")
        st.write(
            "Model key: `attention_stem_multi_head_multiscale_tcn_inference_optimized`\n\n"
            "You can train it from the Training page, or run the command above directly."
        )


def _export_bundle(dataset: ResearchDataset, charts: list, tables: dict[str, pd.DataFrame], summary: dict[str, object]) -> None:
    bundle_dir = create_export_bundle_dir(RESEARCH_EXPORT_SUBDIR)
    saved_paths: list[str] = []
    for chart in charts:
        for fmt in DEFAULT_EXPORT_FORMATS:
            path = export_figure_to_dir(chart.figure, bundle_dir / "figures" / chart.section.lower().replace(" ", "_") / f"{chart.slug}.{fmt}")
            saved_paths.append(str(path.relative_to(REPO_ROOT)))
    for name, table in tables.items():
        if table.empty:
            continue
        path = export_dataframe_to_dir(table, bundle_dir / "tables" / f"{name}.csv")
        saved_paths.append(str(path.relative_to(REPO_ROOT)))
    summary_path = export_json_to_dir(summary, bundle_dir / "summary_statistics.json")
    saved_paths.append(str(summary_path.relative_to(REPO_ROOT)))
    source_path = export_json_to_dir({"sources": dataset.source_labels, "warnings": dataset.warnings}, bundle_dir / "bundle_metadata.json")
    saved_paths.append(str(source_path.relative_to(REPO_ROOT)))
    st.success("Saved research bundle:\n" + "\n".join(saved_paths[:20]) + ("\n..." if len(saved_paths) > 20 else ""))


def render(reports_df: pd.DataFrame, registry: list[dict], base_warnings: list[str] | None = None) -> None:
    """Render the Streamlit view.
    
    Args:
        reports_df: Evaluation reports table.
        registry: Model registry entries.
        base_warnings: Warnings collected before rendering.
    """
    st.title("Research Comparison")
    st.caption("Thesis-facing SNN vs DNN analysis with trade-off, ablation, robustness, and export support.")

    uploads = st.file_uploader(
        "Optional: add result CSV/JSON files",
        type=["csv", "json"],
        accept_multiple_files=True,
        help="Use this to add experiment tables with extra fields such as signal conditions, module flags, or custom ablation metadata.",
    )

    dataset = prepare_research_dataset(reports_df, registry, uploaded_files=uploads)
    all_warnings = list(base_warnings or []) + dataset.warnings
    dataset = _filtered_dataset(dataset)

    if dataset.dataframe.empty:
        st.warning("No research data is available for this page.")
        if all_warnings:
            with st.expander("Warnings", expanded=True):
                for warning in all_warnings:
                    st.warning(warning)
        return

    coverage_warnings = _warning_for_missing_coverage(dataset.dataframe, dataset.module_columns)
    if all_warnings or coverage_warnings:
        with st.expander("Warnings and assumptions", expanded=False):
            for warning in all_warnings + coverage_warnings:
                st.warning(warning)

    st.caption("Data sources: " + ", ".join(dataset.source_labels))

    summary = compute_summary_statistics(dataset.dataframe)
    cards = st.columns(5)
    cards[0].metric("Best SNN corr", _format_metric(summary.get("best_snn_macro_corr")))
    cards[1].metric("Best DNN corr", _format_metric(summary.get("best_dnn_macro_corr")))
    cards[2].metric("Corr gap", _format_metric(summary.get("absolute_performance_gap")))
    cards[3].metric("Param reduction", _format_metric(summary.get("parameter_reduction_factor"), precision=2) + ("x" if summary.get("parameter_reduction_factor") is not None else ""))
    cards[4].metric("Inference speedup", _format_metric(summary.get("inference_speedup_factor"), precision=2) + ("x" if summary.get("inference_speedup_factor") is not None else ""))

    extra_cards = st.columns(5)
    extra_cards[0].metric("Best SNN SNR", _format_metric(summary.get("best_snn_macro_snr"), precision=3))
    extra_cards[1].metric("Best DNN SNR", _format_metric(summary.get("best_dnn_macro_snr"), precision=3))
    extra_cards[2].metric("Relative gap", _format_metric(summary.get("relative_performance_gap"), precision=3))
    extra_cards[3].metric("Training speedup", _format_metric(summary.get("training_speedup_factor"), precision=2) + ("x" if summary.get("training_speedup_factor") is not None else ""))
    extra_cards[4].metric("Best trade-off", str(summary.get("best_tradeoff_model") or "n/a"))

    if st.button("Export summary statistics to app/exports"):
        path = export_json(summary, "research_summary_statistics")
        st.success(f"Saved: `{path.relative_to(REPO_ROOT)}`")

    charts = build_chart_catalog(dataset.dataframe)
    tables = build_results_tables(dataset.dataframe)
    if st.button("Export all research charts, tables, and summary"):
        _export_bundle(dataset, charts, tables, summary)

    tabs = st.tabs(
        [
            "Overview",
            "SNN vs DNN",
            "Trade-offs",
            "Ablations",
            "Signal Conditions",
            "Tables",
            "Summary",
        ]
    )

    chart_groups: dict[str, list] = {}
    for chart in charts:
        chart_groups.setdefault(chart.section, []).append(chart)

    with tabs[0]:
        st.subheader("Overall model comparison charts")
        _render_chart_section("Overall", chart_groups.get("Overall", []))

    with tabs[1]:
        st.subheader("SNN vs DNN comparison charts")
        _render_chart_section("SNN vs DNN", chart_groups.get("SNN vs DNN", []))

    with tabs[2]:
        st.subheader("Accuracy-efficiency trade-off charts")
        _render_chart_section("Trade-offs", chart_groups.get("Trade-offs", []))
        _render_inference_optimization_option()

    with tabs[3]:
        st.subheader("Modular ablation analysis")
        ablation_df = st.session_state.get("research_ablation_base_df", dataset.dataframe.copy())
        if isinstance(ablation_df, pd.DataFrame):
            selected_models = st.session_state.get("research_ablation_selected_models", [])
            selected_modules = st.session_state.get("research_ablation_selected_modules", [])
            if selected_models:
                st.caption("Comparing models: " + ", ".join(selected_models))
            if selected_modules:
                st.caption("Comparing modifications: " + ", ".join(selected_modules))
            corr_modules = compute_module_contributions(ablation_df, "macro_corr")
            snr_modules = compute_module_contributions(ablation_df, "macro_snr")
            removal_df = compute_ablation_removal_impact(ablation_df, "macro_corr")
            if ablation_df.empty:
                st.info("No ablation rows match the selected model/modification scope.")
            if not corr_modules.empty:
                st.caption("Module contribution table for macro correlation")
                render_dataframe(corr_modules)
            if not snr_modules.empty:
                st.caption("Module contribution table for macro SNR")
                render_dataframe(snr_modules)
            if not removal_df.empty:
                st.caption("Estimated impact of removing each module from the strongest available configuration")
                render_dataframe(removal_df)
            ablation_chart_df = prepare_chart_generation_dataframe(ablation_df)
            ablation_figures = [
                (
                    figure_module_impact(ablation_chart_df, "macro_corr", "Macro correlation drop when removing each selected modification"),
                    "ablation_module_removal_corr",
                    "Performance change when each selected modification is removed",
                ),
                (
                    figure_waterfall(ablation_chart_df, "macro_corr"),
                    "ablation_waterfall_corr",
                    "Cumulative performance improvement from selected modifications",
                ),
                (
                    figure_heatmap(ablation_chart_df, "macro_corr"),
                    "ablation_heatmap_corr",
                    "Selected modifications versus resulting performance",
                ),
                (
                    figure_module_ranking(ablation_chart_df, "macro_corr", "Selected modifications ranked by macro correlation gain"),
                    "ablation_rank_corr",
                    "Ranking chart for macro correlation contribution",
                ),
                (
                    figure_module_ranking(ablation_chart_df, "macro_snr", "Selected modifications ranked by macro SNR gain"),
                    "ablation_rank_snr",
                    "Ranking chart for macro SNR contribution",
                ),
            ]
            rendered_any = False
            for fig, slug, title in ablation_figures:
                if fig is not None:
                    rendered_any = True
                    _render_single_figure(fig, slug, title)
            if not rendered_any:
                st.info("No ablation charts were available for the selected model/modification scope.")

    with tabs[4]:
        st.subheader("Signal-condition charts")
        _render_chart_section("Signal Conditions", chart_groups.get("Signal Conditions", []))

    with tabs[5]:
        st.subheader("Research tables")
        for name, table in tables.items():
            _render_table_with_exports(name, table)

    with tabs[6]:
        st.subheader("Thesis summary charts")
        _render_chart_section("Thesis Summary", chart_groups.get("Thesis Summary", []))
        st.caption("Pareto-optimal models")
        pareto_names = summary.get("pareto_optimal_models") or []
        if pareto_names:
            st.write(", ".join(pareto_names))
        else:
            st.info("No Pareto-optimal models could be computed from the current data.")
