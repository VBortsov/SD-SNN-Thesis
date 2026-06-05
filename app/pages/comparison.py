from __future__ import annotations

from textwrap import fill

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from app.services.comparison_service import (
    add_paired_statistics,
    comparison_table_path,
    evaluate_registry_on_components,
    run_model_comparison,
)
from app.services.export_service import export_dataframe
from app.services.paths import REPO_ROOT
from app.services.research_service import prepare_chart_generation_dataframe
from app.services.signal_service import SUPPORTED_COMPONENT_TYPES, validate_pure_component_types


COMPARISON_COLUMNS = [
    "model_name",
    "display_name",
    "family",
    "depth_label",
    "test_loss",
    "macro_corr",
    "macro_snr_db",
    "clean_sum_mse",
    "observed_mixture_mse",
    "parameter_count",
    "inference_time_ms",
    "training_date",
    "training_time_sec",
    "train_samples",
    "val_samples",
    "test_samples",
    "total_samples",
    "epochs",
    "checkpoint_path",
    "report_path",
]

NUMERIC_COMPARISON_COLUMNS = [
    "test_loss",
    "macro_corr",
    "macro_snr_db",
    "clean_sum_mse",
    "observed_mixture_mse",
    "parameter_count",
    "inference_time_ms",
    "training_time_sec",
    "train_samples",
    "val_samples",
    "test_samples",
    "total_samples",
    "epochs",
]


def _chart_size(paper_mode: bool, base_width: float = 10.0, base_height: float = 4.0) -> tuple[float, float]:
    width_scale = 0.5 if paper_mode else 1.0
    return base_width * width_scale, base_height


def _metric_label(column: str) -> str:
    return {
        "macro_corr": "Macro correlation",
        "macro_snr_db": "Macro SNR (dB)",
        "training_time_sec": "Training time (seconds)",
    }.get(column, column.replace("_", " ").title())


def _model_axis_label(label: str, *, paper_mode: bool) -> str:
    text = str(label).strip().replace("_", " ")
    max_len = 34 if paper_mode else 48
    wrap_width = 16 if paper_mode else 22
    if len(text) > max_len:
        text = text[: max_len - 3].rstrip() + "..."
    return fill(text, width=wrap_width)


def _bar_figure(df: pd.DataFrame, x_col: str, y_col: str, title: str, *, paper_mode: bool):
    plot_df = df[[x_col, y_col]].dropna()
    height = max(4.0, 0.48 * len(plot_df) + 1.4)
    fig, ax = plt.subplots(figsize=_chart_size(paper_mode, base_width=10.0, base_height=height))
    labels = [_model_axis_label(label, paper_mode=paper_mode) for label in plot_df[x_col].astype(str)]
    positions = np.arange(len(plot_df))
    values = pd.to_numeric(plot_df[y_col], errors="coerce").to_numpy(dtype=float)
    ax.barh(positions, values)
    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=7 if paper_mode else 8)
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel(_metric_label(y_col))
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    return fig


def _multi_bar_figure(df: pd.DataFrame, x_col: str, y_cols: list[str], title: str, *, paper_mode: bool):
    plot_df = df[[x_col] + y_cols].dropna(how="all", subset=y_cols)
    fig, ax = plt.subplots(figsize=_chart_size(paper_mode, base_width=10.0, base_height=4.2))
    x = np.arange(len(plot_df))
    width = 0.8 / max(1, len(y_cols))
    for idx, col in enumerate(y_cols):
        values = pd.to_numeric(plot_df[col], errors="coerce").to_numpy(dtype=float)
        ax.bar(x + (idx - (len(y_cols) - 1) / 2) * width, values, width=width, label=col)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df[x_col].astype(str), rotation=45, ha="right", fontsize=8 if paper_mode else 9)
    ax.set_title(title)
    ax.legend(fontsize=7 if paper_mode else 8)
    fig.tight_layout()
    return fig


def _scatter_figure(df: pd.DataFrame, x_col: str, y_col: str, group_col: str | None, title: str, *, paper_mode: bool):
    fig, ax = plt.subplots(figsize=_chart_size(paper_mode, base_width=10.0, base_height=4.0))
    plot_df = df.dropna(subset=[x_col, y_col]).copy()
    if group_col and group_col in plot_df.columns:
        for group_name, group_df in plot_df.groupby(group_col):
            ax.scatter(group_df[x_col], group_df[y_col], label=str(group_name), alpha=0.8)
        ax.legend(fontsize=7 if paper_mode else 8)
    else:
        ax.scatter(plot_df[x_col], plot_df[y_col], alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel(x_col.replace("_", " ").title())
    ax.set_ylabel(y_col.replace("_", " ").title())
    fig.tight_layout()
    return fig


def _prepare_comparison_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    if "model_name" not in prepared.columns:
        raise ValueError("CSV must include a 'model_name' column.")
    if "display_name" not in prepared.columns:
        prepared["display_name"] = prepared["model_name"]
    prepared["display_name"] = prepared["display_name"].fillna(prepared["model_name"])
    for col, default in {
        "family": "unknown",
        "depth_label": "unknown",
        "checkpoint_path": "",
        "report_path": "",
    }.items():
        if col not in prepared.columns:
            prepared[col] = default
        prepared[col] = prepared[col].fillna(default)
    for col in NUMERIC_COMPARISON_COLUMNS:
        if col not in prepared.columns:
            prepared[col] = pd.NA
        prepared[col] = pd.to_numeric(prepared[col], errors="coerce")
    for col in COMPARISON_COLUMNS:
        if col not in prepared.columns:
            prepared[col] = pd.NA
    return prepared[COMPARISON_COLUMNS]


def render(reports_df: pd.DataFrame, registry: list[dict] | None = None) -> None:
    """Render the Streamlit view.
    
    Args:
        reports_df: Evaluation reports table.
        registry: Model registry entries.
    """
    st.title("Model Comparison")
    last_run = st.session_state.pop("comparison_run_status", None)
    if last_run:
        if last_run["return_code"] == 0:
            st.success(last_run["message"])
        else:
            st.error(last_run["message"])

    st.subheader("Run Evaluation Comparison")
    run_col, mode_col = st.columns([1, 2])
    with mode_col:
        regenerate_reports = st.checkbox(
            "Run eval scripts before comparing",
            value=True,
            help="When enabled, runs each model eval script and refreshes final_test_report.json files used by this UI.",
        )
    with run_col:
        run_clicked = st.button("Run model comparison", type="primary", use_container_width=True)

    if run_clicked:
        log_placeholder = st.empty()
        with st.spinner("Running model comparison..."):
            try:
                process, cmd = run_model_comparison(skip_run=not regenerate_reports)
                st.code(" ".join(cmd), language="bash")
                logs: list[str] = []
                if process.stdout is not None:
                    for line in process.stdout:
                        logs.append(line.rstrip())
                        log_placeholder.text("\n".join(logs[-80:]))
                return_code = process.wait()
            except Exception as exc:
                st.error(f"Model comparison failed: {exc}")
                return

        table = comparison_table_path()
        if return_code == 0:
            artifact_note = f" Artifact: `{table.relative_to(REPO_ROOT)}`." if table.exists() else ""
            st.session_state["comparison_run_status"] = {
                "return_code": return_code,
                "message": f"Model comparison completed.{artifact_note}",
            }
        else:
            st.session_state["comparison_run_status"] = {
                "return_code": return_code,
                "message": f"Model comparison exited with code {return_code}.",
            }
        st.rerun()

    st.subheader("Compare On Selected Signal Set")
    component_cols = st.columns(len(SUPPORTED_COMPONENT_TYPES))
    comparison_components: list[str] = []
    for idx, component_type in enumerate(SUPPORTED_COMPONENT_TYPES):
        if component_cols[idx].checkbox(
            component_type,
            value=component_type in ["harmonic", "amfm", "chirp"],
            key=f"comparison_component_{component_type}",
        ):
            comparison_components.append(component_type)
    valid_components, component_msg = validate_pure_component_types(comparison_components)
    settings_cols = st.columns(5)
    compare_samples = settings_cols[0].number_input("Samples", min_value=1, max_value=200, value=20, step=1)
    compare_fs = settings_cols[1].number_input("Eval fs", min_value=64, max_value=4096, value=256, step=64)
    compare_duration = settings_cols[2].slider("Eval duration (s)", min_value=0.5, max_value=8.0, value=4.0, step=0.1)
    compare_noise = settings_cols[3].slider("Eval noise", min_value=0.0, max_value=0.2, value=0.03, step=0.005)
    compare_seed = settings_cols[4].number_input("Eval seed", min_value=0, max_value=999999, value=42, step=1)
    permutation_invariant = st.checkbox("Use permutation-invariant alignment for signal-set comparison", value=True)

    if not valid_components:
        st.warning(component_msg)
    if st.button("Run signal-set comparison", disabled=not valid_components or not registry):
        with st.spinner("Evaluating models on selected signal set..."):
            rows = evaluate_registry_on_components(
                registry or [],
                comparison_components,
                fs=int(compare_fs),
                duration=float(compare_duration),
                noise_level=float(compare_noise),
                seed=int(compare_seed),
                num_samples=int(compare_samples),
                permutation_invariant=permutation_invariant,
            )
        st.session_state["signal_set_comparison_df"] = pd.DataFrame(rows)
        st.session_state["signal_set_comparison_components"] = comparison_components

    signal_set_df = st.session_state.get("signal_set_comparison_df")
    if isinstance(signal_set_df, pd.DataFrame) and not signal_set_df.empty:
        st.caption(f"Signal-set comparison components: {', '.join(st.session_state.get('signal_set_comparison_components', []))}")
        stats_cols = st.columns(3)
        stats_metric = stats_cols[0].selectbox(
            "CI / paired test metric",
            ["macro_corr", "macro_snr_db", "observed_mse"],
            index=0,
            key="signal_set_stats_metric",
        )
        stats_ascending = stats_metric == "observed_mse"
        valid_reference_names = (
            signal_set_df.sort_values(stats_metric, ascending=stats_ascending, na_position="last")["display_name"]
            .dropna()
            .astype(str)
            .tolist()
        )
        default_reference = valid_reference_names[0] if valid_reference_names else ""
        reference_model = stats_cols[1].selectbox(
            "Reference model",
            valid_reference_names,
            index=0 if valid_reference_names else None,
            key="signal_set_reference_model",
        )
        confidence_pct = stats_cols[2].selectbox("Confidence level", [90, 95, 99], index=1, key="signal_set_ci_level")

        stats_df = add_paired_statistics(
            signal_set_df,
            metric=stats_metric,
            reference_model=reference_model or default_reference,
            confidence=float(confidence_pct) / 100.0,
        )
        ordered_signal_set = stats_df.sort_values(stats_metric, ascending=stats_ascending, na_position="last")

        display_columns = [
            "display_name",
            "family",
            "depth_label",
            "component_set",
            "n_components",
            "macro_corr",
            "macro_snr_db",
            "observed_mse",
            f"{stats_metric}_ci",
            "paired_mean_diff_vs_reference",
            "paired_p_value",
            "paired_significant_95",
            "status",
            "checkpoint_path",
        ]
        available_display_columns = [col for col in display_columns if col in ordered_signal_set.columns]
        st.caption(
            f"{confidence_pct}% bootstrap confidence intervals and paired permutation-test p-values are computed "
            f"against `{reference_model}` on the same sampled signals."
        )
        st.dataframe(ordered_signal_set[available_display_columns], use_container_width=True, hide_index=True)
        csv = ordered_signal_set.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download signal-set comparison CSV",
            csv,
            file_name="signal_set_model_comparison.csv",
            mime="text/csv",
        )

    st.subheader("All Models Table Source")
    uploaded_csv = st.file_uploader(
        "Choose comparison CSV",
        type=["csv"],
        help="Upload a previously saved comparison CSV to rebuild the all-models table, filters, and charts from that file.",
    )
    active_reports_df = reports_df
    active_source_label = "discovered evaluation reports"
    if uploaded_csv is not None:
        try:
            active_reports_df = _prepare_comparison_dataframe(pd.read_csv(uploaded_csv))
            active_source_label = f"uploaded CSV: {uploaded_csv.name}"
        except Exception as exc:
            st.error(f"Failed to load comparison CSV: {exc}")
            active_reports_df = pd.DataFrame(columns=COMPARISON_COLUMNS)
    elif not reports_df.empty:
        active_reports_df = _prepare_comparison_dataframe(reports_df)

    if not active_reports_df.empty:
        active_reports_df = prepare_chart_generation_dataframe(active_reports_df)

    source_col, export_col = st.columns([3, 1])
    with source_col:
        st.caption(f"Current table source: {active_source_label}")
    with export_col:
        if st.button(
            "Save table CSV",
            use_container_width=True,
            disabled=active_reports_df.empty,
            help="Save the current all-models comparison table to app/exports.",
        ):
            path = export_dataframe(active_reports_df, "model_comparison_table")
            st.success(f"Saved: `{path.relative_to(REPO_ROOT)}`")

    if active_reports_df.empty:
        st.warning("No comparison data available.")
        return

    st.sidebar.subheader("Comparison Filters")
    group_filter = st.sidebar.selectbox("Shallow/Deep", ["all", "shallow", "deep"])
    family_options = sorted([f for f in active_reports_df["family"].dropna().unique().tolist() if str(f).strip()])
    families = st.sidebar.multiselect("Architecture family", family_options, default=family_options)
    checkpoint_only = st.sidebar.checkbox("Available checkpoint only", value=False)
    search = st.sidebar.text_input("Search model")
    primary_metric = st.sidebar.selectbox("Primary metric", ["macro_corr", "macro_snr_db", "test_loss"])
    paper_width_mode = st.sidebar.toggle(
        "Paper width mode (50%)",
        value=bool(st.session_state.get("comparison_paper_width_mode", False)),
        help="Halve chart width and tighten labels for LaTeX/paper screenshots.",
    )
    st.session_state["comparison_paper_width_mode"] = paper_width_mode

    filtered = active_reports_df.copy()
    if group_filter != "all":
        filtered = filtered[filtered["depth_label"] == group_filter]
    if families:
        filtered = filtered[filtered["family"].isin(families)]
    if checkpoint_only:
        filtered = filtered[filtered["checkpoint_path"].fillna("").astype(str).str.len() > 0]
    if search:
        mask = filtered["model_name"].str.contains(search, case=False, na=False) | filtered["display_name"].str.contains(search, case=False, na=False)
        filtered = filtered[mask]

    ascending = primary_metric == "test_loss"
    filtered = filtered.sort_values(primary_metric, ascending=ascending)
    st.dataframe(filtered, use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        st.caption("Macro correlation")
        st.pyplot(_bar_figure(filtered, "display_name", "macro_corr", "Macro correlation", paper_mode=paper_width_mode), clear_figure=True)
    with c2:
        st.caption("Macro SNR (dB)")
        st.pyplot(_bar_figure(filtered, "display_name", "macro_snr_db", "Macro SNR (dB)", paper_mode=paper_width_mode), clear_figure=True)

    timing_df = filtered.dropna(subset=["training_time_sec"])
    if not timing_df.empty:
        c3, c4 = st.columns(2)
        with c3:
            st.caption("Training time (seconds)")
            st.pyplot(
                _bar_figure(timing_df, "display_name", "training_time_sec", "Training time (seconds)", paper_mode=paper_width_mode),
                clear_figure=True,
            )
        with c4:
            dataset_cols = [col for col in ["train_samples", "val_samples", "test_samples"] if col in timing_df.columns]
            if dataset_cols:
                st.caption("Dataset sizes")
                st.pyplot(
                    _multi_bar_figure(timing_df, "display_name", dataset_cols, "Dataset sizes", paper_mode=paper_width_mode),
                    clear_figure=True,
                )

    scatter_df = filtered.dropna(subset=["test_loss", "macro_corr"])
    if not scatter_df.empty:
        st.caption("Test loss vs macro correlation")
        st.pyplot(
            _scatter_figure(
                scatter_df,
                "test_loss",
                "macro_corr",
                "depth_label",
                "Test loss vs macro correlation",
                paper_mode=paper_width_mode,
            ),
            clear_figure=True,
        )

    params_df = filtered.dropna(subset=["parameter_count", "macro_corr"])
    if not params_df.empty:
        st.caption("Parameter count vs macro correlation")
        st.pyplot(
            _scatter_figure(
                params_df,
                "parameter_count",
                "macro_corr",
                "depth_label",
                "Parameter count vs macro correlation",
                paper_mode=paper_width_mode,
            ),
            clear_figure=True,
        )

    grouped = (
        filtered.dropna(subset=["depth_label"])
        .groupby("depth_label")[["macro_corr", "macro_snr_db", "test_loss"]]
        .mean()
        .reset_index()
    )
    st.subheader("Shallow vs Deep Averages")
    st.dataframe(grouped, use_container_width=True, hide_index=True)

    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button("Download filtered comparison CSV", csv, file_name="model_comparison_filtered.csv", mime="text/csv")
    if st.button("Export filtered comparison CSV to app/exports"):
        path = export_dataframe(filtered, "comparison_filtered")
        st.success(f"Saved: `{path.relative_to(REPO_ROOT)}`")
