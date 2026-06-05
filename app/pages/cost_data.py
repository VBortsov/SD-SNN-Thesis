from __future__ import annotations

import math
import re
import time

import pandas as pd
import streamlit as st

from app.components.tables import render_dataframe
from app.services.cost_data_service import (
    COST_METRIC_LABELS,
    COST_RESEARCH_EXPORT_SUBDIR,
    CostResearchDataset,
    build_cost_chart_catalog,
    build_cost_tables,
    compute_cost_summary,
    filter_cost_dataset,
    prepare_cost_research_dataset,
    retest_data_availability_models,
)
from app.services.export_service import (
    create_export_bundle_dir,
    export_dataframe,
    export_dataframe_to_dir,
    export_figure_to_dir,
    export_json,
    export_json_to_dir,
)
from app.services.paths import REPO_ROOT
from app.services.run_store import append_run, create_run_record, update_run
from app.services.signal_service import SUPPORTED_COMPONENT_TYPES, validate_pure_component_types
from app.services.training_service import (
    TrainingRequest,
    drain_training_output,
    elapsed_seconds,
    expected_checkpoint_path,
    parse_batch_line,
    parse_progress_line,
    read_final_report,
    start_training_job,
    training_script_path,
)


def _format_metric(value, precision: int = 4, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float) and math.isnan(value):
        return "n/a"
    if isinstance(value, (float, int)):
        return f"{value:.{precision}f}{suffix}"
    return str(value)


def _slugify(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", value.strip()).strip("_").lower()


def _parse_seed_values(raw: str) -> list[int]:
    values: list[int] = []
    for item in raw.split(","):
        text = item.strip()
        if not text:
            continue
        values.append(int(text))
    return values or [42]


def _parse_fraction_values(selected_labels: list[str], custom_raw: str) -> list[float]:
    values: list[float] = []
    for label in selected_labels:
        values.append(float(label.rstrip("%")) / 100.0)
    for item in custom_raw.split(","):
        text = item.strip()
        if not text:
            continue
        values.append(float(text))
    cleaned = sorted({round(value, 6) for value in values if 0.0 < value <= 1.0})
    return cleaned


def _parse_sample_values(raw: str) -> list[int]:
    values: list[int] = []
    for item in raw.split(","):
        text = item.strip()
        if not text:
            continue
        values.append(max(1, int(text)))
    return sorted(set(values))


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
        key=f"download_cost_{name}",
    )
    if st.button(f"Save {name} to app/exports", key=f"save_cost_{name}"):
        path = export_dataframe(df, name)
        st.success(f"Saved: `{path.relative_to(REPO_ROOT)}`")


def _render_chart_exports(chart) -> None:
    formats = st.multiselect(
        f"Formats for {chart.slug}",
        ["png", "svg", "pdf"],
        default=["png", "svg", "pdf"],
        key=f"cost_formats_{chart.slug}",
        label_visibility="collapsed",
    )
    if st.button(f"Export {chart.title}", key=f"cost_export_{chart.slug}"):
        bundle_dir = create_export_bundle_dir(f"{COST_RESEARCH_EXPORT_SUBDIR}_{chart.slug}")
        saved = []
        for fmt in formats:
            path = export_figure_to_dir(chart.figure, bundle_dir / f"{chart.slug}.{fmt}")
            saved.append(str(path.relative_to(REPO_ROOT)))
        st.success("Saved:\n" + "\n".join(saved))


def _export_bundle(dataset: CostResearchDataset, charts: list, tables: dict[str, pd.DataFrame], summary: dict[str, object]) -> None:
    bundle_dir = create_export_bundle_dir(COST_RESEARCH_EXPORT_SUBDIR)
    saved: list[str] = []
    for chart in charts:
        for fmt in ["png", "svg", "pdf"]:
            path = export_figure_to_dir(chart.figure, bundle_dir / "figures" / chart.section.lower().replace(" ", "_") / f"{chart.slug}.{fmt}")
            saved.append(str(path.relative_to(REPO_ROOT)))
    for name, table in tables.items():
        if table.empty:
            continue
        path = export_dataframe_to_dir(table, bundle_dir / "tables" / f"{name}.csv")
        saved.append(str(path.relative_to(REPO_ROOT)))
    saved.append(str(export_json_to_dir(summary, bundle_dir / "summary_statistics.json").relative_to(REPO_ROOT)))
    saved.append(str(export_json_to_dir({"warnings": dataset.warnings, "sources": dataset.source_labels}, bundle_dir / "bundle_metadata.json").relative_to(REPO_ROOT)))
    st.success("Saved cost/data-efficiency bundle:\n" + "\n".join(saved[:20]) + ("\n..." if len(saved) > 20 else ""))


def _select_models(dataset: CostResearchDataset, registry: list[dict]) -> CostResearchDataset:
    raw = dataset.raw
    if raw.empty:
        return dataset

    st.sidebar.subheader("Cost/Data Filters")
    registry_map = {str(item.get("key", "")): str(item.get("display_name", item.get("key", ""))) for item in registry}
    available_models = sorted(raw["display_name"].dropna().astype(str).unique().tolist())
    available_model_names = sorted(raw["model_name"].dropna().astype(str).unique().tolist())
    preset = st.sidebar.selectbox(
        "Model set preset",
        ["All available", "Registry enabled", "SNN only", "DNN only", "Custom"],
        index=0,
    )
    if preset == "Registry enabled":
        enabled_keys = {str(item.get("key", "")) for item in registry if item.get("enabled", True)}
        default_models = [
            display_name
            for display_name, model_name in zip(raw["display_name"], raw["model_name"])
            if str(model_name) in enabled_keys
        ]
        default_models = sorted(set(default_models))
    elif preset == "SNN only":
        default_models = sorted(raw.loc[raw["model_type"] == "SNN", "display_name"].dropna().astype(str).unique().tolist())
    elif preset == "DNN only":
        default_models = sorted(raw.loc[raw["model_type"] == "DNN", "display_name"].dropna().astype(str).unique().tolist())
    else:
        default_models = available_models

    selected_models = st.sidebar.multiselect(
        "Choose models for this test",
        available_models,
        default=default_models,
        help="Only the selected models will be included in the training-cost and data-availability analysis.",
    )
    selected_types = st.sidebar.multiselect(
        "Model type",
        sorted(raw["model_type"].dropna().astype(str).unique().tolist()),
        default=sorted(raw["model_type"].dropna().astype(str).unique().tolist()),
    )
    selected_families = st.sidebar.multiselect(
        "Architecture family",
        sorted(raw["family"].dropna().astype(str).unique().tolist()),
        default=sorted(raw["family"].dropna().astype(str).unique().tolist()),
    )
    selected_sources = st.sidebar.multiselect(
        "Result source",
        sorted(raw["report_source"].dropna().astype(str).unique().tolist()) if "report_source" in raw.columns else [],
        default=sorted(raw["report_source"].dropna().astype(str).unique().tolist()) if "report_source" in raw.columns else [],
    )
    data_regime = st.sidebar.selectbox("Data regime", ["all", "limited", "full"], index=0)
    st.session_state["cost_data_selected_models"] = selected_models

    return filter_cost_dataset(
        dataset,
        selected_models=selected_models,
        selected_types=selected_types,
        selected_families=selected_families,
        selected_sources=selected_sources,
        data_regime=data_regime,
    )


def _warnings_for_missing_columns(dataset: CostResearchDataset) -> list[str]:
    df = dataset.raw
    warnings: list[str] = []
    for column, label in [
        ("peak_memory_mb", "memory charts"),
        ("train_fraction", "data-availability charts"),
        ("training_time_s", "training-cost charts"),
        ("parameters", "parameter-efficiency charts"),
        ("inference_time_ms", "inference-efficiency charts"),
    ]:
        if column not in df.columns or not df[column].notna().any():
            warnings.append(f"Missing `{column}`; skipping {label}.")
    if "model_type" not in df.columns or not df["model_type"].isin(["SNN", "DNN"]).any():
        warnings.append("SNN/DNN labels are incomplete; group comparisons may be limited.")
    return warnings


def _render_data_availability_training(registry: list[dict]) -> None:
    status = st.session_state.pop("cost_data_sweep_status", None)
    if status:
        st.success(status)

    trainable = [model for model in registry if model.get("enabled", True) and training_script_path(model["key"]) is not None]
    if not trainable:
        st.info("No trainable models are available for data-availability sweeps.")
        return

    display_map = {str(model["display_name"]): model for model in trainable}
    selected_analysis_models = st.session_state.get("cost_data_selected_models", [])
    default_models = [name for name in selected_analysis_models if name in display_map] or list(display_map.keys())

    with st.expander("Run Data Availability Sweep", expanded=False):
        st.caption("Launch repeated training runs across train fractions or absolute train-sample counts. This is sequential and writes separate saved-model directories per run.")
        selected_models = st.multiselect("Sweep models", list(display_map.keys()), default=default_models)
        sweep_mode = st.radio("Sweep mode", ["Fractions of reference train size", "Absolute train sample counts"], horizontal=True)

        if sweep_mode == "Fractions of reference train size":
            fraction_labels = st.multiselect(
                "Train fractions",
                ["5%", "10%", "25%", "50%", "75%", "100%"],
                default=["5%", "10%", "25%", "50%", "75%", "100%"],
            )
            custom_fractions = st.text_input("Custom fractions", value="", help="Optional comma-separated values like 0.15,0.33")
            reference_train_samples = int(st.number_input("Reference full-data train size", min_value=32, max_value=100000, value=4000, step=32))
            fraction_values = _parse_fraction_values(fraction_labels, custom_fractions)
            train_sample_values = [max(32, int(round(reference_train_samples * fraction))) for fraction in fraction_values]
        else:
            absolute_samples = st.text_input("Train sample counts", value="200,400,1000,2000,4000")
            train_sample_values = _parse_sample_values(absolute_samples)
            fraction_values = []
            reference_train_samples = max(train_sample_values) if train_sample_values else 0

        col1, col2, col3 = st.columns(3)
        val_samples = int(col1.number_input("Validation samples", min_value=32, max_value=50000, value=400, step=32, key="cost_val_samples"))
        test_samples = int(col2.number_input("Test samples", min_value=32, max_value=50000, value=400, step=32, key="cost_test_samples"))
        seeds_raw = col3.text_input("Seeds", value="42", help="Comma-separated seeds", key="cost_seeds")
        seeds = _parse_seed_values(seeds_raw)

        col4, col5, col6 = st.columns(3)
        epochs = int(col4.number_input("Epochs", min_value=1, max_value=500, value=20, step=1, key="cost_epochs"))
        batch_size = int(col5.number_input("Batch size", min_value=1, max_value=1024, value=32, step=1, key="cost_batch_size"))
        learning_rate = float(col6.number_input("Learning rate", min_value=1e-6, max_value=1.0, value=1e-3, format="%.6f", key="cost_learning_rate"))

        col7, col8, col9 = st.columns(3)
        weight_decay = float(col7.number_input("Weight decay", min_value=0.0, max_value=1.0, value=1e-5, format="%.6f", key="cost_weight_decay"))
        fs = int(col8.number_input("Sampling rate (Hz)", min_value=64, max_value=4096, value=256, step=64, key="cost_fs"))
        duration = float(col9.slider("Duration (s)", min_value=0.5, max_value=8.0, value=4.0, step=0.1, key="cost_duration"))

        col10, col11, col12 = st.columns(3)
        noise_level = float(col10.slider("Noise level", min_value=0.0, max_value=0.2, value=0.03, step=0.005, key="cost_noise"))
        device = col11.selectbox("Device", ["auto", "cpu"], key="cost_device")
        sweep_suffix = _slugify(col12.text_input("Sweep suffix", value="data_availability", key="cost_sweep_suffix"))

        component_cols = st.columns(len(SUPPORTED_COMPONENT_TYPES))
        component_types: list[str] = []
        for idx, component_type in enumerate(SUPPORTED_COMPONENT_TYPES):
            if component_cols[idx].checkbox(
                component_type,
                value=component_type in ["harmonic", "amfm", "chirp"],
                key=f"cost_component_{component_type}",
            ):
                component_types.append(component_type)
        valid_components, component_message = validate_pure_component_types(component_types)
        if not valid_components:
            st.warning(component_message)

        planned_runs = len(selected_models) * len(train_sample_values) * len(seeds)
        if sweep_mode == "Fractions of reference train size" and fraction_values:
            st.caption("Sweep fractions: " + ", ".join(f"{value:.2f}" for value in fraction_values))
        st.caption(f"Planned runs: {planned_runs}")

        if st.button("Start data-availability sweep", type="primary", disabled=(not selected_models or not train_sample_values or not valid_components)):
            status_placeholder = st.empty()
            table_placeholder = st.empty()
            log_placeholder = st.empty()
            command_placeholder = st.empty()
            summary_rows: list[dict] = []
            run_index = 0
            total_runs = max(planned_runs, 1)

            for model_display in selected_models:
                target_model = display_map[model_display]
                for sample_idx, train_samples in enumerate(train_sample_values):
                    fraction_value = None
                    if sweep_mode == "Fractions of reference train size" and sample_idx < len(fraction_values):
                        fraction_value = fraction_values[sample_idx]
                    for seed in seeds:
                        run_index += 1
                        fraction_tag = f"tf{int(round((fraction_value or (train_samples / max(reference_train_samples, 1))) * 100)):03d}"
                        seed_tag = f"s{seed}"
                        variant_suffix = "_".join(part for part in [sweep_suffix, fraction_tag, seed_tag] if part)
                        output_dir = f"decomposers/ML_methods/NN_based/saved_models/{target_model['key']}_{variant_suffix}"
                        req = TrainingRequest(
                            model_name=target_model["key"],
                            epochs=epochs,
                            batch_size=batch_size,
                            learning_rate=learning_rate,
                            weight_decay=weight_decay,
                            seed=seed,
                            fs=fs,
                            duration=duration,
                            train_samples=int(train_samples),
                            val_samples=val_samples,
                            test_samples=test_samples,
                            noise_level=noise_level,
                            device=device,
                            output_dir=output_dir,
                            component_types=component_types,
                            experiment_name=variant_suffix,
                        )
                        run_record = create_run_record(
                            model_name=target_model["key"],
                            depth_label=target_model.get("depth_label", ""),
                            data_settings={
                                "selected_component_types": component_types,
                                "noise_level": noise_level,
                                "duration": duration,
                                "fs": fs,
                                "train_samples": int(train_samples),
                                "val_samples": int(val_samples),
                                "test_samples": int(test_samples),
                                "train_fraction": fraction_value,
                            },
                            hyperparams={
                                "epochs": epochs,
                                "batch_size": batch_size,
                                "learning_rate": learning_rate,
                                "weight_decay": weight_decay,
                                "seed": seed,
                                "run_name_suffix": variant_suffix,
                            },
                            notes="Data-availability sweep run started from cost/data dashboard.",
                        )
                        append_run(run_record)
                        try:
                            job = start_training_job(target_model, req)
                        except Exception as exc:
                            update_run(run_record["run_id"], {"status": "failed", "notes": f"Training failed to start: {exc}"})
                            summary_rows.append(
                                {
                                    "model": target_model["display_name"],
                                    "train_samples": train_samples,
                                    "train_fraction": fraction_value,
                                    "seed": seed,
                                    "status": f"failed to start: {exc}",
                                }
                            )
                            table_placeholder.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
                            continue

                        logs: list[str] = []
                        latest_progress: dict[str, float] = {}
                        latest_batch: dict[str, float] = {}
                        command_placeholder.code(" ".join(job.command), language="bash")
                        while True:
                            for line in drain_training_output(job):
                                logs.append(line.rstrip())
                                parsed = parse_progress_line(line)
                                if parsed:
                                    latest_progress = parsed
                                parsed_batch = parse_batch_line(line)
                                if parsed_batch:
                                    latest_batch = parsed_batch
                            return_code = job.process.poll()
                            status_placeholder.info(
                                f"Run {run_index}/{total_runs}: {target_model['display_name']} | train_samples={train_samples} | seed={seed}"
                            )
                            summary_preview = summary_rows + [
                                {
                                    "model": target_model["display_name"],
                                    "train_samples": train_samples,
                                    "train_fraction": fraction_value,
                                    "seed": seed,
                                    "status": "running" if return_code is None else ("completed" if return_code == 0 else "failed"),
                                    "epoch": latest_progress.get("epoch"),
                                    "val_loss": latest_progress.get("val_loss"),
                                    "batch_loss": latest_batch.get("loss"),
                                    "elapsed_sec": round(elapsed_seconds(job), 1),
                                }
                            ]
                            table_placeholder.dataframe(pd.DataFrame(summary_preview), use_container_width=True, hide_index=True)
                            log_placeholder.text("\n".join(logs[-40:]))
                            if return_code is not None:
                                break
                            time.sleep(0.75)

                        final_report = read_final_report(output_dir)
                        summary = (final_report or {}).get("test_summary", {})
                        update_run(
                            run_record["run_id"],
                            {
                                "status": "completed" if return_code == 0 else "failed",
                                "test_summary": summary if return_code == 0 else {},
                                "checkpoint_path": expected_checkpoint_path(target_model["key"], output_dir),
                                "training_time_seconds": elapsed_seconds(job),
                                "notes": "Data-availability sweep completed." if return_code == 0 else f"Training exited with code {return_code}",
                            },
                        )
                        summary_rows.append(
                            {
                                "model": target_model["display_name"],
                                "train_samples": train_samples,
                                "train_fraction": fraction_value,
                                "seed": seed,
                                "status": "completed" if return_code == 0 else "failed",
                                "test_loss": (final_report or {}).get("test_loss"),
                                "macro_corr": ((summary or {}).get("macro_average") or {}).get("corr"),
                                "macro_snr": ((summary or {}).get("macro_average") or {}).get("snr_db"),
                            }
                        )

            table_placeholder.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
            st.session_state["cost_data_sweep_status"] = f"Completed {len(summary_rows)} data-availability sweep runs."
            st.rerun()


def render(reports_df: pd.DataFrame, registry: list[dict], base_warnings: list[str] | None = None) -> None:
    """Render the Streamlit view.
    
    Args:
        reports_df: Evaluation reports table.
        registry: Model registry entries.
        base_warnings: Warnings collected before rendering.
    """
    st.title("Training Cost and Data Availability")
    st.caption("Computed training costs and data availability tests: SNNs vs DNNs.")
    _render_data_availability_training(registry)

    uploads = st.file_uploader(
        "Optional: add result CSV/JSON files",
        type=["csv", "json"],
        accept_multiple_files=True,
        help="Add future experiment tables with train-fraction sweeps, seed repeats, memory metrics, or hardware metadata.",
    )

    dataset = prepare_cost_research_dataset(reports_df, registry, uploaded_files=uploads)
    dataset = _select_models(dataset, registry)
    all_warnings = list(base_warnings or []) + dataset.warnings + _warnings_for_missing_columns(dataset)

    if dataset.raw.empty or dataset.aggregated.empty:
        st.warning("No compatible results are available for the selected model set.")
        if all_warnings:
            with st.expander("Warnings", expanded=True):
                for warning in all_warnings:
                    st.warning(warning)
        return

    with st.expander("Warnings and assumptions", expanded=False):
        for warning in all_warnings:
            st.warning(warning)

    st.caption("Data sources: " + ", ".join(dataset.source_labels))
    st.caption(f"Selected models: {dataset.raw['display_name'].nunique()} | Aggregated rows: {len(dataset.aggregated)}")

    summary = compute_cost_summary(dataset)
    row1 = st.columns(6)
    row1[0].metric("Best SNN", str(summary.get("best_snn_model") or "n/a"))
    row1[1].metric("Best DNN", str(summary.get("best_dnn_model") or "n/a"))
    row1[2].metric("Best SNN corr", _format_metric(summary.get("best_snn_macro_corr")))
    row1[3].metric("Best DNN corr", _format_metric(summary.get("best_dnn_macro_corr")))
    row1[4].metric("Abs. gap", _format_metric(summary.get("absolute_performance_gap")))
    row1[5].metric("Rel. gap", _format_metric(summary.get("relative_performance_gap"), precision=3))

    row2 = st.columns(6)
    row2[0].metric("Param reduction", _format_metric(summary.get("snn_parameter_reduction_factor"), precision=2, suffix="x"))
    row2[1].metric("Inference speedup", _format_metric(summary.get("snn_inference_speedup_factor"), precision=2, suffix="x"))
    row2[2].metric("Training speedup", _format_metric(summary.get("snn_training_speedup_factor"), precision=2, suffix="x"))
    row2[3].metric("Best limited-data model", str(summary.get("best_model_under_limited_data") or "n/a"))
    row2[4].metric("Best full-data model", str(summary.get("best_model_under_full_data") or "n/a"))
    row2[5].metric("Best acc/train-s", str(summary.get("best_accuracy_per_training_second_model") or "n/a"))

    row3 = st.columns(5)
    row3[0].metric("Best acc/param", str(summary.get("best_accuracy_per_parameter_model") or "n/a"))
    row3[1].metric("Best acc/sample", str(summary.get("best_accuracy_per_sample_model") or "n/a"))
    row3[2].metric("DNN overtakes at", _format_metric(summary.get("dnn_overtakes_at_fraction"), precision=2))
    row3[3].metric("Closest SNN to DNN", _format_metric(summary.get("closest_fraction"), precision=2))
    row3[4].metric("Best low-data trade-off", _format_metric(summary.get("best_tradeoff_fraction"), precision=2))

    if st.button("Export summary statistics to app/exports", key="cost_summary_export"):
        path = export_json(summary, "cost_data_summary_statistics")
        st.success(f"Saved: `{path.relative_to(REPO_ROOT)}`")

    charts = build_cost_chart_catalog(dataset)
    tables = build_cost_tables(dataset)
    if st.button("Export all cost/data charts, tables, and summary", key="cost_bundle_export"):
        _export_bundle(dataset, charts, tables, summary)

    chart_groups: dict[str, list] = {}
    for chart in charts:
        chart_groups.setdefault(chart.section, []).append(chart)

    tabs = st.tabs(
        [
            "Overview",
            "Training Cost",
            "Data Availability",
            "Trade-offs",
            "SNN vs DNN",
            "Tables",
            "Conclusions",
        ]
    )

    with tabs[0]:
        st.subheader("Overview summary")
        overview_rows = [
            {"statistic": key, "value": value}
            for key, value in summary.items()
            if not isinstance(value, pd.DataFrame)
        ]
        render_dataframe(pd.DataFrame(overview_rows), height=280)

    with tabs[1]:
        st.subheader("Training cost comparison")
        if not chart_groups.get("Training Cost"):
            st.info("No training-cost charts were available for the selected data.")
        for chart in chart_groups.get("Training Cost", []):
            st.subheader(chart.title)
            st.pyplot(chart.figure, clear_figure=False, use_container_width=True)
            _render_chart_exports(chart)

    with tabs[2]:
        st.subheader("Data availability tests")
        if not chart_groups.get("Data Availability"):
            st.info("No data-availability charts were available for the selected data.")
        for chart in chart_groups.get("Data Availability", []):
            st.subheader(chart.title)
            st.pyplot(chart.figure, clear_figure=False, use_container_width=True)
            _render_chart_exports(chart)

        with st.expander("Retest data-availability models with 95% confidence intervals", expanded=False):
            st.caption("Re-evaluate the currently filtered data-availability checkpoints on the same synthetic signal set and report 95% bootstrap confidence intervals.")
            available_retest_df = dataset.raw.copy()
            available_retest_df = available_retest_df[
                available_retest_df["checkpoint_path"].notna()
                & available_retest_df["checkpoint_path"].astype(str).str.len().gt(0)
                & available_retest_df["train_fraction"].notna()
            ].copy()
            if available_retest_df.empty:
                st.info("No data-availability checkpoints are available under the current filters.")
            else:
                retest_component_cols = st.columns(len(SUPPORTED_COMPONENT_TYPES))
                retest_components: list[str] = []
                for idx, component_type in enumerate(SUPPORTED_COMPONENT_TYPES):
                    if retest_component_cols[idx].checkbox(
                        component_type,
                        value=component_type in ["harmonic", "amfm", "chirp"],
                        key=f"cost_retest_component_{component_type}",
                    ):
                        retest_components.append(component_type)
                valid_retest_components, retest_component_msg = validate_pure_component_types(retest_components)
                if not valid_retest_components:
                    st.warning(retest_component_msg)

                retest_cols = st.columns(5)
                retest_samples = retest_cols[0].number_input("Retest samples", min_value=1, max_value=200, value=20, step=1, key="cost_retest_samples")
                retest_fs = retest_cols[1].number_input("Retest fs", min_value=64, max_value=4096, value=256, step=64, key="cost_retest_fs")
                retest_duration = retest_cols[2].slider("Retest duration (s)", min_value=0.5, max_value=8.0, value=4.0, step=0.1, key="cost_retest_duration")
                retest_noise = retest_cols[3].slider("Retest noise", min_value=0.0, max_value=0.2, value=0.03, step=0.005, key="cost_retest_noise")
                retest_seed = retest_cols[4].number_input("Retest seed", min_value=0, max_value=999999, value=42, step=1, key="cost_retest_seed")
                retest_perm = st.checkbox("Use permutation-invariant alignment", value=True, key="cost_retest_perm")

                if st.button(
                    "Run retest with 95% CI",
                    type="primary",
                    disabled=not valid_retest_components,
                    key="cost_retest_run",
                ):
                    with st.spinner("Retesting data-availability checkpoints..."):
                        retest_results = retest_data_availability_models(
                            available_retest_df,
                            retest_components,
                            fs=int(retest_fs),
                            duration=float(retest_duration),
                            noise_level=float(retest_noise),
                            seed=int(retest_seed),
                            num_samples=int(retest_samples),
                            permutation_invariant=bool(retest_perm),
                            confidence=0.95,
                        )
                    st.session_state["cost_data_retest_results"] = retest_results
                    st.session_state["cost_data_retest_components"] = retest_components

                retest_results = st.session_state.get("cost_data_retest_results")
                if isinstance(retest_results, pd.DataFrame) and not retest_results.empty:
                    st.caption(
                        "Retest components: "
                        + ", ".join(st.session_state.get("cost_data_retest_components", []))
                        + " | Confidence interval: 95%"
                    )
                    ordered_retest = retest_results.sort_values(["train_fraction", "macro_corr"], ascending=[True, False], na_position="last")
                    render_dataframe(ordered_retest, height=360)
                    csv = ordered_retest.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Download retest table CSV",
                        csv,
                        file_name="data_availability_retest_ci95.csv",
                        mime="text/csv",
                        key="cost_retest_download",
                    )
                    if st.button("Save retest table to app/exports", key="cost_retest_export"):
                        path = export_dataframe(ordered_retest, "data_availability_retest_ci95")
                        st.success(f"Saved: `{path.relative_to(REPO_ROOT)}`")

    with tabs[3]:
        st.subheader("Compute-performance trade-offs")
        if not chart_groups.get("Trade-offs"):
            st.info("No trade-off charts were available for the selected data.")
        for chart in chart_groups.get("Trade-offs", []):
            st.subheader(chart.title)
            st.pyplot(chart.figure, clear_figure=False, use_container_width=True)
            _render_chart_exports(chart)

    with tabs[4]:
        st.subheader("SNN vs DNN focused comparisons")
        if not chart_groups.get("SNN vs DNN"):
            st.info("No SNN vs DNN charts were available for the selected data.")
        for chart in chart_groups.get("SNN vs DNN", []):
            st.subheader(chart.title)
            st.pyplot(chart.figure, clear_figure=False, use_container_width=True)
            _render_chart_exports(chart)

    with tabs[5]:
        st.subheader("Tables")
        for name, table in tables.items():
            _render_table_with_exports(name, table)

    with tabs[6]:
        st.subheader("Limited-data conclusions")
        conclusion_rows = [
            {"question": "Best model at each train fraction", "available": not tables["best_model_per_train_fraction"].empty},
            {"question": "Best SNN at each train fraction", "available": not tables["best_snn_per_train_fraction"].empty},
            {"question": "Best DNN at each train fraction", "available": not tables["best_dnn_per_train_fraction"].empty},
            {"question": "DNN overtakes at fraction", "value": summary.get("dnn_overtakes_at_fraction")},
            {"question": "SNN closest to DNN at fraction", "value": summary.get("closest_fraction")},
            {"question": "SNN best trade-off fraction", "value": summary.get("best_tradeoff_fraction")},
            {"question": "SNN attractive in low-data regime", "value": summary.get("low_data_snn_advantage")},
        ]
        render_dataframe(pd.DataFrame(conclusion_rows), height=240)
