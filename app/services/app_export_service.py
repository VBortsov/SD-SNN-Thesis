from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from app.components.charts import reconstruction_comparison_figure, reconstruction_overview_figure
from app.services.cost_data_service import (
    COST_RESEARCH_EXPORT_SUBDIR,
    build_cost_chart_catalog,
    build_cost_tables,
    compute_cost_summary,
    prepare_cost_research_dataset,
)
from app.services.export_service import (
    create_export_bundle_dir,
    export_dataframe_to_dir,
    export_figure_to_dir,
    export_json_to_dir,
)
from app.services.research_service import (
    RESEARCH_EXPORT_SUBDIR,
    build_chart_catalog,
    build_results_tables,
    compute_summary_statistics,
    prepare_research_dataset,
)
from app.services.run_store import load_runs, load_sample_analysis
from app.services.statistical_hardcase_service import (
    STATISTICAL_HARDCASE_EXPORT_SUBDIR,
    StatisticalRunResult,
)
from app.services.paths import REPO_ROOT


@dataclass
class ManifestEntry:
    """Structured registry or manifest entry."""
    path: str
    page: str
    section: str
    resource_type: str
    description: str
    status: str = "exported"


def _json_safe(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if isinstance(value, pd.Series):
        return value.to_dict()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _relative(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


def _manifest_dict(entries: list[ManifestEntry], skipped: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "resources": [entry.__dict__ for entry in entries],
        "skipped": skipped,
    }


def _write_readme(bundle_dir: Path, entries: list[ManifestEntry], skipped: list[dict[str, str]]) -> Path:
    lines = [
        "# Full App Export Bundle",
        "",
        "This bundle contains the currently exportable resources from the dashboard.",
        "",
        "## Exported Resources",
        "",
        "| Page | Section | Type | Path | Description |",
        "| --- | --- | --- | --- | --- |",
    ]
    for entry in entries:
        lines.append(
            f"| {entry.page} | {entry.section} | {entry.resource_type} | `{entry.path}` | {entry.description} |"
        )
    lines.extend(["", "## Skipped Resources", ""])
    if skipped:
        for item in skipped:
            lines.append(f"- **{item['page']} / {item['section']}**: {item['reason']}")
    else:
        lines.append("- None.")
    readme_path = bundle_dir / "README.md"
    readme_path.write_text("\n".join(lines), encoding="utf-8")
    return readme_path


def _add_df(
    entries: list[ManifestEntry],
    bundle_dir: Path,
    relative_path: str,
    df: pd.DataFrame,
    *,
    page: str,
    section: str,
    description: str,
) -> None:
    path = export_dataframe_to_dir(df, bundle_dir / relative_path)
    entries.append(
        ManifestEntry(
            path=_relative(path),
            page=page,
            section=section,
            resource_type="csv",
            description=description,
        )
    )


def _add_json(
    entries: list[ManifestEntry],
    bundle_dir: Path,
    relative_path: str,
    payload: dict[str, Any],
    *,
    page: str,
    section: str,
    description: str,
) -> None:
    path = export_json_to_dir(_json_safe(payload), bundle_dir / relative_path)
    entries.append(
        ManifestEntry(
            path=_relative(path),
            page=page,
            section=section,
            resource_type="json",
            description=description,
        )
    )


def _add_figure(
    entries: list[ManifestEntry],
    bundle_dir: Path,
    relative_path: str,
    fig,
    *,
    page: str,
    section: str,
    description: str,
) -> None:
    path = export_figure_to_dir(fig, bundle_dir / relative_path)
    entries.append(
        ManifestEntry(
            path=_relative(path),
            page=page,
            section=section,
            resource_type=Path(relative_path).suffix.lstrip(".") or "figure",
            description=description,
        )
    )


def _export_shared_inputs(
    bundle_dir: Path,
    entries: list[ManifestEntry],
    registry: list[dict],
    reports_df: pd.DataFrame,
    load_warnings: list[str],
) -> None:
    _add_json(
        entries,
        bundle_dir,
        "shared/model_registry.json",
        {"models": registry},
        page="Shared",
        section="Inputs",
        description="Registry snapshot containing the model catalog and enabled state used by the app.",
    )
    _add_df(
        entries,
        bundle_dir,
        "shared/discovered_reports.csv",
        reports_df,
        page="Shared",
        section="Inputs",
        description="Raw discovered report table loaded from saved model evaluation reports.",
    )
    _add_json(
        entries,
        bundle_dir,
        "shared/report_loader_warnings.json",
        {"warnings": load_warnings},
        page="Shared",
        section="Inputs",
        description="Warnings raised while discovering and parsing saved report files.",
    )


def _export_comparison_page(
    bundle_dir: Path,
    entries: list[ManifestEntry],
    skipped: list[dict[str, str]],
    reports_df: pd.DataFrame,
    session_state: Any,
) -> None:
    if reports_df.empty:
        skipped.append({"page": "Model Comparison", "section": "All Models", "reason": "No report dataframe is available."})
        return
    _add_df(
        entries,
        bundle_dir,
        "model_comparison/all_models.csv",
        reports_df,
        page="Model Comparison",
        section="All Models",
        description="All-model comparison table currently loaded by the app.",
    )
    summary = (
        reports_df.groupby("depth_label")[["macro_corr", "macro_snr_db", "test_loss"]]
        .mean(numeric_only=True)
        .reset_index()
    )
    _add_df(
        entries,
        bundle_dir,
        "model_comparison/shallow_vs_deep_summary.csv",
        summary,
        page="Model Comparison",
        section="Summary",
        description="Mean shallow-vs-deep comparison summary derived from the loaded report table.",
    )
    signal_set_df = session_state.get("signal_set_comparison_df")
    if isinstance(signal_set_df, pd.DataFrame) and not signal_set_df.empty:
        _add_df(
            entries,
            bundle_dir,
            "model_comparison/signal_set_comparison.csv",
            signal_set_df,
            page="Model Comparison",
            section="Signal Set Comparison",
            description="Per-model results from the current session's selected signal-set comparison.",
        )
    else:
        skipped.append(
            {
                "page": "Model Comparison",
                "section": "Signal Set Comparison",
                "reason": "No session signal-set comparison results are available.",
            }
        )


def _export_research_page(
    bundle_dir: Path,
    entries: list[ManifestEntry],
    skipped: list[dict[str, str]],
    reports_df: pd.DataFrame,
    registry: list[dict],
) -> None:
    dataset = prepare_research_dataset(reports_df, registry)
    if dataset.dataframe.empty:
        skipped.append({"page": "Research Comparison", "section": "Bundle", "reason": "No research dataframe is available."})
        return
    summary = compute_summary_statistics(dataset.dataframe)
    charts = build_chart_catalog(dataset.dataframe)
    tables = build_results_tables(dataset.dataframe)
    base = f"{RESEARCH_EXPORT_SUBDIR}"
    _add_json(
        entries,
        bundle_dir,
        f"{base}/summary_statistics.json",
        summary,
        page="Research Comparison",
        section="Summary",
        description="Thesis research summary statistics derived from the research dataset.",
    )
    _add_json(
        entries,
        bundle_dir,
        f"{base}/bundle_metadata.json",
        {"sources": dataset.source_labels, "warnings": dataset.warnings},
        page="Research Comparison",
        section="Metadata",
        description="Research bundle metadata describing source labels and preparation warnings.",
    )
    for chart in charts:
        for fmt in ("png", "svg", "pdf"):
            _add_figure(
                entries,
                bundle_dir,
                f"{base}/figures/{chart.section.lower().replace(' ', '_')}/{chart.slug}.{fmt}",
                chart.figure,
                page="Research Comparison",
                section=chart.section,
                description=f"Research chart '{chart.title}' exported in {fmt.upper()} format.",
            )
    for name, table in tables.items():
        if not table.empty:
            _add_df(
                entries,
                bundle_dir,
                f"{base}/tables/{name}.csv",
                table,
                page="Research Comparison",
                section="Tables",
                description=f"Research results table '{name}'.",
            )


def _export_cost_page(
    bundle_dir: Path,
    entries: list[ManifestEntry],
    skipped: list[dict[str, str]],
    reports_df: pd.DataFrame,
    registry: list[dict],
) -> None:
    dataset = prepare_cost_research_dataset(reports_df, registry)
    if dataset.aggregated.empty and dataset.raw.empty:
        skipped.append({"page": "Training Cost and Data Availability", "section": "Bundle", "reason": "No cost/data dataframe is available."})
        return
    summary = compute_cost_summary(dataset)
    charts = build_cost_chart_catalog(dataset)
    tables = build_cost_tables(dataset)
    base = f"{COST_RESEARCH_EXPORT_SUBDIR}"
    _add_json(
        entries,
        bundle_dir,
        f"{base}/summary_statistics.json",
        summary,
        page="Training Cost and Data Availability",
        section="Summary",
        description="Training-cost and data-availability summary statistics for the current dataset.",
    )
    _add_json(
        entries,
        bundle_dir,
        f"{base}/bundle_metadata.json",
        {"sources": dataset.source_labels, "warnings": dataset.warnings},
        page="Training Cost and Data Availability",
        section="Metadata",
        description="Cost/data bundle metadata describing source labels and preparation warnings.",
    )
    for chart in charts:
        for fmt in ("png", "svg", "pdf"):
            _add_figure(
                entries,
                bundle_dir,
                f"{base}/figures/{chart.section.lower().replace(' ', '_')}/{chart.slug}.{fmt}",
                chart.figure,
                page="Training Cost and Data Availability",
                section=chart.section,
                description=f"Cost/data chart '{chart.title}' exported in {fmt.upper()} format.",
            )
    for name, table in tables.items():
        if not table.empty:
            _add_df(
                entries,
                bundle_dir,
                f"{base}/tables/{name}.csv",
                table,
                page="Training Cost and Data Availability",
                section="Tables",
                description=f"Training-cost or data-availability table '{name}'.",
            )


def _export_runs_and_error_analysis(
    bundle_dir: Path,
    entries: list[ManifestEntry],
    skipped: list[dict[str, str]],
    reports_df: pd.DataFrame,
) -> None:
    runs = load_runs()
    if runs:
        runs_df = pd.DataFrame(runs)
        _add_df(
            entries,
            bundle_dir,
            "runs_history/runs_history.csv",
            runs_df,
            page="Runs History",
            section="Runs",
            description="Full run-history table collected by the dashboard.",
        )
        _add_json(
            entries,
            bundle_dir,
            "runs_history/runs_history.json",
            {"runs": runs},
            page="Runs History",
            section="Runs",
            description="Full run-history JSON snapshot collected by the dashboard.",
        )
        if "data_settings" in runs_df.columns:
            expanded = pd.json_normalize(runs_df["data_settings"])
            cols = [col for col in ["run_id", "model", "depth_label", "status"] if col in runs_df.columns]
            run_metrics = pd.concat([runs_df[cols], expanded], axis=1)
            _add_df(
                entries,
                bundle_dir,
                "error_analysis/performance_by_run_conditions.csv",
                run_metrics,
                page="Error Analysis",
                section="Run Conditions",
                description="Expanded run-condition table used by the Error Analysis page.",
            )
    else:
        skipped.append({"page": "Runs History", "section": "Runs", "reason": "No recorded runs are available."})

    samples = load_sample_analysis()
    if samples:
        samples_df = pd.DataFrame(samples)
        _add_df(
            entries,
            bundle_dir,
            "error_analysis/per_sample_saved_analysis.csv",
            samples_df,
            page="Error Analysis",
            section="Per-Sample Saved Analysis",
            description="Saved per-sample analysis artifacts from Reconstruction Inspector.",
        )
        best_case = samples_df.sort_values("macro_corr", ascending=False).iloc[0].to_dict()
        worst_case = samples_df.sort_values("macro_corr", ascending=True).iloc[0].to_dict()
        _add_json(
            entries,
            bundle_dir,
            "error_analysis/best_and_worst_sample_cases.json",
            {"best_case": best_case, "worst_case": worst_case},
            page="Error Analysis",
            section="Per-Sample Saved Analysis",
            description="Best-case and worst-case sample summaries selected by macro correlation.",
        )
    else:
        skipped.append({"page": "Error Analysis", "section": "Per-Sample Saved Analysis", "reason": "No saved sample analysis artifacts are available."})

    if not reports_df.empty:
        best = reports_df.sort_values("macro_corr", ascending=False).head(5)
        worst = reports_df.sort_values("macro_corr", ascending=True).head(5)
        grouped = reports_df.groupby("depth_label")[["macro_corr", "macro_snr_db", "test_loss"]].mean(numeric_only=True).reset_index()
        _add_df(
            entries,
            bundle_dir,
            "error_analysis/best_models_by_macro_corr.csv",
            best[["display_name", "depth_label", "macro_corr", "macro_snr_db", "test_loss"]],
            page="Error Analysis",
            section="Best/Worst Models",
            description="Top five models by macro correlation from the loaded report table.",
        )
        _add_df(
            entries,
            bundle_dir,
            "error_analysis/worst_models_by_macro_corr.csv",
            worst[["display_name", "depth_label", "macro_corr", "macro_snr_db", "test_loss"]],
            page="Error Analysis",
            section="Best/Worst Models",
            description="Bottom five models by macro correlation from the loaded report table.",
        )
        _add_df(
            entries,
            bundle_dir,
            "error_analysis/shallow_vs_deep_across_conditions.csv",
            grouped,
            page="Error Analysis",
            section="Shallow vs Deep Across Conditions",
            description="Mean shallow-vs-deep error-analysis summary across the loaded report table.",
        )
    else:
        skipped.append({"page": "Error Analysis", "section": "Report Tables", "reason": "No report dataframe is available."})


def _export_reconstruction_snapshot(
    bundle_dir: Path,
    entries: list[ManifestEntry],
    skipped: list[dict[str, str]],
    session_state: Any,
) -> None:
    snapshot = session_state.get("last_reconstruction")
    if not snapshot:
        skipped.append({"page": "Reconstruction Inspector", "section": "Session Snapshot", "reason": "No reconstruction run is stored in session."})
        return
    _add_json(
        entries,
        bundle_dir,
        "reconstruction_inspector/reconstruction_snapshot.json",
        snapshot,
        page="Reconstruction Inspector",
        section="Snapshot",
        description="Full reconstruction snapshot from the most recent Reconstruction Inspector run.",
    )
    comparison_rows = snapshot.get("comparison") or []
    if comparison_rows:
        _add_df(
            entries,
            bundle_dir,
            "reconstruction_inspector/model_comparison.csv",
            pd.DataFrame(comparison_rows),
            page="Reconstruction Inspector",
            section="Comparison",
            description="Per-model comparison table from the most recent Reconstruction Inspector comparison run.",
        )
    artifacts = snapshot.get("artifacts") or {}
    if not artifacts:
        skipped.append({"page": "Reconstruction Inspector", "section": "Charts", "reason": "No reconstruction artifact arrays are stored in session."})
        return

    t = np.asarray(artifacts.get("time", []), dtype=float)
    mixture = np.asarray(artifacts.get("mixture", []), dtype=float)
    y_true = np.asarray(artifacts.get("components_true", []), dtype=float)
    component_names = list(artifacts.get("component_names", []))
    results = artifacts.get("results", [])
    zoom = artifacts.get("zoom", [0.0, float(t[-1]) if t.size else 0.0])
    t_min, t_max = float(zoom[0]), float(zoom[1])
    if t.size == 0 or mixture.size == 0 or y_true.size == 0 or not results:
        skipped.append({"page": "Reconstruction Inspector", "section": "Charts", "reason": "Stored reconstruction artifact arrays are incomplete."})
        return

    if bool(artifacts.get("comparison_mode")):
        compare_fig = reconstruction_comparison_figure(
            t,
            mixture,
            y_true,
            [(item.get("display_name", item.get("model", "")), np.asarray(item.get("plot_prediction", []), dtype=float)) for item in results],
            t_min=t_min,
            t_max=t_max,
        )
        _add_figure(
            entries,
            bundle_dir,
            "reconstruction_inspector/reconstruction_comparison.png",
            compare_fig,
            page="Reconstruction Inspector",
            section="Charts",
            description="Comparison reconstruction figure from the most recent Reconstruction Inspector multi-model run.",
        )
        plt.close(compare_fig)
    else:
        result = results[0]
        pred = np.asarray(result.get("plot_prediction", []), dtype=float)
        overview_fig = reconstruction_overview_figure(
            t,
            mixture,
            y_true,
            pred,
            component_names=component_names,
            t_min=t_min,
            t_max=t_max,
        )
        _add_figure(
            entries,
            bundle_dir,
            "reconstruction_inspector/reconstruction_overview.png",
            overview_fig,
            page="Reconstruction Inspector",
            section="Charts",
            description="Overview reconstruction figure from the most recent Reconstruction Inspector single-model run.",
        )
        plt.close(overview_fig)
        component_metrics = pd.DataFrame(result.get("component_metrics", []))
        if not component_metrics.empty:
            component_metrics.insert(0, "component", component_names[: len(component_metrics)])
            _add_df(
                entries,
                bundle_dir,
                "reconstruction_inspector/component_metrics.csv",
                component_metrics,
                page="Reconstruction Inspector",
                section="Metrics",
                description="Per-component reconstruction metrics from the most recent single-model reconstruction run.",
            )


def _export_generator_snapshot(
    bundle_dir: Path,
    entries: list[ManifestEntry],
    skipped: list[dict[str, str]],
    session_state: Any,
) -> None:
    payload = session_state.get("last_generated_signal")
    if not payload:
        skipped.append({"page": "Signal Generator", "section": "Session Snapshot", "reason": "No generated signal snapshot is stored in session."})
        return
    _add_json(
        entries,
        bundle_dir,
        "signal_generator/generated_signal_snapshot.json",
        payload,
        page="Signal Generator",
        section="Snapshot",
        description="Most recent generated signal snapshot from Signal Generator Lab.",
    )
    time_values = np.asarray(payload.get("time", []), dtype=float)
    mixture = np.asarray(payload.get("mixture", []), dtype=float)
    components = np.asarray(payload.get("components", []), dtype=float)
    metadata = payload.get("metadata", {})
    component_names = list(metadata.get("component_names", []))
    if time_values.size and mixture.size:
        signal_df = pd.DataFrame({"time": time_values, "mixture": mixture})
        for idx, name in enumerate(component_names):
            if idx < len(components):
                signal_df[name] = components[idx]
        _add_df(
            entries,
            bundle_dir,
            "signal_generator/generated_signal_timeseries.csv",
            signal_df,
            page="Signal Generator",
            section="Time-Domain Signals",
            description="Time-domain generated signal snapshot including mixture and component channels.",
        )
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(time_values, mixture, label="mixture", color="black", linewidth=1.2)
        for idx, name in enumerate(component_names):
            if idx < len(components):
                ax.plot(time_values, components[idx], label=name, alpha=0.8)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.set_title("Generated Signal Snapshot")
        ax.legend(loc="upper right", fontsize=8)
        _add_figure(
            entries,
            bundle_dir,
            "signal_generator/generated_signal_plot.png",
            fig,
            page="Signal Generator",
            section="Time-Domain Signals",
            description="Time-domain plot recreated from the most recent Signal Generator Lab snapshot.",
        )
        plt.close(fig)
        fs = int(metadata.get("fs", 256))
        freq = np.fft.rfftfreq(len(mixture), d=1.0 / max(fs, 1))
        mag = np.abs(np.fft.rfft(mixture))
        fft_df = pd.DataFrame({"frequency_hz": freq, "magnitude": mag})
        _add_df(
            entries,
            bundle_dir,
            "signal_generator/generated_signal_fft.csv",
            fft_df,
            page="Signal Generator",
            section="FFT Magnitude",
            description="FFT magnitude table for the most recent generated mixture signal.",
        )
        fft_fig, fft_ax = plt.subplots(figsize=(10, 4))
        fft_ax.plot(freq, mag, color="#1f77b4")
        fft_ax.set_xlabel("Frequency (Hz)")
        fft_ax.set_ylabel("Magnitude")
        fft_ax.set_title("Generated Signal FFT Magnitude")
        _add_figure(
            entries,
            bundle_dir,
            "signal_generator/generated_signal_fft.png",
            fft_fig,
            page="Signal Generator",
            section="FFT Magnitude",
            description="FFT magnitude plot recreated from the most recent Signal Generator Lab snapshot.",
        )
        plt.close(fft_fig)


def _export_hard_signal_snapshots(
    bundle_dir: Path,
    entries: list[ManifestEntry],
    skipped: list[dict[str, str]],
    session_state: Any,
) -> None:
    payload = session_state.get("test_signal_payload")
    if payload:
        _add_json(
            entries,
            bundle_dir,
            "test_signal/test_signal_payload.json",
            payload,
            page="Test Signal",
            section="Diagnostics",
            description="Signal payload from the most recent Test Signal diagnostic run.",
        )
    else:
        skipped.append({"page": "Test Signal", "section": "Diagnostics", "reason": "No test-signal payload is stored in session."})
    results = session_state.get("test_signal_results")
    if isinstance(results, list) and results:
        _add_json(
            entries,
            bundle_dir,
            "test_signal/test_signal_results.json",
            {"results": results},
            page="Test Signal",
            section="Diagnostics",
            description="Per-model diagnostic results from the most recent Test Signal run.",
        )
    top_cases = session_state.get("hard_signal_cases")
    if isinstance(top_cases, list) and top_cases:
        _add_df(
            entries,
            bundle_dir,
            "test_signal/hard_signal_top_cases.csv",
            pd.DataFrame(top_cases),
            page="Test Signal",
            section="Hard Cases",
            description="Top ranked hard-signal cases from the most recent hard-signal search.",
        )
    search_settings = session_state.get("hard_signal_search_settings")
    if isinstance(search_settings, dict) and search_settings:
        _add_json(
            entries,
            bundle_dir,
            "test_signal/hard_signal_search_settings.json",
            search_settings,
            page="Test Signal",
            section="Hard Cases",
            description="Search settings used for the most recent hard-signal case mining run.",
        )


def _export_statistical_hardcase_snapshot(
    bundle_dir: Path,
    entries: list[ManifestEntry],
    skipped: list[dict[str, str]],
    session_state: Any,
) -> None:
    result: StatisticalRunResult | None = session_state.get("statistical_hardcase_result")
    config = session_state.get("statistical_hardcase_config", {})
    if result is None:
        skipped.append({"page": "Test Signal", "section": "Statistical Hard-Case Testing", "reason": "No statistical hard-case result is stored in session."})
        return
    base = STATISTICAL_HARDCASE_EXPORT_SUBDIR
    table_specs = [
        ("signals.csv", result.signal_table, "Generated signal table for the statistical hard-case testing run."),
        ("model_results.csv", result.model_table, "Per-model results for the statistical hard-case testing run."),
        ("component_results.csv", result.component_table, "Per-component results for the statistical hard-case testing run."),
        ("signal_comparison.csv", result.signal_comparison_table, "Best SNN vs best DNN comparison table per signal."),
        ("hard_cases.csv", result.hard_case_table, "Ranked hard-case table for the statistical hard-case testing run."),
    ]
    for filename, table, description in table_specs:
        _add_df(
            entries,
            bundle_dir,
            f"{base}/tables/{filename}",
            table,
            page="Test Signal",
            section="Statistical Hard-Case Testing",
            description=description,
        )
    _add_json(
        entries,
        bundle_dir,
        f"{base}/summary.json",
        result.summary,
        page="Test Signal",
        section="Statistical Hard-Case Testing",
        description="Summary statistics for the statistical hard-case testing run.",
    )
    _add_json(
        entries,
        bundle_dir,
        f"{base}/run_config.json",
        config if isinstance(config, dict) else {},
        page="Test Signal",
        section="Statistical Hard-Case Testing",
        description="Run configuration used for the statistical hard-case testing run.",
    )
    _add_json(
        entries,
        bundle_dir,
        f"{base}/warnings.json",
        {"warnings": result.warnings},
        page="Test Signal",
        section="Statistical Hard-Case Testing",
        description="Warnings emitted while building the statistical hard-case testing result.",
    )
    for artifact_key, artifact in result.artifacts.items():
        _add_json(
            entries,
            bundle_dir,
            f"{base}/artifacts/{artifact_key}.json",
            artifact,
            page="Test Signal",
            section="Statistical Hard-Case Testing",
            description=f"Detailed artifact payload for statistical hard-case item '{artifact_key}'.",
        )


def export_complete_app_bundle(
    *,
    reports_df: pd.DataFrame,
    registry: list[dict],
    load_warnings: list[str] | None = None,
    session_state: Any,
) -> Path:
    """Export complete app bundle.
    
    Args:
        reports_df: Evaluation reports table.
        registry: Model registry entries.
    """
    bundle_dir = create_export_bundle_dir("full_app_export")
    entries: list[ManifestEntry] = []
    skipped: list[dict[str, str]] = []

    _export_shared_inputs(bundle_dir, entries, registry, reports_df, list(load_warnings or []))
    _export_comparison_page(bundle_dir, entries, skipped, reports_df, session_state)
    _export_research_page(bundle_dir, entries, skipped, reports_df, registry)
    _export_cost_page(bundle_dir, entries, skipped, reports_df, registry)
    _export_runs_and_error_analysis(bundle_dir, entries, skipped, reports_df)
    _export_reconstruction_snapshot(bundle_dir, entries, skipped, session_state)
    _export_generator_snapshot(bundle_dir, entries, skipped, session_state)
    _export_hard_signal_snapshots(bundle_dir, entries, skipped, session_state)
    _export_statistical_hardcase_snapshot(bundle_dir, entries, skipped, session_state)

    manifest_path = export_json_to_dir(
        _json_safe(_manifest_dict(entries, skipped)),
        bundle_dir / "export_manifest.json",
    )
    entries.append(
        ManifestEntry(
            path=_relative(manifest_path),
            page="Export Assets",
            section="Manifest",
            resource_type="json",
            description="Machine-readable manifest describing every exported resource and every skipped page section.",
        )
    )
    readme_path = _write_readme(bundle_dir, entries, skipped)
    entries.append(
        ManifestEntry(
            path=_relative(readme_path),
            page="Export Assets",
            section="Manifest",
            resource_type="md",
            description="Human-readable overview of the full export bundle and resource descriptions.",
        )
    )
    export_json_to_dir(_json_safe(_manifest_dict(entries, skipped)), bundle_dir / "export_manifest.json")
    return bundle_dir
