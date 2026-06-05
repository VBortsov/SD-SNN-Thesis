from __future__ import annotations

from datetime import datetime
import io

import numpy as np
import pandas as pd
import streamlit as st

from app.components.charts import fft_comparison_figure, reconstruction_comparison_figure, reconstruction_overview_figure
from app.components.metrics_cards import render_primary_metrics
from app.services.export_service import export_dataframe, export_figure
from app.services.inference_service import evaluate_prediction, load_model, run_inference
from app.services.run_store import append_sample_analysis
from app.services.signal_service import (
    PURE_MODE,
    SUPPORTED_COMPONENT_TYPES,
    SignalConfig,
    fft_magnitude,
    generate_signal,
    validate_pure_component_types,
)
from app.services.paths import REPO_ROOT


def _available_checkpoints(selected_model: dict) -> list[str]:
    default_ckpt = str(selected_model.get("default_checkpoint", "")).strip()
    if not default_ckpt:
        return [""]
    path = (REPO_ROOT / default_ckpt).parent
    if not path.exists():
        return [default_ckpt]
    options = sorted([str(p.relative_to(REPO_ROOT)) for p in path.glob("*.pt")])
    return [default_ckpt] + [opt for opt in options if opt != default_ckpt]


def _run_reconstruction_for_model(
    selected_model: dict,
    checkpoint: str,
    component_count: int,
    mixture: np.ndarray,
    y_true: np.ndarray,
    use_permutation_alignment: bool,
) -> dict:
    model_key = selected_model.get("model_key", selected_model["key"])
    model, load_msg = load_model(model_key, out_channels=component_count, checkpoint_path=checkpoint)
    y_pred, elapsed_ms = run_inference(model, mixture)
    if y_pred.ndim != 2:
        raise ValueError(f"Unexpected prediction shape: {y_pred.shape}. Expected 2D (components, length).")
    if y_pred.shape != y_true.shape:
        raise ValueError(
            "Shape mismatch between generated truth and model output. "
            f"Generated components={y_true.shape[0]}, predicted components={y_pred.shape[0]}, "
            f"truth shape={y_true.shape}, prediction shape={y_pred.shape}."
        )

    report = evaluate_prediction(y_true, y_pred, mixture, permutation_invariant=use_permutation_alignment)
    y_plot = report["aligned_prediction"] if use_permutation_alignment else y_pred
    return {
        "model": selected_model,
        "checkpoint": checkpoint,
        "load_msg": load_msg,
        "prediction": y_pred,
        "plot_prediction": y_plot,
        "elapsed_ms": elapsed_ms,
        "report": report,
    }


def _metrics_row(result: dict) -> dict:
    selected_model = result["model"]
    report = result["report"]
    macro = report.get("macro_average", {})
    observed = report.get("observed_mixture_metrics", {})
    return {
        "model": selected_model.get("key", selected_model.get("model_key", "")),
        "display_name": selected_model.get("display_name", selected_model.get("key", "")),
        "family": selected_model.get("family", ""),
        "depth_label": selected_model.get("depth_label", ""),
        "macro_corr": float(macro.get("corr", float("nan"))),
        "macro_snr_db": float(macro.get("snr_db", float("nan"))),
        "observed_mse": float(observed.get("mse", float("nan"))),
        "latency_ms": float(result["elapsed_ms"]),
        "checkpoint": result["checkpoint"],
        "status": "ok",
    }


def _component_metrics_frame(report: dict, component_names: list[str], model_name: str | None = None) -> pd.DataFrame:
    comp_rows = []
    for idx, metrics in enumerate(report.get("component_metrics", [])):
        label = component_names[idx] if idx < len(component_names) else f"C{idx + 1}"
        row = {"component": label}
        if model_name is not None:
            row["model"] = model_name
        row.update(metrics)
        comp_rows.append(row)
    return pd.DataFrame(comp_rows)


def render(registry: list[dict]) -> None:
    """Render the Streamlit view.
    
    Args:
        registry: Model registry entries.
    """
    st.title("Reconstruction Inspector")
    enabled_models = [m for m in registry if m.get("enabled", True)]
    if not enabled_models:
        st.error("No enabled models in registry.")
        return

    model_map = {m["display_name"]: m for m in enabled_models}
    with st.sidebar:
        st.subheader("Inspector Controls")
        selection_mode = st.radio("Model selection", ["Single model", "Compare models"])
        selected_model_runs: list[dict] = []
        if selection_mode == "Single model":
            model_display = st.selectbox("Model", list(model_map.keys()))
            selected_model = model_map[model_display]
            checkpoints = _available_checkpoints(selected_model)
            checkpoint = st.selectbox("Checkpoint", checkpoints)
            selected_model_runs.append({"display_name": model_display, "model": selected_model, "checkpoint": checkpoint})
        else:
            default_models = list(model_map.keys())[: min(3, len(model_map))]
            selected_displays = st.multiselect("Models", list(model_map.keys()), default=default_models)
            configure_checkpoints = st.checkbox("Choose checkpoints per model", value=False)
            for model_display in selected_displays:
                selected_model = model_map[model_display]
                checkpoints = _available_checkpoints(selected_model)
                checkpoint = checkpoints[0] if checkpoints else ""
                if configure_checkpoints:
                    checkpoint = st.selectbox(f"{model_display} checkpoint", checkpoints, key=f"recon_compare_ckpt_{selected_model['key']}")
                selected_model_runs.append({"display_name": model_display, "model": selected_model, "checkpoint": checkpoint})
        generation_label = st.radio("Component mode", ["Non-pure (preset mode)", "Pure (manual components)"])
        generation_mode = PURE_MODE if generation_label.startswith("Pure") else "preset"
        signal_type = "mixed"
        n_components = 3
        selected_component_types: list[str] = []
        if generation_mode == PURE_MODE:
            st.caption("Select the exact component families to include.")
            for component_type in SUPPORTED_COMPONENT_TYPES:
                if st.checkbox(component_type, value=component_type in ["harmonic", "chirp"], key=f"recon_component_{component_type}"):
                    selected_component_types.append(component_type)
            n_components = len(selected_component_types)
        else:
            signal_type = st.selectbox("Generator preset", ["mixed", "harmonic", "amfm", "chirp"])
            n_components = st.slider("Number of components", min_value=1, max_value=5, value=3, step=1)
        fs = st.number_input("Sampling rate", min_value=64, max_value=4096, value=256, step=64)
        duration = st.slider("Duration (s)", min_value=0.5, max_value=8.0, value=4.0, step=0.1)
        noise_level = st.slider("Noise level (std)", min_value=0.0, max_value=0.2, value=0.03, step=0.005)
        seed = st.number_input("Seed", min_value=0, max_value=999999, value=42, step=1)
        use_permutation_alignment = st.checkbox("Permutation-invariant alignment", value=True)
        show_fft = st.checkbox("Show FFT comparison", value=True)
        show_spectrogram = st.checkbox("Show spectrogram", value=False)
        run = st.button("Run inference", type="primary", use_container_width=True)

    if not run:
        st.info("Configure controls and run inference.")
        return

    if generation_mode == PURE_MODE:
        valid, msg = validate_pure_component_types(selected_component_types)
        if not valid:
            st.warning(msg)
            return

    config = SignalConfig(
        signal_type=signal_type,
        n_components=n_components,
        duration=float(duration),
        fs=int(fs),
        noise_level=float(noise_level),
        seed=int(seed),
        generation_mode=generation_mode,
        selected_component_types=selected_component_types,
    )
    try:
        generated = generate_signal(config)
    except ValueError as exc:
        st.warning(str(exc))
        return

    t = generated.t
    y_true = generated.components
    mixture = generated.mixture
    component_names = generated.component_names
    component_count = len(component_names)
    st.caption(
        f"Generation mode: {'Pure' if generation_mode == PURE_MODE else 'Non-pure preset'} | "
        f"Components: {', '.join(component_names)} | Count: {component_count}"
    )

    if not selected_model_runs:
        st.warning("Select at least one model.")
        return

    results = []
    failed_rows = []
    with st.spinner(f"Running inference for {len(selected_model_runs)} model(s)..."):
        for selected in selected_model_runs:
            selected_model = selected["model"]
            display_name = selected["display_name"]
            checkpoint = selected["checkpoint"]
            try:
                result = _run_reconstruction_for_model(
                    selected_model,
                    checkpoint,
                    component_count,
                    mixture,
                    y_true,
                    use_permutation_alignment,
                )
                results.append(result)
            except Exception as exc:
                failed_rows.append(
                    {
                        "model": selected_model.get("key", selected_model.get("model_key", "")),
                        "display_name": display_name,
                        "family": selected_model.get("family", ""),
                        "depth_label": selected_model.get("depth_label", ""),
                        "macro_corr": float("nan"),
                        "macro_snr_db": float("nan"),
                        "observed_mse": float("nan"),
                        "latency_ms": float("nan"),
                        "checkpoint": checkpoint,
                        "status": f"failed: {exc}",
                    }
                )

    if not results:
        st.error("Inference failed for all selected models.")
        st.dataframe(pd.DataFrame(failed_rows), use_container_width=True, hide_index=True)
        return

    is_comparison = selection_mode == "Compare models"
    metric_rows = [_metrics_row(result) for result in results] + failed_rows
    metrics_df = pd.DataFrame(metric_rows)
    ordered_metrics_df = metrics_df.sort_values("macro_corr", ascending=False, na_position="last")

    best_result = max(
        results,
        key=lambda item: float(item["report"].get("macro_average", {}).get("corr", float("-inf"))),
    )
    report = best_result["report"]
    y_plot = best_result["plot_prediction"]
    selected_model = best_result["model"]
    checkpoint = best_result["checkpoint"]

    macro = report.get("macro_average", {})
    observed = report.get("observed_mixture_metrics", {})
    render_primary_metrics(
        macro_corr=float(macro.get("corr", float("nan"))),
        macro_snr=float(macro.get("snr_db", float("nan"))),
        observed_mse=float(observed.get("mse", float("nan"))),
    )
    if is_comparison:
        st.caption(f"Best model by macro correlation: {selected_model.get('display_name', selected_model.get('key', ''))}")
        st.dataframe(ordered_metrics_df, use_container_width=True, hide_index=True)
    else:
        st.caption(best_result["load_msg"])
        st.caption(f"Inference latency: {best_result['elapsed_ms']:.2f} ms")

    zoom = st.slider("Zoom time range (s)", min_value=0.0, max_value=float(duration), value=(0.0, float(duration)), step=0.05)
    if is_comparison:
        fig = reconstruction_comparison_figure(
            t,
            mixture,
            y_true,
            [(result["model"].get("display_name", result["model"].get("key", "")), result["plot_prediction"]) for result in results],
            t_min=zoom[0],
            t_max=zoom[1],
        )
        st.pyplot(fig, clear_figure=False)

        tabs = st.tabs([result["model"].get("display_name", result["model"].get("key", "")) for result in results])
        for tab, result in zip(tabs, results):
            with tab:
                st.caption(result["load_msg"])
                st.caption(f"Inference latency: {result['elapsed_ms']:.2f} ms")
                detail_fig = reconstruction_overview_figure(
                    t,
                    mixture,
                    y_true,
                    result["plot_prediction"],
                    component_names=component_names,
                    t_min=zoom[0],
                    t_max=zoom[1],
                )
                st.pyplot(detail_fig, clear_figure=False)
                model_name = result["model"].get("display_name", result["model"].get("key", ""))
                comp_df = _component_metrics_frame(result["report"], component_names, model_name=model_name)
                if not comp_df.empty:
                    st.dataframe(comp_df, use_container_width=True, hide_index=True)
    else:
        fig = reconstruction_overview_figure(t, mixture, y_true, y_plot, component_names=component_names, t_min=zoom[0], t_max=zoom[1])
        st.pyplot(fig, clear_figure=False)

    comp_df = _component_metrics_frame(report, component_names)
    if not is_comparison and not comp_df.empty:
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

    if show_fft:
        freq, true_mag = fft_magnitude(np.sum(y_true, axis=0), int(fs))
        _, pred_mag = fft_magnitude(np.sum(y_plot, axis=0), int(fs))
        fft_fig = fft_comparison_figure(freq, true_mag, pred_mag)
        st.pyplot(fft_fig, clear_figure=False)

    if show_spectrogram:
        st.caption("Spectrogram (observed mixture)")
        spec_fig = None
        try:
            import matplotlib.pyplot as plt

            spec_fig, ax = plt.subplots(figsize=(10, 3.5))
            ax.specgram(mixture, Fs=int(fs))
            ax.set_xlabel("Time")
            ax.set_ylabel("Frequency")
            st.pyplot(spec_fig)
        finally:
            if spec_fig is not None:
                spec_fig.clf()

    if is_comparison:
        csv_bytes = ordered_metrics_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download model comparison CSV", data=csv_bytes, file_name="reconstruction_model_comparison.csv", mime="text/csv")
    else:
        csv_bytes = comp_df.to_csv(index=False).encode("utf-8") if not comp_df.empty else b""
        st.download_button("Download component metrics CSV", data=csv_bytes, file_name="component_metrics.csv", mime="text/csv")
    png_buffer = io.BytesIO()
    fig.savefig(png_buffer, format="png", dpi=160, bbox_inches="tight")
    png_buffer.seek(0)
    png_name = "reconstruction_comparison.png" if is_comparison else "reconstruction.png"
    st.download_button("Download reconstruction PNG", data=png_buffer.getvalue(), file_name=png_name, mime="image/png")

    if st.button("Export current reconstruction plot to app/exports"):
        path = export_figure(fig, "reconstruction_plot")
        st.success(f"Saved: `{path.relative_to(REPO_ROOT)}`")

    metrics_to_export = ordered_metrics_df if is_comparison else comp_df
    if not metrics_to_export.empty and st.button("Export metrics table to app/exports"):
        export_name = "reconstruction_model_comparison" if is_comparison else "reconstruction_metrics"
        path = export_dataframe(metrics_to_export, export_name)
        st.success(f"Saved: `{path.relative_to(REPO_ROOT)}`")

    save_label = "Save best model sample for Error Analysis" if is_comparison else "Save sample for Error Analysis"
    if st.button(save_label):
        append_sample_analysis(
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "model": selected_model["key"],
                "depth_label": selected_model.get("depth_label", ""),
                "generation_mode": generation_mode,
                "signal_type": signal_type,
                "n_components": component_count,
                "selected_component_types": selected_component_types if generation_mode == PURE_MODE else [],
                "component_names": component_names,
                "noise_level": noise_level,
                "macro_corr": float(macro.get("corr", float("nan"))),
                "macro_snr_db": float(macro.get("snr_db", float("nan"))),
                "observed_mse": float(observed.get("mse", float("nan"))),
                "checkpoint": checkpoint,
            }
        )
        st.success("Saved sample artifact for error analysis.")

    st.session_state["last_reconstruction"] = {
        "model": selected_model["key"],
        "checkpoint": checkpoint,
        "metrics": report,
        "comparison": ordered_metrics_df.to_dict("records") if is_comparison else [],
        "signal": {
            "fs": int(fs),
            "duration": float(duration),
            "generation_mode": generation_mode,
            "signal_type": signal_type,
            "n_components": component_count,
            "selected_component_types": selected_component_types if generation_mode == PURE_MODE else [],
            "component_names": component_names,
            "noise_level": float(noise_level),
            "seed": int(seed),
        },
        "artifacts": {
            "comparison_mode": is_comparison,
            "time": t.tolist(),
            "mixture": mixture.tolist(),
            "components_true": y_true.tolist(),
            "component_names": component_names,
            "zoom": [float(zoom[0]), float(zoom[1])],
            "results": [
                {
                    "display_name": result["model"].get("display_name", result["model"].get("key", "")),
                    "model": result["model"].get("key", result["model"].get("model_key", "")),
                    "checkpoint": result["checkpoint"],
                    "elapsed_ms": float(result["elapsed_ms"]),
                    "plot_prediction": result["plot_prediction"].tolist(),
                    "component_metrics": result["report"].get("component_metrics", []),
                }
                for result in results
            ],
        },
    }
