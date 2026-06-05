from __future__ import annotations

import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from app.services.export_service import export_dataframe, export_figure
from app.services.inference_service import count_parameters, load_model, load_train_config, run_inference
from app.services.paths import REAL_DATA_DIR, REPO_ROOT, SAVED_MODELS_DIR
from app.services.real_data_service import list_real_data_files, load_edf_segment, parse_summary_annotations, read_edf_metadata


def _checkpoint_sort_key(path_str: str) -> tuple[int, int, str]:
    name = Path(path_str).name
    if name.endswith("_best.pt"):
        return (0, 0, name)
    if name.endswith("_last.pt"):
        return (1, 0, name)
    if "_epoch_" in name:
        try:
            epoch = int(name.rsplit("_epoch_", 1)[1].split(".pt", 1)[0])
        except ValueError:
            epoch = 999999
        return (2, epoch, name)
    return (3, 0, name)


def _available_checkpoints(selected_model: dict) -> list[str]:
    model_key = str(selected_model.get("key", "")).strip()
    if not model_key or not SAVED_MODELS_DIR.exists():
        return []

    checkpoint_options: list[str] = []
    for run_dir in sorted(SAVED_MODELS_DIR.iterdir()):
        if not run_dir.is_dir():
            continue
        run_name = run_dir.name
        if run_name != model_key and not run_name.startswith(f"{model_key}_data_availability_"):
            continue
        train_config = load_train_config(str(run_dir / "placeholder.pt"))
        if int(train_config.get("train_samples", -1)) != 8000:
            continue
        for checkpoint in sorted(run_dir.glob("*.pt")):
            if checkpoint.name.endswith("_weights_only.pt"):
                continue
            checkpoint_options.append(str(checkpoint.relative_to(REPO_ROOT)))

    return sorted(set(checkpoint_options), key=_checkpoint_sort_key)


def _common_signal_length(selected_runs: list[dict]) -> int | None:
    lengths = []
    for item in selected_runs:
        train_config = load_train_config(item["checkpoint"])
        signal_length = train_config.get("signal_length")
        if signal_length is not None:
            lengths.append(int(signal_length))
    if not lengths:
        return None
    return lengths[0] if len(set(lengths)) == 1 else min(lengths)


def _preprocess_signal(signal: np.ndarray, mode: str) -> tuple[np.ndarray, dict]:
    x = np.asarray(signal, dtype=np.float32).reshape(-1)
    stats = {
        "raw_mean": float(np.mean(x)),
        "raw_std": float(np.std(x)),
        "raw_min": float(np.min(x)),
        "raw_max": float(np.max(x)),
    }
    if mode == "Raw":
        return x, stats
    centered = x - np.mean(x)
    if mode == "Demean":
        return centered, stats
    if mode == "Z-score":
        scale = float(np.std(centered))
        scale = scale if scale > 1e-8 else 1.0
        return centered / scale, stats
    median = float(np.median(x))
    mad = float(np.median(np.abs(x - median)))
    scale = (1.4826 * mad) if mad > 1e-8 else 1.0
    return (x - median) / scale, stats


def _adapt_length(signal: np.ndarray, target_length: int | None) -> tuple[np.ndarray, str]:
    x = np.asarray(signal, dtype=np.float32).reshape(-1)
    if target_length is None or target_length <= 0:
        return x, "unchanged"
    if x.size == target_length:
        return x, "unchanged"
    if x.size > target_length:
        return x[:target_length], f"cropped to {target_length} samples"
    padded = np.zeros(target_length, dtype=np.float32)
    padded[: x.size] = x
    return padded, f"zero-padded to {target_length} samples"


def _proxy_metrics(observed: np.ndarray, predicted_components: np.ndarray) -> dict:
    reconstruction = np.sum(predicted_components, axis=0)
    residual = observed - reconstruction
    signal_power = float(np.mean(observed**2))
    residual_power = float(np.mean(residual**2))
    corr = float(np.corrcoef(observed, reconstruction)[0, 1]) if np.std(observed) > 1e-8 and np.std(reconstruction) > 1e-8 else float("nan")
    mae = float(np.mean(np.abs(residual)))
    explained_variance = 1.0 - (float(np.var(residual)) / (float(np.var(observed)) + 1e-12))
    return {
        "reconstruction": reconstruction,
        "residual": residual,
        "residual_mse": residual_power,
        "residual_mae": mae,
        "residual_ratio": residual_power / (signal_power + 1e-12),
        "reconstruction_corr": corr,
        "explained_variance": explained_variance,
    }


def _component_frame(prediction: np.ndarray, model_name: str) -> pd.DataFrame:
    rows = []
    for idx, component in enumerate(prediction):
        rows.append(
            {
                "model": model_name,
                "component": f"C{idx + 1}",
                "mean": float(np.mean(component)),
                "std": float(np.std(component)),
                "energy": float(np.mean(component**2)),
                "peak_abs": float(np.max(np.abs(component))),
            }
        )
    return pd.DataFrame(rows)


def _component_spectral_frame(prediction: np.ndarray, model_name: str, fs: float) -> tuple[pd.DataFrame, dict]:
    rows = []
    spectra = []
    for idx, component in enumerate(prediction):
        centered = component - np.mean(component)
        spectrum = np.abs(np.fft.rfft(centered))
        freq = np.fft.rfftfreq(component.size, d=1.0 / fs)
        spectra.append(spectrum)
        dominant_idx = int(np.argmax(spectrum[1:]) + 1) if spectrum.size > 1 else 0
        centroid = float(np.sum(freq * spectrum) / (np.sum(spectrum) + 1e-12))
        rows.append(
            {
                "model": model_name,
                "component": f"C{idx + 1}",
                "dominant_freq_hz": float(freq[dominant_idx]) if dominant_idx < freq.size else float("nan"),
                "spectral_centroid_hz": centroid,
                "band_power": float(np.sum(spectrum**2)),
            }
        )

    pairwise_overlap = []
    pairwise_corr = []
    for i in range(len(spectra)):
        for j in range(i + 1, len(spectra)):
            spec_i = spectra[i]
            spec_j = spectra[j]
            overlap = float(np.sum(np.minimum(spec_i, spec_j)) / (min(np.sum(spec_i), np.sum(spec_j)) + 1e-12))
            corr = float(np.corrcoef(spec_i, spec_j)[0, 1]) if np.std(spec_i) > 1e-8 and np.std(spec_j) > 1e-8 else float("nan")
            pairwise_overlap.append(overlap)
            pairwise_corr.append(corr)

    summary = {
        "mean_pairwise_spectral_overlap": float(np.mean(pairwise_overlap)) if pairwise_overlap else float("nan"),
        "max_pairwise_spectral_overlap": float(np.max(pairwise_overlap)) if pairwise_overlap else float("nan"),
        "mean_pairwise_spectral_corr": float(np.nanmean(pairwise_corr)) if pairwise_corr else float("nan"),
        "spectral_diversity_score": float(1.0 - np.mean(pairwise_overlap)) if pairwise_overlap else float("nan"),
    }
    return pd.DataFrame(rows), summary


def _real_data_figure(time_axis: np.ndarray, observed: np.ndarray, prediction: np.ndarray, model_name: str):
    reconstruction = np.sum(prediction, axis=0)
    residual = observed - reconstruction
    rows = prediction.shape[0] + 2
    fig, axes = plt.subplots(rows, 1, figsize=(12, 2.3 * rows), sharex=True)
    axes[0].plot(time_axis, observed, color="black", linewidth=1.1, label="Observed segment")
    axes[0].set_title("Observed Signal")
    axes[0].legend(loc="upper right")

    axes[1].plot(time_axis, observed, color="black", linewidth=1.0, alpha=0.8, label="Observed")
    axes[1].plot(time_axis, reconstruction, linewidth=1.0, label="Reconstructed sum")
    axes[1].plot(time_axis, residual, linewidth=0.9, alpha=0.75, label="Residual")
    axes[1].set_title(f"Model Reconstruction: {model_name}")
    axes[1].legend(loc="upper right")

    for idx in range(prediction.shape[0]):
        axes[idx + 2].plot(time_axis, prediction[idx], linewidth=1.0, label=f"Predicted component C{idx + 1}")
        axes[idx + 2].legend(loc="upper right")

    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    return fig


def _reconstruction_comparison_figure(results: list[dict]):
    rows = len(results)
    fig, axes = plt.subplots(rows, 1, figsize=(12, 2.8 * rows), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, result in zip(axes, results):
        observed = result["observed"]
        reconstruction = result["proxy"]["reconstruction"]
        residual = result["proxy"]["residual"]
        time_axis = result["time_axis"]
        ax.plot(time_axis, observed, color="black", linewidth=1.0, alpha=0.85, label="Observed")
        ax.plot(time_axis, reconstruction, linewidth=1.0, label="Reconstructed sum")
        ax.plot(time_axis, residual, linewidth=0.85, alpha=0.75, label="Residual")
        ax.set_title(result["display_name"])
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    return fig


def _component_spectra_figure(prediction: np.ndarray, fs: float, model_name: str):
    fig, ax = plt.subplots(figsize=(11, 3.8))
    for idx, component in enumerate(prediction):
        centered = component - np.mean(component)
        spectrum = np.abs(np.fft.rfft(centered))
        freq = np.fft.rfftfreq(component.size, d=1.0 / fs)
        ax.plot(freq, spectrum, linewidth=1.0, label=f"C{idx + 1}")
    ax.set_title(f"Spectral Content of Predicted EEG Components: {model_name}")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def render(registry: list[dict]) -> None:
    """Render the Streamlit view.
    
    Args:
        registry: Model registry entries.
    """
    st.title("Real Data Reconstruction")
    st.caption("Runs checkpoint inference on EDF data from `RealData/`. This page does not call any training code.")
    st.caption("Checkpoint choices on this page are restricted to runs with `train_samples = 8000`.")

    enabled_models = [m for m in registry if m.get("enabled", True)]
    if not enabled_models:
        st.error("No enabled models in registry.")
        return

    edf_files = list_real_data_files()
    if not edf_files:
        st.error(f"No EDF files found in `{REAL_DATA_DIR}`.")
        return

    annotations = parse_summary_annotations()
    model_map = {m["display_name"]: m for m in enabled_models}

    with st.sidebar:
        st.subheader("Real Data Controls")
        selection_mode = st.radio("Model selection", ["Single model", "Compare models"])
        selected_runs: list[dict] = []
        if selection_mode == "Single model":
            model_display = st.selectbox("Model", list(model_map.keys()))
            model_spec = model_map[model_display]
            checkpoints = _available_checkpoints(model_spec)
            if not checkpoints:
                st.warning("No checkpoint found for this model with train_samples = 8000.")
                return
            checkpoint = st.selectbox("Checkpoint", checkpoints)
            selected_runs.append({"display_name": model_display, "model": model_spec, "checkpoint": checkpoint})
        else:
            defaults = list(model_map.keys())[: min(3, len(model_map))]
            selected_models = st.multiselect("Models", list(model_map.keys()), default=defaults)
            configure_checkpoints = st.checkbox("Choose checkpoints per model", value=False)
            for model_display in selected_models:
                model_spec = model_map[model_display]
                checkpoints = _available_checkpoints(model_spec)
                if not checkpoints:
                    continue
                checkpoint = checkpoints[0]
                if configure_checkpoints:
                    checkpoint = st.selectbox(f"{model_display} checkpoint", checkpoints, key=f"real_ckpt_{model_spec['key']}")
                selected_runs.append({"display_name": model_display, "model": model_spec, "checkpoint": checkpoint})

        file_options = {path.name: path for path in edf_files}
        file_name = st.selectbox("EDF file", list(file_options.keys()))
        metadata = read_edf_metadata(str(file_options[file_name]))
        file_annotations = annotations.get(file_name, {})
        seizure_windows = file_annotations.get("seizures", [])

        channel_name = st.selectbox("Channel", metadata.channel_names)
        common_length = _common_signal_length(selected_runs) or 1024
        default_duration = round(common_length / max(metadata.sample_rates[0], 1.0), 3)
        start_default = float(seizure_windows[0]["start_sec"]) if seizure_windows else 0.0
        start_seconds = st.number_input(
            "Start time (s)",
            min_value=0.0,
            max_value=max(0.0, metadata.duration_seconds - (1.0 / max(metadata.sample_rates[0], 1.0))),
            value=min(start_default, max(0.0, metadata.duration_seconds - default_duration)),
            step=1.0,
        )
        duration_seconds = st.number_input(
            "Window duration (s)",
            min_value=0.25,
            max_value=max(0.25, metadata.duration_seconds),
            value=min(default_duration, metadata.duration_seconds),
            step=0.25,
        )
        preprocess_mode = st.selectbox("Preprocessing", ["Z-score", "Demean", "Robust Z-score", "Raw"])
        run = st.button("Run reconstruction", type="primary", use_container_width=True)

    info_cols = st.columns(4)
    info_cols[0].metric("Dataset folder", "RealData")
    info_cols[1].metric("Channels", str(len(metadata.channel_names)))
    info_cols[2].metric("Sampling rate", f"{metadata.sample_rates[0]:.0f} Hz")
    info_cols[3].metric("File duration", f"{metadata.duration_seconds:.1f} s")
    if seizure_windows:
        st.caption(
            "Annotated seizure intervals: "
            + ", ".join(f"{item['start_sec']:.0f}-{item['end_sec']:.0f}s" for item in seizure_windows)
        )

    if not run:
        st.info("Select an EDF file, channel, and model set, then run reconstruction.")
        return

    if not selected_runs:
        st.warning("Select at least one model.")
        return

    try:
        segment = load_edf_segment(file_options[file_name], channel_name, float(start_seconds), float(duration_seconds))
    except Exception as exc:
        st.error(f"Could not load EDF segment: {exc}")
        return

    processed_signal, preprocess_stats = _preprocess_signal(segment.raw_signal, preprocess_mode)
    results = []
    failures = []

    with st.spinner(f"Running inference for {len(selected_runs)} model(s)..."):
        for selected in selected_runs:
            model_spec = selected["model"]
            checkpoint = selected["checkpoint"]
            train_config = load_train_config(checkpoint)
            target_length = int(train_config["signal_length"]) if train_config.get("signal_length") else None
            model_input, input_note = _adapt_length(processed_signal, target_length)
            time_axis = np.arange(model_input.size, dtype=float) / segment.sampling_rate
            try:
                model, load_msg = load_model(
                    model_spec.get("model_key", model_spec["key"]),
                    out_channels=int(train_config.get("out_channels", 3)),
                    checkpoint_path=checkpoint,
                )
                prediction, elapsed_ms = run_inference(model, model_input)
                if prediction.ndim != 2:
                    raise ValueError(f"Unexpected prediction shape: {prediction.shape}")
                proxies = _proxy_metrics(model_input, prediction)
                spectral_df, spectral_summary = _component_spectral_frame(prediction, selected["display_name"], float(segment.sampling_rate))
                checkpoint_path = REPO_ROOT / checkpoint
                results.append(
                    {
                        "display_name": selected["display_name"],
                        "model": model_spec,
                        "checkpoint": checkpoint,
                        "load_msg": load_msg,
                        "input_note": input_note,
                        "time_axis": time_axis,
                        "observed": model_input,
                        "prediction": prediction,
                        "elapsed_ms": elapsed_ms,
                        "parameter_count": int(count_parameters(model)),
                        "checkpoint_mb": float(checkpoint_path.stat().st_size / (1024 * 1024)) if checkpoint_path.exists() else float("nan"),
                        "proxy": proxies,
                        "component_df": _component_frame(prediction, selected["display_name"]),
                        "spectral_df": spectral_df,
                        "spectral_summary": spectral_summary,
                    }
                )
            except Exception as exc:
                failures.append(
                    {
                        "model": selected["display_name"],
                        "checkpoint": checkpoint,
                        "status": f"failed: {exc}",
                    }
                )

    if failures:
        st.warning("Some models could not run on the selected segment.")
        st.dataframe(pd.DataFrame(failures), use_container_width=True, hide_index=True)
    if not results:
        st.error("Inference failed for all selected models.")
        return

    summary_rows = []
    for result in results:
        proxy = result["proxy"]
        summary_rows.append(
            {
                "model": result["display_name"],
                "checkpoint": result["checkpoint"],
                "latency_ms": float(result["elapsed_ms"]),
                "residual_mse": float(proxy["residual_mse"]),
                "residual_mae": float(proxy["residual_mae"]),
                "residual_ratio": float(proxy["residual_ratio"]),
                "reconstruction_corr": float(proxy["reconstruction_corr"]),
                "explained_variance": float(proxy["explained_variance"]),
                "spectral_diversity_score": float(result["spectral_summary"]["spectral_diversity_score"]),
                "mean_pairwise_spectral_overlap": float(result["spectral_summary"]["mean_pairwise_spectral_overlap"]),
                "components": int(result["prediction"].shape[0]),
                "input_length": int(result["observed"].size),
                "input_prep": result["input_note"],
            }
        )
    summary_df = pd.DataFrame(summary_rows).sort_values("residual_mse", ascending=True)
    deployment_df = pd.DataFrame(
        [
            {
                "model": result["display_name"],
                "checkpoint": result["checkpoint"],
                "params": int(result["parameter_count"]),
                "checkpoint_mb": float(result["checkpoint_mb"]),
                "latency_ms": float(result["elapsed_ms"]),
                "signal_length": int(result["observed"].size),
                "sampling_rate_hz": float(segment.sampling_rate),
                "output_components": int(result["prediction"].shape[0]),
            }
            for result in results
        ]
    ).sort_values(["latency_ms", "params"], ascending=[True, True])

    preview_fig, preview_ax = plt.subplots(figsize=(12, 3.0))
    preview_ax.plot(segment.time_axis, segment.raw_signal, color="black", linewidth=0.9)
    preview_ax.set_title(f"Raw EDF segment: {segment.source} | {segment.channel_name} | start {segment.start_seconds:.2f}s")
    preview_ax.set_xlabel("Time (s)")
    preview_fig.tight_layout()
    st.pyplot(preview_fig, clear_figure=False)

    stats_cols = st.columns(4)
    stats_cols[0].metric("Raw mean", f"{preprocess_stats['raw_mean']:.3f}")
    stats_cols[1].metric("Raw std", f"{preprocess_stats['raw_std']:.3f}")
    stats_cols[2].metric("Raw min", f"{preprocess_stats['raw_min']:.3f}")
    stats_cols[3].metric("Raw max", f"{preprocess_stats['raw_max']:.3f}")

    if len(results) > 1:
        st.subheader("Model Reconstruction Comparison")
        comparison_fig = _reconstruction_comparison_figure(results)
        st.pyplot(comparison_fig, clear_figure=False)

    st.subheader("Real EEG Proxy Metrics")
    st.dataframe(summary_df, use_container_width=True, hide_index=True)
    st.subheader("Deployment Cost")
    st.dataframe(deployment_df, use_container_width=True, hide_index=True)

    tabs = st.tabs([result["display_name"] for result in results])
    figures = []
    components_export = []
    spectral_export = []
    for tab, result in zip(tabs, results):
        with tab:
            st.caption(result["load_msg"])
            st.caption(f"Input handling: {result['input_note']} | Inference latency: {result['elapsed_ms']:.2f} ms")
            fig = _real_data_figure(result["time_axis"], result["observed"], result["prediction"], result["display_name"])
            figures.append((result["display_name"], fig))
            st.pyplot(fig, clear_figure=False)
            st.dataframe(result["component_df"], use_container_width=True, hide_index=True)
            components_export.append(result["component_df"])
            st.caption(
                "Spectral diversity is better when overlap between predicted components is lower. "
                f"Current score: {result['spectral_summary']['spectral_diversity_score']:.3f}"
            )
            spectra_fig = _component_spectra_figure(result["prediction"], float(segment.sampling_rate), result["display_name"])
            st.pyplot(spectra_fig, clear_figure=False)
            st.dataframe(result["spectral_df"], use_container_width=True, hide_index=True)
            spectral_export.append(result["spectral_df"])

    st.download_button(
        "Download summary CSV",
        summary_df.to_csv(index=False).encode("utf-8"),
        file_name="real_data_reconstruction_summary.csv",
        mime="text/csv",
    )

    if st.button("Export summary to app/exports"):
        path = export_dataframe(summary_df, "real_data_reconstruction_summary")
        st.success(f"Saved: `{path.relative_to(REPO_ROOT)}`")

    if components_export and st.button("Export component stats to app/exports"):
        path = export_dataframe(pd.concat(components_export, ignore_index=True), "real_data_component_stats")
        st.success(f"Saved: `{path.relative_to(REPO_ROOT)}`")

    if spectral_export and st.button("Export spectral stats to app/exports"):
        path = export_dataframe(pd.concat(spectral_export, ignore_index=True), "real_data_component_spectral_stats")
        st.success(f"Saved: `{path.relative_to(REPO_ROOT)}`")

    if st.button("Export deployment cost to app/exports"):
        path = export_dataframe(deployment_df, "real_data_deployment_cost")
        st.success(f"Saved: `{path.relative_to(REPO_ROOT)}`")

    if figures:
        png_buffer = io.BytesIO()
        figures[0][1].savefig(png_buffer, format="png", dpi=160, bbox_inches="tight")
        png_buffer.seek(0)
        st.download_button(
            "Download first model plot PNG",
            png_buffer.getvalue(),
            file_name="real_data_reconstruction.png",
            mime="image/png",
        )
        if st.button("Export first model plot to app/exports"):
            path = export_figure(figures[0][1], "real_data_reconstruction")
            st.success(f"Saved: `{path.relative_to(REPO_ROOT)}`")

    st.session_state["last_real_data_reconstruction"] = {
        "file": file_name,
        "channel": channel_name,
        "start_seconds": float(segment.start_seconds),
        "duration_seconds": float(duration_seconds),
        "preprocessing": preprocess_mode,
        "summary": summary_df.to_dict("records"),
    }
