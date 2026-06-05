from __future__ import annotations

import time
import re

import pandas as pd
import streamlit as st

from app.services.paths import REPO_ROOT
from app.services.run_store import append_run, update_run, create_run_record
from app.services.signal_service import PURE_MODE, SUPPORTED_COMPONENT_TYPES, validate_pure_component_types
from app.services.training_service import (
    TrainingRequest,
    drain_training_output,
    elapsed_seconds,
    expected_checkpoint_path,
    parse_batch_line,
    parse_progress_line,
    read_final_report,
    run_training,
    start_training_job,
    training_script_path,
)


def _slugify_suffix(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", value.strip()).strip("_").lower()
    return slug


def render(registry: list[dict]) -> None:
    """Render the Streamlit view.
    
    Args:
        registry: Model registry entries.
    """
    st.title("Training")
    enabled = [m for m in registry if m.get("enabled", True)]
    if not enabled:
        st.warning("No enabled models available in registry.")
        return

    trainable = [m for m in enabled if training_script_path(m["key"]) is not None]
    train_mode = st.radio("Training target", ["Single model", "All trainable models"], horizontal=True)
    model_map = {m["display_name"]: m for m in trainable}
    model: dict | None = None
    if train_mode == "Single model":
        model_display = st.selectbox("Model", list(model_map.keys()))
        model = model_map[model_display]
    else:
        st.caption(f"Will train {len(trainable)} enabled models with configured train scripts.")

    if train_mode == "Single model" and model and training_script_path(model["key"]) is None:
        st.error(f"No training script mapping found for `{model['key']}`.")
        return
    if train_mode == "All trainable models" and not trainable:
        st.error("No enabled trainable models found.")
        return

    col1, col2, col3 = st.columns(3)
    epochs = col1.number_input("Epochs", min_value=1, max_value=500, value=20, step=1)
    batch_size = col2.number_input("Batch size", min_value=1, max_value=1024, value=32, step=1)
    learning_rate = col3.number_input("Learning rate", min_value=1e-6, max_value=1.0, value=1e-3, format="%.6f")

    col4, col5, col6 = st.columns(3)
    weight_decay = col4.number_input("Weight decay", min_value=0.0, max_value=1.0, value=1e-5, format="%.6f")
    seed = col5.number_input("Seed", min_value=0, max_value=999999, value=42, step=1)
    device = col6.selectbox("Device", ["auto", "cpu"])

    loss_cols = st.columns(3)
    component_loss_weight = loss_cols[0].number_input("Component loss weight", min_value=0.0, max_value=10.0, value=1.0, step=0.05)
    reconstruction_loss_weight = loss_cols[1].number_input("Reconstruction loss weight", min_value=0.0, max_value=10.0, value=0.5, step=0.05)
    spectral_loss_weight = loss_cols[2].number_input("Spectral loss weight", min_value=0.0, max_value=10.0, value=0.05, step=0.01, format="%.3f")

    st.subheader("Signal Generation Controls")
    training_mode_label = st.radio("Training signal mode", ["Non-pure preset", "Pure components"])
    generation_mode = PURE_MODE if training_mode_label == "Pure components" else "preset"
    signal_type = "mixed"
    selected_component_types: list[str]

    if generation_mode == PURE_MODE:
        selected_component_types = []
        cols = st.columns(len(SUPPORTED_COMPONENT_TYPES))
        for idx, component_type in enumerate(SUPPORTED_COMPONENT_TYPES):
            if cols[idx].checkbox(component_type, value=component_type in ["harmonic", "amfm", "chirp"], key=f"train_component_{component_type}"):
                selected_component_types.append(component_type)
    else:
        c1, c2 = st.columns(2)
        signal_type = c1.selectbox("Signal preset", ["mixed", "harmonic", "amfm", "chirp"])
        preset_count = c2.slider("Number of components", min_value=1, max_value=5, value=3, step=1)
        sequence = ["harmonic", "amfm", "chirp"]
        if signal_type == "mixed":
            selected_component_types = [sequence[idx % len(sequence)] for idx in range(int(preset_count))]
        else:
            selected_component_types = [signal_type for _ in range(int(preset_count))]

    valid_components, component_msg = validate_pure_component_types(selected_component_types)
    if not valid_components:
        st.warning(component_msg)
    n_components = len(selected_component_types)
    st.caption(
        f"Training mode: {'Pure components' if generation_mode == PURE_MODE else 'Non-pure preset'} | "
        f"Components: {', '.join(selected_component_types) if selected_component_types else 'none'} | Count: {n_components}"
    )

    c3, c4 = st.columns(2)
    noise_level = c3.slider("Noise level", min_value=0.0, max_value=0.2, value=0.03, step=0.005)
    duration = c4.slider("Duration (s)", min_value=0.5, max_value=8.0, value=4.0, step=0.1)
    fs = st.number_input("Sampling rate (Hz)", min_value=64, max_value=4096, value=256, step=64)

    d1, d2 = st.columns(2)
    train_samples = d1.number_input("Train set size", min_value=32, max_value=100000, value=4000, step=32)
    val_samples = d2.number_input("Validation set size", min_value=32, max_value=50000, value=400, step=32)
    test_samples = st.number_input("Test set size", min_value=32, max_value=50000, value=400, step=32)

    st.subheader("Optional Depth Expansion")
    variant_col, mode_col = st.columns([2, 3])
    run_name_suffix = variant_col.text_input(
        "Run name suffix",
        value="",
        help="Optional suffix used to name this trained variant separately in saved outputs.",
    )
    apply_depth_expansion = mode_col.checkbox(
        "Apply depth expansion to shallow Conv1D models",
        value=False,
        help="Adds configurable extra Conv1D refinement layers to shallow Conv1D baselines without changing the modular ablation system.",
    )
    depth_cfg = {
        "extra_conv_layers": 0,
        "extra_conv_kernel_size": 3,
        "extra_conv_channels": 0,
        "extra_conv_dilation": 1,
        "extra_conv_activation": "gelu",
        "extra_conv_norm": "groupnorm",
        "extra_conv_dropout": 0.1,
        "extra_conv_num_groups": 8,
        "extra_conv_residual": False,
    }
    if apply_depth_expansion:
        depth_cols_1 = st.columns(4)
        depth_cols_2 = st.columns(4)
        depth_cfg["extra_conv_layers"] = int(depth_cols_1[0].selectbox("Extra Conv1D layers", [0, 1, 2, 3, 4, 5, 6], index=2))
        depth_cfg["extra_conv_kernel_size"] = int(depth_cols_1[1].selectbox("Extra kernel size", [3, 5, 7, 9], index=0))
        depth_cfg["extra_conv_channels"] = int(depth_cols_1[2].number_input("Extra channels (0=same)", min_value=0, max_value=512, value=0, step=8))
        depth_cfg["extra_conv_dilation"] = int(depth_cols_1[3].selectbox("Extra dilation", [1, 2, 4, 8], index=0))
        depth_cfg["extra_conv_activation"] = depth_cols_2[0].selectbox("Extra activation", ["relu", "gelu", "tanh"], index=1)
        depth_cfg["extra_conv_norm"] = depth_cols_2[1].selectbox("Extra normalization", ["groupnorm", "batchnorm", "none"], index=0)
        depth_cfg["extra_conv_dropout"] = float(depth_cols_2[2].slider("Extra dropout", min_value=0.0, max_value=0.6, value=0.1, step=0.05))
        depth_cfg["extra_conv_num_groups"] = int(depth_cols_2[3].selectbox("Extra GroupNorm groups", [1, 2, 4, 8, 16], index=3))
        depth_cfg["extra_conv_residual"] = st.checkbox("Use residual connection inside extra Conv1D blocks", value=False)
        st.caption("Depth expansion is applied only to shallow Conv1D-family models. Deep and non-Conv1D models keep their original architecture.")

    suffix_slug = _slugify_suffix(run_name_suffix)
    if train_mode == "Single model" and model and generation_mode == PURE_MODE:
        component_slug = "_".join(selected_component_types) if selected_component_types else "none"
        output_dir = f"decomposers/ML_methods/NN_based/saved_models/{model['key']}_pure_{component_slug}"
    elif train_mode == "Single model" and model:
        output_dir = f"decomposers/ML_methods/NN_based/saved_models/{model['key']}"
    if train_mode == "Single model" and model and suffix_slug:
        output_dir = f"{output_dir}_{suffix_slug}"
    st.caption("Training uses the selected component families; the model output channels are set to the selected count.")

    def build_data_settings() -> dict:
        return {
            "generation_mode": generation_mode,
            "signal_type": signal_type,
            "n_components": int(n_components),
            "selected_component_types": selected_component_types,
            "noise_level": float(noise_level),
            "duration": float(duration),
            "fs": int(fs),
            "train_samples": int(train_samples),
            "val_samples": int(val_samples),
            "test_samples": int(test_samples),
            "total_samples": int(train_samples) + int(val_samples) + int(test_samples),
        }

    def build_hyperparams() -> dict:
        return {
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "learning_rate": float(learning_rate),
            "weight_decay": float(weight_decay),
            "component_loss_weight": float(component_loss_weight),
            "reconstruction_loss_weight": float(reconstruction_loss_weight),
            "spectral_loss_weight": float(spectral_loss_weight),
            "seed": int(seed),
            "device": device,
            "run_name_suffix": suffix_slug,
            **depth_cfg,
        }

    def build_request(target_model: dict, target_output_dir: str) -> TrainingRequest:
        supports_depth_expansion = str(target_model.get("family", "")).strip().lower() == "conv1d"
        return TrainingRequest(
            model_name=target_model["key"],
            epochs=int(epochs),
            batch_size=int(batch_size),
            learning_rate=float(learning_rate),
            weight_decay=float(weight_decay),
            seed=int(seed),
            fs=int(fs),
            duration=float(duration),
            train_samples=int(train_samples),
            val_samples=int(val_samples),
            test_samples=int(test_samples),
            noise_level=float(noise_level),
            device=device,
            output_dir=target_output_dir,
            component_types=selected_component_types,
            component_loss_weight=float(component_loss_weight),
            reconstruction_loss_weight=float(reconstruction_loss_weight),
            spectral_loss_weight=float(spectral_loss_weight),
            experiment_name=suffix_slug,
            supports_depth_expansion=supports_depth_expansion and apply_depth_expansion,
            **depth_cfg,
        )

    start_label = "Start training" if train_mode == "Single model" else "Start training all"
    if st.button(start_label, type="primary", disabled=not valid_components):
        if train_mode == "All trainable models":
            st.subheader("Live Training Output")
            status_placeholder = st.empty()
            table_placeholder = st.empty()
            log_placeholder = st.empty()
            command_placeholder = st.empty()
            jobs = {}
            run_records = {}
            logs_by_model: dict[str, list[str]] = {}
            progress_by_model: dict[str, dict] = {}
            batch_by_model: dict[str, dict] = {}

            for idx, target_model in enumerate(trainable):
                if generation_mode == PURE_MODE:
                    component_slug = "_".join(selected_component_types) if selected_component_types else "none"
                    target_output_dir = f"decomposers/ML_methods/NN_based/saved_models/{target_model['key']}_pure_{component_slug}"
                else:
                    target_output_dir = f"decomposers/ML_methods/NN_based/saved_models/{target_model['key']}"
                if suffix_slug:
                    target_output_dir = f"{target_output_dir}_{suffix_slug}"
                run_record = create_run_record(
                    model_name=target_model["key"],
                    depth_label=target_model.get("depth_label", ""),
                    data_settings=build_data_settings(),
                    hyperparams=build_hyperparams(),
                    notes="Parallel train-all run started from Streamlit dashboard.",
                )
                append_run(run_record)
                req = build_request(target_model, target_output_dir)
                try:
                    job = start_training_job(target_model, req)
                    jobs[target_model["key"]] = job
                    run_records[target_model["key"]] = run_record
                    logs_by_model[target_model["key"]] = []
                except Exception as exc:
                    update_run(run_record["run_id"], {"status": "failed", "notes": f"Training failed to start: {exc}"})
                    st.error(f"{target_model['key']} failed to start: {exc}")
            command_placeholder.code("\n".join(" ".join(job.command) for job in jobs.values()), language="bash")

            while jobs:
                rows = []
                completed_keys = []
                for key, job in list(jobs.items()):
                    for line in drain_training_output(job):
                        logs_by_model[key].append(line.rstrip())
                        parsed = parse_progress_line(line)
                        if parsed:
                            progress_by_model[key] = parsed
                            update_run(run_records[key]["run_id"], {"best_validation_metric": parsed.get("val_loss")})
                        parsed_batch = parse_batch_line(line)
                        if parsed_batch:
                            batch_by_model[key] = parsed_batch

                    return_code = job.process.poll()
                    progress = progress_by_model.get(key, {})
                    batch = batch_by_model.get(key, {})
                    rows.append(
                        {
                            "model": key,
                            "status": "running" if return_code is None else ("completed" if return_code == 0 else "failed"),
                            "elapsed_sec": round(elapsed_seconds(job), 1),
                            "epoch": progress.get("epoch"),
                            "train_loss": progress.get("train_loss"),
                            "val_loss": progress.get("val_loss"),
                            "val_corr": progress.get("val_corr"),
                            "latest_batch_loss": batch.get("loss"),
                            "train_samples": int(train_samples),
                            "val_samples": int(val_samples),
                            "test_samples": int(test_samples),
                        }
                    )
                    if return_code is not None:
                        final_report = read_final_report(job.request.output_dir)
                        summary = (final_report or {}).get("test_summary", {})
                        update_run(
                            run_records[key]["run_id"],
                            {
                                "status": "completed" if return_code == 0 else "failed",
                                "test_summary": summary if return_code == 0 else {},
                                "checkpoint_path": expected_checkpoint_path(key, job.request.output_dir),
                                "training_time_seconds": elapsed_seconds(job),
                                "notes": "Parallel train-all completed." if return_code == 0 else f"Training exited with code {return_code}",
                            },
                        )
                        completed_keys.append(key)

                for key in completed_keys:
                    jobs.pop(key, None)

                status_placeholder.info(f"Training {len(jobs)} model(s). Completed {len(run_records) - len(jobs)} of {len(run_records)}.")
                if rows:
                    table_placeholder.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                log_text = []
                for key, lines in logs_by_model.items():
                    log_text.append(f"[{key}]")
                    log_text.extend(lines[-8:])
                log_placeholder.text("\n".join(log_text[-120:]))
                if jobs:
                    time.sleep(0.75)

            st.success("Train-all run finished.")
            st.info("Open Model Comparison to inspect updated reports, training time, and dataset sizes.")
            return

        if model is None:
            st.error("No model selected.")
            return
        run_record = create_run_record(
            model_name=model["key"],
            depth_label=model.get("depth_label", ""),
            data_settings=build_data_settings(),
            hyperparams=build_hyperparams(),
        )
        append_run(run_record)

        req = build_request(model, output_dir)
        st.subheader("Live Training Output")
        progress_bar = st.progress(0.0)
        metric_cols = st.columns(4)
        status_placeholder = st.empty()
        epoch_table_placeholder = st.empty()
        loss_chart_placeholder = st.empty()
        validation_chart_placeholder = st.empty()
        batch_chart_placeholder = st.empty()
        log_placeholder = st.empty()
        with st.spinner("Training in progress..."):
            try:
                process, cmd = run_training(req)
                st.code(" ".join(cmd), language="bash")
                status_placeholder.info("Training process started. Waiting for first output...")
                logs: list[str] = []
                latest_progress = {}
                epoch_rows: list[dict] = []
                batch_rows: list[dict] = []
                if process.stdout is not None:
                    for line in process.stdout:
                        logs.append(line.rstrip())
                        status_placeholder.info(f"Latest output: {line.rstrip()}")
                        parsed_batch = parse_batch_line(line)
                        if parsed_batch:
                            parsed_batch["epoch"] = latest_progress.get("epoch")
                            batch_rows.append(parsed_batch)
                            metric_cols[3].metric("Latest batch loss", f"{parsed_batch['loss']:.6f}")
                            batch_df = pd.DataFrame(batch_rows[-200:])
                            batch_chart_placeholder.line_chart(batch_df[["loss"]])

                        parsed = parse_progress_line(line)
                        if parsed:
                            latest_progress = parsed
                            epoch_rows.append(parsed)
                            progress_bar.progress(min(float(parsed["epoch"]) / float(epochs), 1.0))
                            metric_cols[0].metric("Epoch", f"{parsed['epoch']} / {int(epochs)}")
                            metric_cols[1].metric("Train loss", f"{parsed['train_loss']:.6f}")
                            metric_cols[2].metric("Val loss", f"{parsed['val_loss']:.6f}")

                            epoch_df = pd.DataFrame(epoch_rows)
                            epoch_table_placeholder.dataframe(epoch_df, use_container_width=True, hide_index=True)
                            loss_chart_placeholder.line_chart(epoch_df.set_index("epoch")[["train_loss", "val_loss"]])
                            validation_cols = [col for col in ["val_corr", "val_snr_db", "mixture_mse"] if col in epoch_df.columns]
                            if validation_cols:
                                validation_chart_placeholder.line_chart(epoch_df.set_index("epoch")[validation_cols])

                        log_placeholder.text("\n".join(logs[-25:]))
                return_code = process.wait()
            except Exception as exc:
                update_run(run_record["run_id"], {"status": "failed", "notes": f"Training failed: {exc}"})
                st.error(f"Training failed: {exc}")
                return

        final_report = read_final_report(output_dir)
        if return_code == 0:
            summary = (final_report or {}).get("test_summary", {})
            macro = summary.get("macro_average", {}) if isinstance(summary, dict) else {}
            update_run(
                run_record["run_id"],
                {
                    "status": "completed",
                    "best_validation_metric": latest_progress.get("val_loss"),
                    "test_summary": summary,
                    "checkpoint_path": expected_checkpoint_path(model["key"], output_dir),
                    "training_time_seconds": (final_report.get("training_metadata", {}) or {}).get("elapsed_seconds"),
                    "notes": "Training completed via Streamlit dashboard.",
                },
            )
            st.success("Training completed.")
            if macro:
                st.write(f"Best available test macro corr: {macro.get('corr', float('nan')):.4f}")
        else:
            update_run(
                run_record["run_id"],
                {"status": "failed", "notes": f"Training process exited with code {return_code}"},
            )
            st.error(f"Training process exited with code {return_code}.")

        st.info(
            "Next steps: open Reconstruction Inspector for qualitative checks, Model Comparison for report-level analysis, "
            "or Runs History for run tracking."
        )
        st.caption(f"Output directory: `{output_dir}` (`{(REPO_ROOT / output_dir)}`)")
