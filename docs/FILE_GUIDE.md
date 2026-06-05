# File Guide

Concise map of the maintained source/config files in this repository. Generated artifacts, cached bytecode, exported reports, saved checkpoints, and local data folders are intentionally omitted.

Use this as orientation documentation; it is not a replacement for API-level docstrings on complex functions.

## Application

### `app/__init__.py`
- Purpose: Streamlit research dashboard package.

### `app/components/__init__.py`
- Purpose: Reusable Streamlit component package.

### `app/components/charts.py`
- Purpose: Reusable Matplotlib figures for reconstruction, model comparison, and frequency-domain views.
- Main functions: reconstruction_overview_figure, reconstruction_comparison_figure, fft_comparison_figure

### `app/components/forms.py`
- Purpose: Small reusable Streamlit form controls.
- Main functions: bool_filter

### `app/components/metrics_cards.py`
- Purpose: Metric-card rendering helpers for dashboard summaries.
- Main functions: render_primary_metrics

### `app/components/tables.py`
- Purpose: Reusable dataframe/table rendering helpers.
- Main functions: render_dataframe

### `app/config/model_registry.json`
- Purpose: Model registry consumed by training, inference, evaluation, and dashboard comparison flows.

### `app/main.py`
- Purpose: Streamlit dashboard entry point and page router.
- Main functions: main

### `app/pages/__init__.py`
- Purpose: Streamlit page package.

### `app/pages/comparison.py`
- Purpose: Streamlit model-comparison page for evaluation runs, filtered tables, and comparison charts.
- Main functions: render

### `app/pages/cost_data.py`
- Purpose: Streamlit page for limited-data/cost comparison research outputs.
- Main functions: render

### `app/pages/dashboard.py`
- Purpose: Streamlit landing page with high-level project and run summaries.
- Main functions: render

### `app/pages/error_analysis.py`
- Purpose: Streamlit page for inspecting prediction errors and hard cases.
- Main functions: render

### `app/pages/export_page.py`
- Purpose: Streamlit page for exporting dashboard data, figures, and app bundles.
- Main functions: render

### `app/pages/generator_lab.py`
- Purpose: Streamlit signal-generation playground for synthetic mixtures and components.
- Main functions: render

### `app/pages/hard_signal_mining.py`
- Purpose: Streamlit page for mining statistically hard synthetic signal cases.
- Main functions: estimate_descriptors, compute_hardness, render

### `app/pages/real_data.py`
- Purpose: Streamlit page for loading and inspecting real EEG/EDF signal segments.
- Main functions: render

### `app/pages/reconstruction.py`
- Purpose: Streamlit reconstruction page for running inference and visualizing component predictions.
- Main functions: render

### `app/pages/registry.py`
- Purpose: Streamlit page for viewing and editing model registry entries.
- Main functions: render

### `app/pages/research.py`
- Purpose: Streamlit research dashboard for aggregate model, ablation, and trade-off charts.
- Main functions: render

### `app/pages/runs.py`
- Purpose: Streamlit page for saved runs, favorites, and sample-level analysis records.
- Main functions: render

### `app/pages/training.py`
- Purpose: Streamlit training launcher and live log/progress page.
- Main functions: render

### `app/services/__init__.py`
- Purpose: Dashboard service-layer package.

### `app/services/app_export_service.py`
- Purpose: Builds full dashboard export bundles with source files, data, and manifest metadata.
- Main classes: ManifestEntry
- Main functions: export_complete_app_bundle

### `app/services/comparison_service.py`
- Purpose: Runs model-comparison scripts and evaluates registered models on selected signal sets.
- Main functions: comparison_script_path, build_comparison_command, run_model_comparison, comparison_table_path, resolve_model_checkpoint, add_paired_statistics, evaluate_registry_on_components

### `app/services/cost_data_service.py`
- Purpose: Loads, aggregates, and charts limited-data/cost comparison research results.
- Main classes: CostResearchDataset
- Main functions: prepare_cost_research_dataset, aggregate_cost_dataset, filter_cost_dataset, compute_cost_summary, compute_limited_data_conclusions, build_cost_tables, missing_data_report, build_cost_chart_catalog, retest_data_availability_models

### `app/services/export_service.py`
- Purpose: Shared export helpers for dataframes, JSON payloads, figures, and export directories.
- Main functions: export_dataframe, export_json, export_figure, create_export_bundle_dir, export_dataframe_to_dir, export_json_to_dir, export_figure_to_dir

### `app/services/inference_service.py`
- Purpose: Loads trained models, runs inference, computes metrics, and counts parameters.
- Main functions: load_train_config, load_model, run_inference, evaluate_prediction, count_parameters

### `app/services/paths.py`
- Purpose: Central repository and app directory constants plus directory creation helpers.
- Main functions: ensure_app_dirs

### `app/services/real_data_service.py`
- Purpose: Discovers EDF files, reads metadata, extracts segments, and parses annotations.
- Main classes: EdfMetadata, RealSignalSegment
- Main functions: list_real_data_files, read_edf_metadata, load_edf_segment, parse_summary_annotations

### `app/services/registry_service.py`
- Purpose: Loads, validates, saves, and enriches model registry entries.
- Main classes: ModelRegistryEntry
- Main functions: ensure_registry_file, load_registry, save_registry, validate_registry, discover_checkpoints, absolute_path_from_repo

### `app/services/report_loader.py`
- Purpose: Discovers evaluation reports and converts them into comparison dataframes.
- Main classes: ReportLoadResult
- Main functions: discover_report_files, infer_checkpoint, load_reports_dataframe, enrich_with_model_stats

### `app/services/research_service.py`
- Purpose: Normalizes research data and builds aggregate charts/tables for thesis analysis.
- Main classes: ResearchDataset, ChartSpec
- Main functions: prepare_research_dataset, available_columns, prepare_chart_generation_dataframe, available_module_columns, module_display_names, apply_ablation_scope, best_row, group_summary, compute_summary_statistics, best_accuracy_efficiency_tradeoff, compute_pareto_frontier, build_results_tables

### `app/services/run_store.py`
- Purpose: Persists run history and sample-analysis records for the dashboard.
- Main functions: load_runs, save_runs, create_run_record, append_run, update_run, delete_run, toggle_favorite, load_sample_analysis, append_sample_analysis

### `app/services/signal_service.py`
- Purpose: Generates synthetic mixtures/components and FFT views used by the app.
- Main classes: SignalConfig, GeneratedSignal
- Main functions: build_time_axis, component_builders, make_unique_component_names, validate_pure_component_types, infer_component_sequence, generate_signal, generate_components, fft_magnitude

### `app/services/statistical_hardcase_service.py`
- Purpose: Evaluates models on generated hard-case signals and builds statistical summaries/figures.
- Main classes: StatisticalRunResult
- Main functions: generate_statistical_signal, evaluate_model_on_signal, compute_hard_case_score, compute_signal_level_comparison, build_hard_case_table, summarize_statistical_results, build_overall_summary_text, boxplot_metric_by_model, violin_metric_by_model_type, histogram_hardcase_scores, bar_failure_rate_per_model, bar_average_hardcase_per_model

### `app/services/test_signal_diagnostics.py`
- Purpose: Computes diagnostic labels, thresholds, explanations, and figures for failed signals.
- Main classes: FailureThresholds
- Main functions: thresholds_to_dict, build_thresholds, compute_component_diagnostics, compute_signal_difficulty_descriptors, build_explanation_text, build_signal_difficulty_text, compare_model_results, figure_prediction_preview, figure_component_metric_comparison, figure_failure_label_heatmap, save_diagnostic_artifact

### `app/services/training_service.py`
- Purpose: Builds and runs training commands, tracks subprocess logs, and reads final reports.
- Main classes: TrainingRequest, TrainingJob
- Main functions: training_script_path, build_command, expected_checkpoint_path, run_training, start_training_job, drain_training_output, elapsed_seconds, parse_progress_line, parse_batch_line, read_final_report


## Neural Decomposers

### `decomposers/ML_methods/NN_based/datasets/synthetic.py`
- Purpose: PyTorch dataset for generated synthetic decomposition training samples.
- Main classes: SyntheticSignalDataset

### `decomposers/ML_methods/NN_based/datasets/utils.py`
- Purpose: Signal normalization, random mixing weights, and noise utilities for datasets.
- Main functions: normalize_signal, random_weights, add_noise

### `decomposers/ML_methods/NN_based/experiment_catalog.py`
- Purpose: Central catalog of supported training/evaluation scripts, reports, and model keys.
- Main classes: ModelExperimentSpec
- Main functions: get_experiment_spec, training_specs, evaluation_specs

### `decomposers/ML_methods/NN_based/models/__init__.py`
- Purpose: Model factory and shared exports for deep and shallow decomposer architectures.
- Main classes: FuseShallowDecomposer, ThreeParallelShallowBranchDecomposer
- Main functions: create_model

### `decomposers/ML_methods/NN_based/models/autoencoderbased/__init__.py`
- Purpose: Package initializer for this module group.

### `decomposers/ML_methods/NN_based/models/autoencoderbased/autoencoderbased.py`
- Purpose: Implements AutoencoderDecomposerConfig, AutoencoderDecomposer, ConvAutoencoderDecomposer.
- Main classes: AutoencoderDecomposerConfig, AutoencoderDecomposer, ConvAutoencoderDecomposer

### `decomposers/ML_methods/NN_based/models/autoencoderbased/trainAutoencoderbased.py`
- Purpose: Training entry point for Autoencoderbased.
- Main classes: TrainConfig
- Main functions: parse_hidden_channels, parse_args, create_model, main

### `decomposers/ML_methods/NN_based/models/rnnbased/__init__.py`
- Purpose: Package initializer for this module group.

### `decomposers/ML_methods/NN_based/models/rnnbased/rnnbased.py`
- Purpose: Implements RNNDecomposerConfig, RNNDecomposer, LSTMDecomposer.
- Main classes: RNNDecomposerConfig, RNNDecomposer, LSTMDecomposer, GRUDecomposer

### `decomposers/ML_methods/NN_based/models/rnnbased/trainRnnbased.py`
- Purpose: Training entry point for Rnnbased.
- Main classes: TrainConfig
- Main functions: parse_args, create_model, main

### `decomposers/ML_methods/NN_based/models/sepformer/__init__.py`
- Purpose: Package initializer for this module group.

### `decomposers/ML_methods/NN_based/models/sepformer/sepformer.py`
- Purpose: Implements SepFormerDecomposerConfig, DualPathTransformerBlock, SepFormerDecomposer.
- Main classes: SepFormerDecomposerConfig, DualPathTransformerBlock, SepFormerDecomposer

### `decomposers/ML_methods/NN_based/models/sepformer/trainSepformer.py`
- Purpose: Training entry point for Sepformer.
- Main classes: TrainConfig
- Main functions: parse_args, create_model, main

### `decomposers/ML_methods/NN_based/models/tasnet/__init__.py`
- Purpose: Package initializer for this module group.

### `decomposers/ML_methods/NN_based/models/tasnet/tasnet.py`
- Purpose: Implements TasNetDecomposerConfig, TemporalBlock, TasNetDecomposer.
- Main classes: TasNetDecomposerConfig, TemporalBlock, TasNetDecomposer, ConvTasNetDecomposer

### `decomposers/ML_methods/NN_based/models/tasnet/trainTasnet.py`
- Purpose: Training entry point for Tasnet.
- Main classes: TrainConfig
- Main functions: parse_args, create_model, main

### `decomposers/ML_methods/NN_based/models/unet1d/__init__.py`
- Purpose: Package initializer for this module group.

### `decomposers/ML_methods/NN_based/models/unet1d/trainUnet1d.py`
- Purpose: Training entry point for Unet1d.
- Main classes: TrainConfig
- Main functions: parse_args, create_model, main

### `decomposers/ML_methods/NN_based/models/unet1d/unet1d.py`
- Purpose: Implements DoubleConv1D, Down1D, Up1D.
- Main classes: DoubleConv1D, Down1D, Up1D, UNet1D

### `decomposers/ML_methods/NN_based/models_shallow/__init__.py`
- Purpose: Package initializer for this module group.

### `decomposers/ML_methods/NN_based/models_shallow/common/__init__.py`
- Purpose: Package initializer for this module group.

### `decomposers/ML_methods/NN_based/models_shallow/common/depth_expansion.py`
- Purpose: Implements DepthExpansionBlock, DepthExpansion1D.
- Main classes: DepthExpansionBlock, DepthExpansion1D
- Main functions: make_activation, make_norm

### `decomposers/ML_methods/NN_based/models_shallow/common/multiscale_branches.py`
- Purpose: Implements ThreeParallelShallowBranches.
- Main classes: ThreeParallelShallowBranches

### `decomposers/ML_methods/NN_based/models_shallow/conv1Dnetwork/__init__.py`
- Purpose: Package initializer for this module group.

### `decomposers/ML_methods/NN_based/models_shallow/conv1Dnetwork/conv1Dnetwork.py`
- Purpose: Implements ShallowConv1DDecomposerConfig, ShallowConv1DDecomposer.
- Main classes: ShallowConv1DDecomposerConfig, ShallowConv1DDecomposer

### `decomposers/ML_methods/NN_based/models_shallow/conv1Dnetwork/trainConv1Dnetwork.py`
- Purpose: Training entry point for Conv1Dnetwork.
- Main classes: TrainConfig
- Main functions: parse_args, create_model, main

### `decomposers/ML_methods/NN_based/models_shallow/fuse/__init__.py`
- Purpose: Package initializer for this module group.

### `decomposers/ML_methods/NN_based/models_shallow/fuse/fuse.py`
- Purpose: Implements Fuse.
- Main classes: Fuse

### `decomposers/ML_methods/NN_based/models_shallow/fuse/trainFuse.py`
- Purpose: Training entry point for Fuse.
- Main classes: TrainConfig, ThreeParallelWithFuseDecomposer
- Main functions: parse_args, create_model, main

### `decomposers/ML_methods/NN_based/models_shallow/mlp/__init__.py`
- Purpose: Package initializer for this module group.

### `decomposers/ML_methods/NN_based/models_shallow/mlp/singlehead/__init__.py`
- Purpose: Package initializer for this module group.

### `decomposers/ML_methods/NN_based/models_shallow/mlp/singlehead/singlehead.py`
- Purpose: Implements SingleHeadMLPDecomposerConfig, SingleHeadMLPDecomposer.
- Main classes: SingleHeadMLPDecomposerConfig, SingleHeadMLPDecomposer

### `decomposers/ML_methods/NN_based/models_shallow/mlp/singlehead/trainSingleHeadMLP.py`
- Purpose: Training entry point for SingleHeadMLP.
- Main classes: TrainConfig
- Main functions: parse_args, create_model, main

### `decomposers/ML_methods/NN_based/models_shallow/multiscale_branches/__init__.py`
- Purpose: Package initializer for this module group.

### `decomposers/ML_methods/NN_based/models_shallow/multiscale_branches/attention_stem_bilinear_fusion_multiple_head_multiscale_branches_decomposer.py`
- Purpose: Implements AttentionStemBilinearFusionMultipleHeadMultiScaleBranchesDecomposerConfig, BilinearFusion, AttentionStemBilinearFusionMultipleHeadMultiScaleBranchesDecomposer.
- Main classes: AttentionStemBilinearFusionMultipleHeadMultiScaleBranchesDecomposerConfig, BilinearFusion, AttentionStemBilinearFusionMultipleHeadMultiScaleBranchesDecomposer

### `decomposers/ML_methods/NN_based/models_shallow/multiscale_branches/attention_stem_multi_head_multiscale_tcn_decomposer.py`
- Purpose: Implements AttentionStemMultiHeadMultiScaleTCNDecomposerConfig, SameLengthConv1d, TCNResidualBlock.
- Main classes: AttentionStemMultiHeadMultiScaleTCNDecomposerConfig, SameLengthConv1d, TCNResidualBlock, ComponentHead, AttentionStemMultiHeadMultiScaleTCNDecomposer

### `decomposers/ML_methods/NN_based/models_shallow/multiscale_branches/attention_stem_multiple_head_multiscale_branches_decomposer.py`
- Purpose: Implements AttentionStemMultipleHeadMultiScaleBranchesDecomposerConfig, AttentionStemMultipleHeadMultiScaleBranchesDecomposer.
- Main classes: AttentionStemMultipleHeadMultiScaleBranchesDecomposerConfig, AttentionStemMultipleHeadMultiScaleBranchesDecomposer

### `decomposers/ML_methods/NN_based/models_shallow/multiscale_branches/multiple_head_multiscale_branches_decomposer.py`
- Purpose: Implements MultipleHeadMultiScaleBranchesDecomposerConfig, MultipleHeadMultiScaleBranchesDecomposer.
- Main classes: MultipleHeadMultiScaleBranchesDecomposerConfig, MultipleHeadMultiScaleBranchesDecomposer

### `decomposers/ML_methods/NN_based/models_shallow/multiscale_branches/multiscale_branches_decomposer.py`
- Purpose: Implements MultiScaleBranchesDecomposerConfig, MultiScaleBranchesDecomposer.
- Main classes: MultiScaleBranchesDecomposerConfig, MultiScaleBranchesDecomposer

### `decomposers/ML_methods/NN_based/models_shallow/multiscale_branches/trainAttentionStemBilinearFusionMultipleHeadMultiScaleBranches.py`
- Purpose: Training entry point for AttentionStemBilinearFusionMultipleHeadMultiScaleBranches.
- Main classes: TrainConfig
- Main functions: parse_args, create_model, main

### `decomposers/ML_methods/NN_based/models_shallow/multiscale_branches/trainAttentionStemMultiHeadMultiScaleTCN.py`
- Purpose: Training entry point for AttentionStemMultiHeadMultiScaleTCN.
- Main classes: TrainConfig
- Main functions: parse_args, create_model, main

### `decomposers/ML_methods/NN_based/models_shallow/multiscale_branches/trainAttentionStemMultiHeadMultiScaleTCNInferenceOptimized.py`
- Purpose: Training entry point for AttentionStemMultiHeadMultiScaleTCNInferenceOptimized.
- Main classes: TrainConfig
- Main functions: parse_args, create_model, main

### `decomposers/ML_methods/NN_based/models_shallow/multiscale_branches/trainAttentionStemMultipleHeadMultiScaleBranches.py`
- Purpose: Training entry point for AttentionStemMultipleHeadMultiScaleBranches.
- Main classes: TrainConfig
- Main functions: parse_args, create_model, main

### `decomposers/ML_methods/NN_based/models_shallow/multiscale_branches/trainMultipleHeadMultiScaleBranches.py`
- Purpose: Training entry point for MultipleHeadMultiScaleBranches.
- Main classes: TrainConfig
- Main functions: parse_args, create_model, main

### `decomposers/ML_methods/NN_based/models_shallow/multiscale_branches/trainMultiscaleBranches.py`
- Purpose: Training entry point for MultiscaleBranches.
- Main classes: TrainConfig
- Main functions: parse_args, create_model, main

### `decomposers/ML_methods/NN_based/models_shallow/multiscale_dilated/__init__.py`
- Purpose: Package initializer for this module group.

### `decomposers/ML_methods/NN_based/models_shallow/multiscale_dilated/multiscale_dilated_decomposer.py`
- Purpose: Implements ShallowMultiScaleDilatedDecomposerConfig, ShallowMultiScaleDilatedDecomposer.
- Main classes: ShallowMultiScaleDilatedDecomposerConfig, ShallowMultiScaleDilatedDecomposer

### `decomposers/ML_methods/NN_based/models_shallow/multiscale_dilated/trainMultiscaleDilated.py`
- Purpose: Training entry point for MultiscaleDilated.
- Main classes: TrainConfig
- Main functions: parse_args, create_model, main

### `decomposers/ML_methods/NN_based/run_train.py`
- Purpose: Command-line dispatcher for launching a cataloged training script by model key.
- Main functions: main

### `decomposers/ML_methods/NN_based/test/test_rnn_autoencoder_models.py`
- Purpose: Automated tests or smoke tests for the related model/service behavior.
- Main functions: test_rnn_decomposer_forward_shape_and_reconstruction, test_gru_variant_forward_shape, test_autoencoder_decomposer_forward_shape_and_reconstruction

### `decomposers/ML_methods/NN_based/test/test_shallow_models.py`
- Purpose: Automated tests or smoke tests for the related model/service behavior.
- Main functions: test_shallow_conv1d_forward_and_reconstruction_num_layers_1, test_shallow_conv1d_forward_and_reconstruction_num_layers_2, test_shallow_conv1d_depth_expansion_preserves_shape_and_backward, test_single_head_mlp_forward_and_reconstruction, test_model_registry_supports_shallow_models, test_fuse_forward_preserves_length_and_reduces_channels, test_fuse_rejects_wrong_channel_count, test_multiscale_branches_with_fusion_forward_and_reconstruction, test_multiscale_branches_depth_expansion_preserves_shape, test_multiple_head_multiscale_branches_forward_and_reconstruction, test_attention_stem_multiple_head_multiscale_branches_forward_attention_and_reconstruction, test_bilinear_fusion_forward_shape

### `decomposers/ML_methods/NN_based/test/test_tasnet_sepformer_models.py`
- Purpose: Automated tests or smoke tests for the related model/service behavior.
- Main functions: test_tasnet_forward_shape_and_reconstruction, test_sepformer_forward_shape_and_reconstruction, test_model_registry_supports_tasnet_and_sepformer

### `decomposers/ML_methods/NN_based/test/test_unet_on_generators.py`
- Purpose: Automated tests or smoke tests for the related model/service behavior.
- Main classes: SyntheticDecompositionDataset
- Main functions: train_one_epoch, evaluate, plot_example, main

### `decomposers/ML_methods/NN_based/training_common.py`
- Purpose: Shared training/evaluation loops, losses, metrics, checkpoint payloads, and data loaders.
- Main classes: DecompositionLoss
- Main functions: build_decomposition_loss, component_names_from_env, set_seed, build_loaders, move_batch, compute_loss, evaluate_loss, evaluate_decomposition_loader, infer_single_batch_metrics, train_one_epoch, save_json, checkpoint_payload


## Evaluation

### `evaluation/__init__.py`
- Purpose: Evaluation package.

### `evaluation/chbmit_eeg_test.py`
- Purpose: chbmit_eeg_test.py ================== Tests trained decomposition models on real EEG data from the CHB-MIT Scalp EEG Corpus (https://physionet.org/content/chbmit/1.0.0/). CHB-MIT records at exactly 256 Hz with a bipolar ...
- Main functions: load_edf_channel, extract_windows, build_semi_synthetic_batch, run_semi_synthetic_evaluation, run_alpha_sweep, run_direct_eeg, save_distribution_figure, parse_args, main

### `evaluation/decomposition.py`
- Purpose: Classical decomposition helpers and abstractions used by evaluation demos.
- Main classes: DecompositionEvaluationError
- Main functions: find_best_permutation, reorder_prediction, evaluate_decomposition, format_report

### `evaluation/demo.py`
- Purpose: Demo script for running decomposition examples.
- Main functions: main

### `evaluation/metrics.py`
- Purpose: Metric implementations for comparing reconstructed components and mixtures.
- Main functions: mse, rmse, mae, normalized_mse, relative_l2_error, snr_db, psnr_db, correlation_coefficient, cosine_similarity, scale_invariant_sdr_db, spectral_convergence, log_spectral_distance

### `evaluation/model_eval/10fuseeval.py`
- Purpose: Model-specific evaluation entry point for fuse.
- Main functions: main

### `evaluation/model_eval/11multiscaledilatedeval.py`
- Purpose: Model-specific evaluation entry point for multiscaledilated.
- Main functions: main

### `evaluation/model_eval/1uneteval.py`
- Purpose: Model-specific evaluation entry point for unet.
- Main functions: main

### `evaluation/model_eval/2rnneval.py`
- Purpose: Model-specific evaluation entry point for rnn.
- Main functions: main

### `evaluation/model_eval/3autoencodereval.py`
- Purpose: Model-specific evaluation entry point for autoencoder.
- Main functions: main

### `evaluation/model_eval/4conv1dshalloweval.py`
- Purpose: Model-specific evaluation entry point for conv1dshallow.
- Main functions: main

### `evaluation/model_eval/5mlpsingleheadeval.py`
- Purpose: Model-specific evaluation entry point for mlpsinglehead.
- Main functions: main

### `evaluation/model_eval/6compare_all_evals.py`
- Purpose: Aggregates all model evaluation reports into a markdown table and comparison plots.
- Main functions: maybe_run_eval, load_report, format_table, format_model_label, save_bar_comparison, main

### `evaluation/model_eval/7tasneteval.py`
- Purpose: Model-specific evaluation entry point for tasnet.
- Main functions: main

### `evaluation/model_eval/8sepformereval.py`
- Purpose: Model-specific evaluation entry point for sepformer.
- Main functions: main

### `evaluation/model_eval/9threeparallelshallowbrancheseval.py`
- Purpose: Model-specific evaluation entry point for threeparallelshallowbranches.
- Main functions: main

### `evaluation/model_eval/eval_common.py`
- Purpose: Shared model-evaluation utilities for loading checkpoints, datasets, metrics, and reports.
- Main functions: evaluate_experiment

### `evaluation/model_eval/eval_template.py`
- Purpose: Template for adding a new model-specific evaluation script.
- Main functions: find_candidate_checkpoints, resolve_checkpoint, mse, rmse, mae, nmse, relative_l2, max_abs_error, correlation, cosine_similarity, explained_variance, snr_db

### `evaluation/model_eval/run_eval.py`
- Purpose: Command-line dispatcher for launching a cataloged evaluation script by model key.
- Main functions: main

### `evaluation/ui.py`
- Purpose: Evaluation UI helpers for interactive demos.


## Signal Generators

### `generators/__init__.py`
- Purpose: Signal generator package.

### `generators/basic/__init__.py`
- Purpose: Basic signal generator package.

### `generators/basic/basicSignalGenerator.py`
- Purpose: Basic synthetic signal generator for harmonic, chirp, AM/FM, and mixed components.
- Main classes: basicSignalGenerator

### `generators/basic/test.py`
- Purpose: Small generator smoke-test/demo script.


## Repository Root

### `main.py`
- Purpose: Support module for main.

### `test1.py`
- Purpose: Standalone exploratory plotting/test script kept at repository root.
