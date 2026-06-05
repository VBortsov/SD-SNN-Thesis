from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelExperimentSpec:
    """One training/evaluation target in the experiment catalog."""
    key: str
    model_key: str
    display_name: str
    family: str
    depth_label: str
    train_script: str | None
    eval_script: str | None
    saved_model_dir: str
    default_checkpoint: str
    report: str
    enabled: bool = True


MODEL_EXPERIMENTS: dict[str, ModelExperimentSpec] = {
    "unet1d": ModelExperimentSpec(
        key="unet1d",
        model_key="unet1d",
        display_name="UNet1D",
        family="unet",
        depth_label="deep",
        train_script="decomposers/ML_methods/NN_based/models/unet1d/trainUnet1d.py",
        eval_script="evaluation/model_eval/1uneteval.py",
        saved_model_dir="decomposers/ML_methods/NN_based/saved_models/unet1d",
        default_checkpoint="decomposers/ML_methods/NN_based/saved_models/unet1d/unet1d_best.pt",
        report="decomposers/ML_methods/NN_based/saved_models/unet1d/final_test_report.json",
    ),
    "rnnbased": ModelExperimentSpec(
        key="rnnbased",
        model_key="rnnbased",
        display_name="RNN (LSTM/GRU)",
        family="rnn",
        depth_label="deep",
        train_script="decomposers/ML_methods/NN_based/models/rnnbased/trainRnnbased.py",
        eval_script="evaluation/model_eval/2rnneval.py",
        saved_model_dir="decomposers/ML_methods/NN_based/saved_models/rnnbased",
        default_checkpoint="decomposers/ML_methods/NN_based/saved_models/rnnbased/rnnbased_best.pt",
        report="decomposers/ML_methods/NN_based/saved_models/rnnbased/final_test_report.json",
    ),
    "autoencoderbased": ModelExperimentSpec(
        key="autoencoderbased",
        model_key="autoencoderbased",
        display_name="Autoencoder",
        family="autoencoder",
        depth_label="deep",
        train_script="decomposers/ML_methods/NN_based/models/autoencoderbased/trainAutoencoderbased.py",
        eval_script="evaluation/model_eval/3autoencodereval.py",
        saved_model_dir="decomposers/ML_methods/NN_based/saved_models/autoencoderbased",
        default_checkpoint="decomposers/ML_methods/NN_based/saved_models/autoencoderbased/autoencoderbased_best.pt",
        report="decomposers/ML_methods/NN_based/saved_models/autoencoderbased/final_test_report.json",
    ),
    "conv1dnetwork": ModelExperimentSpec(
        key="conv1dnetwork",
        model_key="conv1dnetwork",
        display_name="Shallow Conv1D",
        family="conv1d",
        depth_label="shallow",
        train_script="decomposers/ML_methods/NN_based/models_shallow/conv1Dnetwork/trainConv1Dnetwork.py",
        eval_script="evaluation/model_eval/4conv1dshalloweval.py",
        saved_model_dir="decomposers/ML_methods/NN_based/saved_models/conv1Dnetwork",
        default_checkpoint="decomposers/ML_methods/NN_based/saved_models/conv1Dnetwork/conv1Dnetwork_best.pt",
        report="decomposers/ML_methods/NN_based/saved_models/conv1Dnetwork/final_test_report.json",
    ),
    "mlp_singlehead": ModelExperimentSpec(
        key="mlp_singlehead",
        model_key="mlp_singlehead",
        display_name="SingleHead MLP",
        family="mlp",
        depth_label="shallow",
        train_script="decomposers/ML_methods/NN_based/models_shallow/mlp/singlehead/trainSingleHeadMLP.py",
        eval_script="evaluation/model_eval/5mlpsingleheadeval.py",
        saved_model_dir="decomposers/ML_methods/NN_based/saved_models/mlp_singlehead",
        default_checkpoint="decomposers/ML_methods/NN_based/saved_models/mlp_singlehead/mlp_singlehead_best.pt",
        report="decomposers/ML_methods/NN_based/saved_models/mlp_singlehead/final_test_report.json",
    ),
    "tasnet": ModelExperimentSpec(
        key="tasnet",
        model_key="tasnet",
        display_name="TasNet",
        family="tasnet",
        depth_label="deep",
        train_script="decomposers/ML_methods/NN_based/models/tasnet/trainTasnet.py",
        eval_script="evaluation/model_eval/7tasneteval.py",
        saved_model_dir="decomposers/ML_methods/NN_based/saved_models/tasnet",
        default_checkpoint="decomposers/ML_methods/NN_based/saved_models/tasnet/tasnet_best.pt",
        report="decomposers/ML_methods/NN_based/saved_models/tasnet/final_test_report.json",
    ),
    "sepformer": ModelExperimentSpec(
        key="sepformer",
        model_key="sepformer",
        display_name="SepFormer",
        family="sepformer",
        depth_label="deep",
        train_script="decomposers/ML_methods/NN_based/models/sepformer/trainSepformer.py",
        eval_script="evaluation/model_eval/8sepformereval.py",
        saved_model_dir="decomposers/ML_methods/NN_based/saved_models/sepformer",
        default_checkpoint="decomposers/ML_methods/NN_based/saved_models/sepformer/sepformer_best.pt",
        report="decomposers/ML_methods/NN_based/saved_models/sepformer/final_test_report.json",
    ),
    "three_parallel_shallow_branches": ModelExperimentSpec(
        key="three_parallel_shallow_branches",
        model_key="three_parallel_shallow_branches",
        display_name="Three Parallel Shallow Branches",
        family="conv1d",
        depth_label="shallow",
        train_script=None,
        eval_script="evaluation/model_eval/9threeparallelshallowbrancheseval.py",
        saved_model_dir="decomposers/ML_methods/NN_based/saved_models/three_parallel_shallow_branches",
        default_checkpoint="decomposers/ML_methods/NN_based/saved_models/three_parallel_shallow_branches/three_parallel_shallow_branches_best.pt",
        report="decomposers/ML_methods/NN_based/saved_models/three_parallel_shallow_branches/final_test_report.json",
        enabled=False,
    ),
    "fuse_shallow": ModelExperimentSpec(
        key="fuse_shallow",
        model_key="fuse_shallow",
        display_name="Fuse Shallow",
        family="conv1d",
        depth_label="shallow",
        train_script="decomposers/ML_methods/NN_based/models_shallow/fuse/trainFuse.py",
        eval_script="evaluation/model_eval/10fuseeval.py",
        saved_model_dir="decomposers/ML_methods/NN_based/saved_models/fuse_shallow",
        default_checkpoint="decomposers/ML_methods/NN_based/saved_models/fuse_shallow/fuse_shallow_best.pt",
        report="decomposers/ML_methods/NN_based/saved_models/fuse_shallow/final_test_report.json",
    ),
    "multiscale_dilated": ModelExperimentSpec(
        key="multiscale_dilated",
        model_key="multiscale_dilated",
        display_name="Multiscale Dilated",
        family="conv1d",
        depth_label="shallow",
        train_script="decomposers/ML_methods/NN_based/models_shallow/multiscale_dilated/trainMultiscaleDilated.py",
        eval_script="evaluation/model_eval/11multiscaledilatedeval.py",
        saved_model_dir="decomposers/ML_methods/NN_based/saved_models/multiscale_dilated",
        default_checkpoint="decomposers/ML_methods/NN_based/saved_models/multiscale_dilated/multiscale_dilated_best.pt",
        report="decomposers/ML_methods/NN_based/saved_models/multiscale_dilated/final_test_report.json",
    ),
    "multiscale_branches": ModelExperimentSpec(
        key="multiscale_branches",
        model_key="multiscale_branches",
        display_name="Multiscale Branches",
        family="conv1d",
        depth_label="shallow",
        train_script="decomposers/ML_methods/NN_based/models_shallow/multiscale_branches/trainMultiscaleBranches.py",
        eval_script=None,
        saved_model_dir="decomposers/ML_methods/NN_based/saved_models/multiscale_branches",
        default_checkpoint="decomposers/ML_methods/NN_based/saved_models/multiscale_branches/multiscale_branches_best.pt",
        report="decomposers/ML_methods/NN_based/saved_models/multiscale_branches/final_test_report.json",
    ),
    "multiple_head_multiscale_branches": ModelExperimentSpec(
        key="multiple_head_multiscale_branches",
        model_key="multiple_head_multiscale_branches",
        display_name="Multiple-Head Multiscale Branches",
        family="conv1d",
        depth_label="shallow",
        train_script="decomposers/ML_methods/NN_based/models_shallow/multiscale_branches/trainMultipleHeadMultiScaleBranches.py",
        eval_script=None,
        saved_model_dir="decomposers/ML_methods/NN_based/saved_models/multiple_head_multiscale_branches",
        default_checkpoint="decomposers/ML_methods/NN_based/saved_models/multiple_head_multiscale_branches/multiple_head_multiscale_branches_best.pt",
        report="decomposers/ML_methods/NN_based/saved_models/multiple_head_multiscale_branches/final_test_report.json",
    ),
    "attention_stem_multiple_head_multiscale_branches": ModelExperimentSpec(
        key="attention_stem_multiple_head_multiscale_branches",
        model_key="attention_stem_multiple_head_multiscale_branches",
        display_name="Attention-Stem Multi-Head Multiscale Branches",
        family="conv1d",
        depth_label="shallow",
        train_script="decomposers/ML_methods/NN_based/models_shallow/multiscale_branches/trainAttentionStemMultipleHeadMultiScaleBranches.py",
        eval_script=None,
        saved_model_dir="decomposers/ML_methods/NN_based/saved_models/attention_stem_multiple_head_multiscale_branches",
        default_checkpoint="decomposers/ML_methods/NN_based/saved_models/attention_stem_multiple_head_multiscale_branches/attention_stem_multiple_head_multiscale_branches_best.pt",
        report="decomposers/ML_methods/NN_based/saved_models/attention_stem_multiple_head_multiscale_branches/final_test_report.json",
    ),
    "attention_stem_bilinear_fusion_multiple_head_multiscale_branches": ModelExperimentSpec(
        key="attention_stem_bilinear_fusion_multiple_head_multiscale_branches",
        model_key="attention_stem_bilinear_fusion_multiple_head_multiscale_branches",
        display_name="Attention-Stem Bilinear-Fusion Multi-Head Multiscale Branches",
        family="conv1d",
        depth_label="shallow",
        train_script="decomposers/ML_methods/NN_based/models_shallow/multiscale_branches/trainAttentionStemBilinearFusionMultipleHeadMultiScaleBranches.py",
        eval_script=None,
        saved_model_dir="decomposers/ML_methods/NN_based/saved_models/attention_stem_bilinear_fusion_multiple_head_multiscale_branches",
        default_checkpoint="decomposers/ML_methods/NN_based/saved_models/attention_stem_bilinear_fusion_multiple_head_multiscale_branches/attention_stem_bilinear_fusion_multiple_head_multiscale_branches_best.pt",
        report="decomposers/ML_methods/NN_based/saved_models/attention_stem_bilinear_fusion_multiple_head_multiscale_branches/final_test_report.json",
    ),
    "attention_stem_multi_head_multiscale_tcn": ModelExperimentSpec(
        key="attention_stem_multi_head_multiscale_tcn",
        model_key="attention_stem_multi_head_multiscale_tcn",
        display_name="Attention-Stem Multi-Head Multiscale TCN",
        family="conv1d",
        depth_label="shallow",
        train_script="decomposers/ML_methods/NN_based/models_shallow/multiscale_branches/trainAttentionStemMultiHeadMultiScaleTCN.py",
        eval_script=None,
        saved_model_dir="decomposers/ML_methods/NN_based/saved_models/attention_stem_multi_head_multiscale_tcn",
        default_checkpoint="decomposers/ML_methods/NN_based/saved_models/attention_stem_multi_head_multiscale_tcn/attention_stem_multi_head_multiscale_tcn_best.pt",
        report="decomposers/ML_methods/NN_based/saved_models/attention_stem_multi_head_multiscale_tcn/final_test_report.json",
    ),
    "attention_stem_multi_head_multiscale_tcn_inference_optimized": ModelExperimentSpec(
        key="attention_stem_multi_head_multiscale_tcn_inference_optimized",
        model_key="attention_stem_multi_head_multiscale_tcn_inference_optimized",
        display_name="Attention-Stem Multi-Head Multiscale TCN (Inference Optimized)",
        family="conv1d",
        depth_label="shallow",
        train_script="decomposers/ML_methods/NN_based/models_shallow/multiscale_branches/trainAttentionStemMultiHeadMultiScaleTCNInferenceOptimized.py",
        eval_script=None,
        saved_model_dir="decomposers/ML_methods/NN_based/saved_models/attention_stem_multi_head_multiscale_tcn_inference_optimized",
        default_checkpoint="decomposers/ML_methods/NN_based/saved_models/attention_stem_multi_head_multiscale_tcn_inference_optimized/attention_stem_multi_head_multiscale_tcn_inference_optimized_best.pt",
        report="decomposers/ML_methods/NN_based/saved_models/attention_stem_multi_head_multiscale_tcn_inference_optimized/final_test_report.json",
    ),
}


def get_experiment_spec(model_key: str) -> ModelExperimentSpec | None:
    """Get experiment spec.
    
    Args:
        model_key: Registry key for the model.
    """
    return MODEL_EXPERIMENTS.get(model_key)


def training_specs() -> list[ModelExperimentSpec]:
    """Training specs."""
    return [spec for spec in MODEL_EXPERIMENTS.values() if spec.train_script]


def evaluation_specs() -> list[ModelExperimentSpec]:
    """Evaluation specs."""
    return [spec for spec in MODEL_EXPERIMENTS.values() if spec.eval_script]
