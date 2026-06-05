from typing import Any

import torch

from decomposers.ML_methods.NN_based.models.autoencoderbased import AutoencoderDecomposer
from decomposers.ML_methods.NN_based.models.rnnbased import RNNDecomposer
from decomposers.ML_methods.NN_based.models.sepformer import SepFormerDecomposer
from decomposers.ML_methods.NN_based.models.tasnet import TasNetDecomposer
from decomposers.ML_methods.NN_based.models.unet1d import UNet1D
from decomposers.ML_methods.NN_based.models_shallow.common import ThreeParallelShallowBranches
from decomposers.ML_methods.NN_based.models_shallow.conv1Dnetwork import ShallowConv1DDecomposer
from decomposers.ML_methods.NN_based.models_shallow.fuse.trainFuse import (
    ThreeParallelWithFuseDecomposer,
    TrainConfig as FuseTrainConfig,
)
from decomposers.ML_methods.NN_based.models_shallow.mlp.singlehead import SingleHeadMLPDecomposer
from decomposers.ML_methods.NN_based.models_shallow.multiscale_branches import (
    AttentionStemBilinearFusionMultipleHeadMultiScaleBranchesDecomposer,
    AttentionStemMultiHeadMultiScaleTCNDecomposer,
    AttentionStemMultipleHeadMultiScaleBranchesDecomposer,
    MultiScaleBranchesDecomposer,
    MultipleHeadMultiScaleBranchesDecomposer,
)
from decomposers.ML_methods.NN_based.models_shallow.multiscale_dilated import ShallowMultiScaleDilatedDecomposer


class FuseShallowDecomposer(ThreeParallelWithFuseDecomposer):
    """Factory-compatible wrapper for the existing fuse training decomposer."""

    def __init__(self, in_channels: int = 1, out_channels: int = 3, **kwargs: Any):
        """Initialize layers and settings.
        
        Args:
            in_channels: Number of input channels.
            out_channels: Number of output component channels.
            kwargs: Extra keyword arguments passed through.
        """
        cfg = FuseTrainConfig(
            in_channels=in_channels,
            out_channels=out_channels,
            branch_channels=kwargs.get("branch_channels", 32),
            fused_channels=kwargs.get("fused_channels", 48),
            norm=kwargs.get("norm", "groupnorm"),
            activation=kwargs.get("activation", "gelu"),
            dropout=kwargs.get("dropout", 0.0),
            num_groups=kwargs.get("num_groups", 8),
            experiment_name=kwargs.get("experiment_name", ""),
            extra_conv_layers=kwargs.get("extra_conv_layers", 0),
            extra_conv_kernel_size=kwargs.get("extra_conv_kernel_size", 3),
            extra_conv_channels=kwargs.get("extra_conv_channels", 0),
            extra_conv_dilation=kwargs.get("extra_conv_dilation", 1),
            extra_conv_activation=kwargs.get("extra_conv_activation", "gelu"),
            extra_conv_norm=kwargs.get("extra_conv_norm", "groupnorm"),
            extra_conv_dropout=kwargs.get("extra_conv_dropout", 0.0),
            extra_conv_num_groups=kwargs.get("extra_conv_num_groups", 8),
            extra_conv_residual=kwargs.get("extra_conv_residual", False),
        )
        super().__init__(cfg=cfg)


class ThreeParallelShallowBranchDecomposer(torch.nn.Module):
    """Factory-compatible wrapper around the shared three-branch feature block."""

    def __init__(self, in_channels: int = 1, out_channels: int = 3, branch_channels: int = 32, **_: Any):
        """Initialize layers and settings."""
        super().__init__()
        self.features = ThreeParallelShallowBranches(
            in_channels=in_channels,
            branch_channels=branch_channels,
            kernel_sizes=(5, 7, 9),
            dilations=(1, 2, 4),
            norm="groupnorm",
            activation="gelu",
            dropout=0.0,
            num_groups=8,
        )
        self.head = torch.nn.Conv1d(3 * branch_channels, out_channels, kernel_size=1)

    def forward(self, x):
        """Run the forward pass.
        
        Args:
            x: Input tensor.
        """
        return self.head(self.features(x))


MODEL_REGISTRY = {
    "unet1d": UNet1D,
    "rnn": RNNDecomposer,
    "rnnbased": RNNDecomposer,
    "autoencoder": AutoencoderDecomposer,
    "autoencoderbased": AutoencoderDecomposer,
    "tasnet": TasNetDecomposer,
    "convtasnet": TasNetDecomposer,
    "sepformer": SepFormerDecomposer,
    "conv1dnetwork": ShallowConv1DDecomposer,
    "shallowconv1d": ShallowConv1DDecomposer,
    "mlp_singlehead": SingleHeadMLPDecomposer,
    "singleheadmlp": SingleHeadMLPDecomposer,
    "multiscale_dilated": ShallowMultiScaleDilatedDecomposer,
    "shallowmultiscaledilated": ShallowMultiScaleDilatedDecomposer,
    "multiscale_branches": MultiScaleBranchesDecomposer,
    "multiscalebranches": MultiScaleBranchesDecomposer,
    "multiple_head_multiscale_branches": MultipleHeadMultiScaleBranchesDecomposer,
    "multipleheadmultiscalebranches": MultipleHeadMultiScaleBranchesDecomposer,
    "attention_stem_multiple_head_multiscale_branches": AttentionStemMultipleHeadMultiScaleBranchesDecomposer,
    "attentionstemmultipleheadmultiscalebranches": AttentionStemMultipleHeadMultiScaleBranchesDecomposer,
    "attention_stem_multi_head_multiscale_tcn": AttentionStemMultiHeadMultiScaleTCNDecomposer,
    "attentionstemmultiheadmultiscaletcn": AttentionStemMultiHeadMultiScaleTCNDecomposer,
    "attentionstemmultiheadmultiscaletcndecomposer": AttentionStemMultiHeadMultiScaleTCNDecomposer,
    "attention_stem_multi_head_multiscale_tcn_decomposer": AttentionStemMultiHeadMultiScaleTCNDecomposer,
    "attention_stem_multi_head_multiscale_tcn_inference_optimized": AttentionStemMultiHeadMultiScaleTCNDecomposer,
    "attentionstemmultiheadmultiscaletcninferenceoptimized": AttentionStemMultiHeadMultiScaleTCNDecomposer,
    "attention_stem_bilinear_fusion_multiple_head_multiscale_branches": AttentionStemBilinearFusionMultipleHeadMultiScaleBranchesDecomposer,
    "attentionstembilinearfusionmultipleheadmultiscalebranches": AttentionStemBilinearFusionMultipleHeadMultiScaleBranchesDecomposer,
    "fuse_shallow": FuseShallowDecomposer,
    "three_parallel_with_fuse": FuseShallowDecomposer,
    "three_parallel_shallow_branches": ThreeParallelShallowBranchDecomposer,
}


def create_model(model_name: str, **kwargs: Any):
    """Build the model instance.
    
    Args:
        model_name: Name used for lookup or display.
        kwargs: Extra keyword arguments passed through.
    """
    key = model_name.lower()
    if key not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unknown model '{model_name}'. Available: {available}")
    return MODEL_REGISTRY[key](**kwargs)


__all__ = [
    "UNet1D",
    "RNNDecomposer",
    "AutoencoderDecomposer",
    "TasNetDecomposer",
    "SepFormerDecomposer",
    "ShallowConv1DDecomposer",
    "SingleHeadMLPDecomposer",
    "ShallowMultiScaleDilatedDecomposer",
    "MultiScaleBranchesDecomposer",
    "MultipleHeadMultiScaleBranchesDecomposer",
    "AttentionStemMultipleHeadMultiScaleBranchesDecomposer",
    "AttentionStemMultiHeadMultiScaleTCNDecomposer",
    "AttentionStemBilinearFusionMultipleHeadMultiScaleBranchesDecomposer",
    "FuseShallowDecomposer",
    "ThreeParallelShallowBranchDecomposer",
    "MODEL_REGISTRY",
    "create_model",
]
