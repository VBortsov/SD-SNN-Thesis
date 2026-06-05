from .conv1Dnetwork import ShallowConv1DDecomposer, ShallowConv1DDecomposerConfig
from .mlp import SingleHeadMLPDecomposer, SingleHeadMLPDecomposerConfig
from .common import ThreeParallelShallowBranches
from .fuse import Fuse
from .multiscale_branches import (
    AttentionStemBilinearFusionMultipleHeadMultiScaleBranchesDecomposer,
    AttentionStemBilinearFusionMultipleHeadMultiScaleBranchesDecomposerConfig,
    AttentionStemMultiHeadMultiScaleTCNDecomposer,
    AttentionStemMultiHeadMultiScaleTCNDecomposerConfig,
    AttentionStemMultipleHeadMultiScaleBranchesDecomposer,
    AttentionStemMultipleHeadMultiScaleBranchesDecomposerConfig,
    BilinearFusion,
    MultiScaleBranchesDecomposer,
    MultiScaleBranchesDecomposerConfig,
    MultipleHeadMultiScaleBranchesDecomposer,
    MultipleHeadMultiScaleBranchesDecomposerConfig,
)
from .multiscale_dilated import ShallowMultiScaleDilatedDecomposer, ShallowMultiScaleDilatedDecomposerConfig

__all__ = [
    "ShallowConv1DDecomposerConfig",
    "ShallowConv1DDecomposer",
    "SingleHeadMLPDecomposerConfig",
    "SingleHeadMLPDecomposer",
    "ThreeParallelShallowBranches",
    "Fuse",
    "MultiScaleBranchesDecomposerConfig",
    "MultiScaleBranchesDecomposer",
    "MultipleHeadMultiScaleBranchesDecomposerConfig",
    "MultipleHeadMultiScaleBranchesDecomposer",
    "AttentionStemMultipleHeadMultiScaleBranchesDecomposerConfig",
    "AttentionStemMultipleHeadMultiScaleBranchesDecomposer",
    "AttentionStemMultiHeadMultiScaleTCNDecomposerConfig",
    "AttentionStemMultiHeadMultiScaleTCNDecomposer",
    "AttentionStemBilinearFusionMultipleHeadMultiScaleBranchesDecomposerConfig",
    "AttentionStemBilinearFusionMultipleHeadMultiScaleBranchesDecomposer",
    "BilinearFusion",
    "ShallowMultiScaleDilatedDecomposerConfig",
    "ShallowMultiScaleDilatedDecomposer",
]
