from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn as nn

from decomposers.ML_methods.NN_based.models_shallow.common import (
    DepthExpansion1D,
    ThreeParallelShallowBranches,
)


@dataclass
class MultiScaleBranchesDecomposerConfig:
    """Configuration for :class:`MultiScaleBranchesDecomposer`."""

    in_channels: int = 1
    out_channels: int = 3
    branch_channels: int = 32
    fused_channels: int = 48
    kernel_sizes: tuple[int, int, int] = (5, 7, 9)
    dilations: tuple[int, int, int] = (1, 2, 4)
    norm: Literal["groupnorm", "batchnorm", "none"] = "groupnorm"
    activation: Literal["relu", "gelu", "tanh"] = "gelu"
    dropout: float = 0.1
    num_groups: int = 8
    extra_conv_layers: int = 0
    extra_conv_kernel_size: int = 3
    extra_conv_channels: Optional[int] = None
    extra_conv_dilation: int = 1
    extra_conv_activation: Literal["relu", "gelu", "tanh"] = "gelu"
    extra_conv_norm: Literal["groupnorm", "batchnorm", "none"] = "groupnorm"
    extra_conv_dropout: float = 0.1
    extra_conv_num_groups: int = 8
    extra_conv_residual: bool = False


class MultiScaleBranchesDecomposer(nn.Module):
    """Shallow decomposition model built from three multi-scale Conv1D branches."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 3,
        branch_channels: int = 32,
        fused_channels: int = 48,
        kernel_sizes: tuple[int, int, int] = (5, 7, 9),
        dilations: tuple[int, int, int] = (1, 2, 4),
        norm: Literal["groupnorm", "batchnorm", "none"] = "groupnorm",
        activation: Literal["relu", "gelu", "tanh"] = "gelu",
        dropout: float = 0.1,
        num_groups: int = 8,
        extra_conv_layers: int = 0,
        extra_conv_kernel_size: int = 3,
        extra_conv_channels: Optional[int] = None,
        extra_conv_dilation: int = 1,
        extra_conv_activation: Literal["relu", "gelu", "tanh"] = "gelu",
        extra_conv_norm: Literal["groupnorm", "batchnorm", "none"] = "groupnorm",
        extra_conv_dropout: float = 0.1,
        extra_conv_num_groups: int = 8,
        extra_conv_residual: bool = False,
    ):
        """Initialize layers and settings."""
        super().__init__()

        self.features = ThreeParallelShallowBranches(
            in_channels=in_channels,
            branch_channels=branch_channels,
            kernel_sizes=kernel_sizes,
            dilations=dilations,
            norm=norm,
            activation=activation,
            dropout=dropout,
            num_groups=num_groups,
        )
        self.fusion = nn.Sequential(
            nn.Conv1d(3 * branch_channels, fused_channels, kernel_size=1),
            self._make_norm(norm=norm, channels=fused_channels, num_groups=num_groups),
            self._make_activation(activation),
        )
        self.depth_expansion = DepthExpansion1D(
            in_channels=fused_channels,
            extra_conv_layers=extra_conv_layers,
            extra_conv_channels=extra_conv_channels,
            kernel_size=extra_conv_kernel_size,
            dilation=extra_conv_dilation,
            activation=extra_conv_activation,
            norm=extra_conv_norm,
            dropout=extra_conv_dropout,
            num_groups=extra_conv_num_groups,
            residual=extra_conv_residual,
        )
        self.head = nn.Conv1d(self.depth_expansion.output_channels, out_channels, kernel_size=1)

    @staticmethod
    def _make_norm(
        norm: Literal["groupnorm", "batchnorm", "none"],
        channels: int,
        num_groups: int,
    ) -> nn.Module:
        key = norm.lower()
        if key == "none":
            return nn.Identity()
        if key == "batchnorm":
            return nn.BatchNorm1d(channels)
        if key == "groupnorm":
            if num_groups <= 0:
                raise ValueError(f"num_groups must be > 0, got {num_groups}.")
            if channels % num_groups != 0:
                raise ValueError(
                    f"fused_channels ({channels}) must be divisible by num_groups ({num_groups}) "
                    "when norm='groupnorm'."
                )
            return nn.GroupNorm(num_groups=num_groups, num_channels=channels)
        raise ValueError(f"Unsupported norm '{norm}'.")

    @staticmethod
    def _make_activation(name: Literal["relu", "gelu", "tanh"]) -> nn.Module:
        key = name.lower()
        if key == "relu":
            return nn.ReLU(inplace=True)
        if key == "gelu":
            return nn.GELU()
        if key == "tanh":
            return nn.Tanh()
        raise ValueError(f"Unsupported activation '{name}'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the forward pass.
        
        Args:
            x: Input tensor.
        """
        if x.ndim != 3:
            raise ValueError("Expected x to have shape [batch, channels, length].")
        return self.head(self.depth_expansion(self.fusion(self.features(x))))

    @staticmethod
    def reconstruction(components: torch.Tensor) -> torch.Tensor:
        return components.sum(dim=1, keepdim=True)
