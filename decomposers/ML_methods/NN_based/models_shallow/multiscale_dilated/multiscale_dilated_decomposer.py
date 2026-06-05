from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn as nn

from decomposers.ML_methods.NN_based.models_shallow.common import DepthExpansion1D


@dataclass
class ShallowMultiScaleDilatedDecomposerConfig:
    """Configuration for :class:`ShallowMultiScaleDilatedDecomposer`."""

    in_channels: int = 1
    out_channels: int = 3
    hidden_channels: int = 32
    stem_kernel_size: int = 9
    activation: Literal["relu", "gelu", "tanh"] = "gelu"
    dropout: float = 0.1
    extra_conv_layers: int = 0
    extra_conv_kernel_size: int = 3
    extra_conv_channels: Optional[int] = None
    extra_conv_dilation: int = 1
    extra_conv_activation: Literal["relu", "gelu", "tanh"] = "gelu"
    extra_conv_norm: Literal["groupnorm", "batchnorm", "none"] = "groupnorm"
    extra_conv_dropout: float = 0.1
    extra_conv_num_groups: int = 8
    extra_conv_residual: bool = False


class ShallowMultiScaleDilatedDecomposer(nn.Module):
    """Shallow multi-scale 1D decomposer with a stronger local stem stage."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 3,
        hidden_channels: int = 32,
        stem_kernel_size: int = 9,
        activation: Literal["relu", "gelu", "tanh"] = "gelu",
        dropout: float = 0.1,
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

        if stem_kernel_size <= 0 or stem_kernel_size % 2 == 0:
            raise ValueError(
                f"stem_kernel_size must be a positive odd integer, got {stem_kernel_size}."
            )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.stem_kernel_size = stem_kernel_size
        self.activation = activation
        self.dropout = dropout

        self.stem = self._build_stem()

        self.branches = nn.ModuleList(
            [
                nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, dilation=1, padding=1),
                nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, dilation=2, padding=2),
                nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, dilation=4, padding=4),
            ]
        )
        self.branch_fuse = nn.Conv1d(3 * hidden_channels, hidden_channels, kernel_size=1)
        self.depth_expansion = DepthExpansion1D(
            in_channels=hidden_channels,
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

    def _build_stem(self) -> nn.Sequential:
        """Build the shallow local stem used before multi-scale branches."""
        num_groups = self._choose_group_count(self.hidden_channels)
        return nn.Sequential(
            nn.Conv1d(
                self.in_channels,
                self.hidden_channels,
                kernel_size=self.stem_kernel_size,
                padding=self.stem_kernel_size // 2,
                bias=False,
            ),
            nn.GroupNorm(num_groups=num_groups, num_channels=self.hidden_channels),
            self._make_activation(self.activation),
            nn.Dropout(self.dropout),
        )

    @staticmethod
    def _choose_group_count(num_channels: int) -> int:
        """Choose the largest supported GroupNorm divisor from [8, 4, 2, 1]."""
        for candidate in (8, 4, 2, 1):
            if num_channels % candidate == 0:
                return candidate
        return 1

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
        """Run stem, shallow multi-scale branches, and projection head."""
        if x.ndim != 3:
            raise ValueError("Expected x to have shape [batch, channels, length].")

        features = self.stem(x)
        branch_features = [branch(features) for branch in self.branches]
        fused = self.depth_expansion(self.branch_fuse(torch.cat(branch_features, dim=1)))
        return self.head(fused)

    @staticmethod
    def reconstruction(components: torch.Tensor) -> torch.Tensor:
        """Compute additive reconstruction from predicted components."""
        return components.sum(dim=1, keepdim=True)
