from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn as nn

from decomposers.ML_methods.NN_based.models_shallow.common import (
    DepthExpansion1D,
    ThreeParallelShallowBranches,
)


@dataclass
class AttentionStemMultipleHeadMultiScaleBranchesDecomposerConfig:
    """Configuration for :class:`AttentionStemMultipleHeadMultiScaleBranchesDecomposer`."""

    in_channels: int = 1
    out_channels: int = 3
    stem_channels: int = 16
    branch_channels: int = 32
    fused_channels: int = 48
    kernel_sizes: tuple[int, int, int] = (5, 7, 9)
    dilations: tuple[int, int, int] = (1, 2, 4)
    norm: Literal["groupnorm", "batchnorm", "none"] = "groupnorm"
    activation: Literal["relu", "gelu", "tanh"] = "gelu"
    dropout: float = 0.1
    num_groups: int = 8
    attention_hidden_channels: int = 16
    extra_conv_layers: int = 0
    extra_conv_kernel_size: int = 3
    extra_conv_channels: Optional[int] = None
    extra_conv_dilation: int = 1
    extra_conv_activation: Literal["relu", "gelu", "tanh"] = "gelu"
    extra_conv_norm: Literal["groupnorm", "batchnorm", "none"] = "groupnorm"
    extra_conv_dropout: float = 0.1
    extra_conv_num_groups: int = 8
    extra_conv_residual: bool = False


class AttentionStemMultipleHeadMultiScaleBranchesDecomposer(nn.Module):
    """Stem and branch-attention variant of the multi-head multiscale decomposer.

    Expected input shape is ``[batch, in_channels, length]`` and output shape is
    ``[batch, out_channels, length]``. Compared with
    ``MultipleHeadMultiScaleBranchesDecomposer``, this model adds:

    * a local Conv1D stem for short-range temporal features before branching;
    * a lightweight shared branch-attention gate that adaptively weights the
      three multiscale branches per sample;
    * the same independent one-channel heads, preserving component-specific
      specialization.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 3,
        stem_channels: int = 16,
        branch_channels: int = 32,
        fused_channels: int = 48,
        kernel_sizes: tuple[int, int, int] = (5, 7, 9),
        dilations: tuple[int, int, int] = (1, 2, 4),
        norm: Literal["groupnorm", "batchnorm", "none"] = "groupnorm",
        activation: Literal["relu", "gelu", "tanh"] = "gelu",
        dropout: float = 0.1,
        num_groups: int = 8,
        attention_hidden_channels: int = 16,
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
        if out_channels <= 0:
            raise ValueError(f"out_channels must be > 0, got {out_channels}.")
        if stem_channels <= 0:
            raise ValueError(f"stem_channels must be > 0, got {stem_channels}.")
        if attention_hidden_channels <= 0:
            raise ValueError(
                f"attention_hidden_channels must be > 0, got {attention_hidden_channels}."
            )

        self.out_channels = out_channels
        self.branch_channels = branch_channels
        self.num_branches = 3

        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, stem_channels, kernel_size=3, padding=1),
            self._make_norm(norm=norm, channels=stem_channels, num_groups=num_groups, label="stem_channels"),
            self._make_activation(activation),
            nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity(),
        )
        self.features = ThreeParallelShallowBranches(
            in_channels=stem_channels,
            branch_channels=branch_channels,
            kernel_sizes=kernel_sizes,
            dilations=dilations,
            norm=norm,
            activation=activation,
            dropout=dropout,
            num_groups=num_groups,
        )
        self.branch_attention = nn.Sequential(
            nn.Linear(self.num_branches * branch_channels, attention_hidden_channels),
            self._make_activation(activation),
            nn.Linear(attention_hidden_channels, self.num_branches),
            nn.Softmax(dim=1),
        )
        self.fusion = nn.Sequential(
            nn.Conv1d(self.num_branches * branch_channels, fused_channels, kernel_size=1),
            self._make_norm(norm=norm, channels=fused_channels, num_groups=num_groups, label="fused_channels"),
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
        self.heads = nn.ModuleList(
            [nn.Conv1d(self.depth_expansion.output_channels, 1, kernel_size=1) for _ in range(out_channels)]
        )

    @staticmethod
    def _make_norm(
        norm: Literal["groupnorm", "batchnorm", "none"],
        channels: int,
        num_groups: int,
        label: str = "channels",
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
                    f"{label} ({channels}) must be divisible by num_groups ({num_groups}) "
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

    def branch_attention_weights(self, branch_features: torch.Tensor) -> torch.Tensor:
        """Return per-sample weights for the three branch feature groups."""
        if branch_features.ndim != 3:
            raise ValueError("Expected branch_features to have shape [batch, channels, length].")
        expected_channels = self.num_branches * self.branch_channels
        if branch_features.shape[1] != expected_channels:
            raise ValueError(
                f"Expected branch_features.shape[1] == {expected_channels}, got {branch_features.shape[1]}."
            )
        pooled = branch_features.mean(dim=2)
        return self.branch_attention(pooled)

    def apply_branch_attention(self, branch_features: torch.Tensor) -> torch.Tensor:
        """Split concatenated branch features, weight each branch, then re-concatenate."""
        weights = self.branch_attention_weights(branch_features)
        branches = torch.chunk(branch_features, chunks=self.num_branches, dim=1)
        weighted = [
            branch * weights[:, idx].view(-1, 1, 1)
            for idx, branch in enumerate(branches)
        ]
        return torch.cat(weighted, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the forward pass.
        
        Args:
            x: Input tensor.
        """
        if x.ndim != 3:
            raise ValueError("Expected x to have shape [batch, channels, length].")
        stemmed = self.stem(x)
        branch_features = self.features(stemmed)
        attended = self.apply_branch_attention(branch_features)
        fused = self.depth_expansion(self.fusion(attended))
        return torch.cat([head(fused) for head in self.heads], dim=1)

    @staticmethod
    def reconstruction(components: torch.Tensor) -> torch.Tensor:
        return components.sum(dim=1, keepdim=True)
