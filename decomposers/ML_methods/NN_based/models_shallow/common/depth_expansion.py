from __future__ import annotations

import math
from typing import Literal, Optional

import torch
import torch.nn as nn


ActivationName = Literal["relu", "gelu", "tanh"]
NormName = Literal["groupnorm", "batchnorm", "none"]


def make_activation(name: ActivationName) -> nn.Module:
    """Make activation.
    
    Args:
        name: Project value for this call.
    """
    key = name.lower()
    if key == "relu":
        return nn.ReLU(inplace=True)
    if key == "gelu":
        return nn.GELU()
    if key == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation '{name}'.")


def make_norm(
    norm: NormName,
    channels: int,
    num_groups: int,
    *,
    label: str = "channels",
) -> nn.Module:
    """Make norm."""
    key = norm.lower()
    if key == "none":
        return nn.Identity()
    if key == "batchnorm":
        return nn.BatchNorm1d(channels)
    if key == "groupnorm":
        if num_groups <= 0:
            raise ValueError(f"num_groups must be > 0, got {num_groups}.")
        groups = math.gcd(channels, num_groups) or 1
        return nn.GroupNorm(num_groups=groups, num_channels=channels)
    raise ValueError(f"Unsupported norm '{norm}'.")


class DepthExpansionBlock(nn.Module):
    """Single same-length Conv1D refinement block with optional residual path."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        dilation: int,
        activation: ActivationName,
        norm: NormName,
        dropout: float,
        num_groups: int,
        residual: bool,
    ):
        """Initialize layers and settings."""
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be a positive odd integer, got {kernel_size}.")
        if dilation <= 0:
            raise ValueError(f"dilation must be > 0, got {dilation}.")
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError(f"dropout must be in [0.0, 1.0), got {dropout}.")

        padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
            bias=(norm == "none"),
        )
        self.norm = make_norm(norm, out_channels, num_groups, label="extra_conv_channels")
        self.activation = make_activation(activation)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
        self.residual = residual
        self.residual_projection = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv1d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the forward pass.
        
        Args:
            x: Input tensor.
        """
        out = self.dropout(self.activation(self.norm(self.conv(x))))
        if self.residual:
            out = out + self.residual_projection(x)
        return out


class DepthExpansion1D(nn.Module):
    """Reusable optional depth expansion stack for shallow Conv1D feature maps."""

    def __init__(
        self,
        in_channels: int,
        *,
        extra_conv_layers: int = 0,
        extra_conv_channels: Optional[int] = None,
        kernel_size: int = 3,
        dilation: int = 1,
        activation: ActivationName = "gelu",
        norm: NormName = "groupnorm",
        dropout: float = 0.0,
        num_groups: int = 8,
        residual: bool = False,
    ):
        """Initialize layers and settings."""
        super().__init__()
        if extra_conv_layers < 0:
            raise ValueError(f"extra_conv_layers must be >= 0, got {extra_conv_layers}.")
        if extra_conv_channels is not None and extra_conv_channels <= 0:
            raise ValueError(f"extra_conv_channels must be > 0, got {extra_conv_channels}.")

        self.output_channels = extra_conv_channels or in_channels
        blocks: list[nn.Module] = []
        current_channels = in_channels
        for _ in range(extra_conv_layers):
            blocks.append(
                DepthExpansionBlock(
                    in_channels=current_channels,
                    out_channels=self.output_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    activation=activation,
                    norm=norm,
                    dropout=dropout,
                    num_groups=num_groups,
                    residual=residual,
                )
            )
            current_channels = self.output_channels
        self.layers = nn.Sequential(*blocks) if blocks else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the forward pass.
        
        Args:
            x: Input tensor.
        """
        return self.layers(x)
