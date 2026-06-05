from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn as nn

from decomposers.ML_methods.NN_based.models_shallow.common import DepthExpansion1D


@dataclass
class ShallowConv1DDecomposerConfig:
    """Configuration for :class:`ShallowConv1DDecomposer`."""

    in_channels: int = 1
    out_channels: int = 3
    hidden_channels: int = 32
    kernel_size: int = 5
    num_layers: int = 2
    activation: Literal["relu", "gelu", "tanh"] = "relu"
    dropout: float = 0.1
    extra_conv_layers: int = 0
    extra_conv_kernel_size: int = 3
    extra_conv_channels: Optional[int] = None
    extra_conv_dilation: int = 1
    extra_conv_activation: Literal["relu", "gelu", "tanh"] = "relu"
    extra_conv_norm: Literal["groupnorm", "batchnorm", "none"] = "none"
    extra_conv_dropout: float = 0.1
    extra_conv_num_groups: int = 8
    extra_conv_residual: bool = False


class ShallowConv1DDecomposer(nn.Module):
    """Shallow Conv1D baseline for 1D signal decomposition.

    Expects input of shape ``[batch, in_channels, signal_length]`` and returns
    components of shape ``[batch, out_channels, signal_length]``.

    Additive reconstruction is available via
    ``components.sum(dim=1, keepdim=True)``.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 3,
        hidden_channels: int = 32,
        kernel_size: int = 5,
        num_layers: int = 2,
        activation: Literal["relu", "gelu", "tanh"] = "relu",
        dropout: float = 0.1,
        extra_conv_layers: int = 0,
        extra_conv_kernel_size: int = 3,
        extra_conv_channels: Optional[int] = None,
        extra_conv_dilation: int = 1,
        extra_conv_activation: Literal["relu", "gelu", "tanh"] = "relu",
        extra_conv_norm: Literal["groupnorm", "batchnorm", "none"] = "none",
        extra_conv_dropout: float = 0.1,
        extra_conv_num_groups: int = 8,
        extra_conv_residual: bool = False,
    ):
        """Initialize layers and settings."""
        super().__init__()

        if num_layers not in {1, 2}:
            raise ValueError(f"num_layers must be 1 or 2, got {num_layers}.")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.extra_conv_layers = extra_conv_layers

        act = self._make_activation(activation)
        padding = kernel_size // 2

        layers: list[nn.Module] = [
            nn.Conv1d(in_channels, hidden_channels, kernel_size=kernel_size, padding=padding),
            act,
            nn.Dropout(dropout),
        ]

        if num_layers == 2:
            layers.extend(
                [
                    nn.Conv1d(hidden_channels, hidden_channels, kernel_size=kernel_size, padding=padding),
                    self._make_activation(activation),
                    nn.Dropout(dropout),
                ]
            )

        self.features = nn.Sequential(*layers)
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
        """Forward pass.

        Args:
            x: Input tensor shaped ``[batch, in_channels, signal_length]``.

        Returns:
            Tensor shaped ``[batch, out_channels, signal_length]``.
        """
        if x.ndim != 3:
            raise ValueError("Expected x to have shape [batch, channels, length].")

        components = self.depth_expansion(self.features(x))
        return self.head(components)

    @staticmethod
    def reconstruction(components: torch.Tensor) -> torch.Tensor:
        """Compute additive reconstruction from predicted components."""
        return components.sum(dim=1, keepdim=True)
