from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn


@dataclass
class SingleHeadMLPDecomposerConfig:
    """Configuration for :class:`SingleHeadMLPDecomposer`."""

    in_channels: int = 1
    out_channels: int = 3
    hidden_dim: int = 128
    activation: Literal["relu", "gelu", "tanh"] = "relu"
    dropout: float = 0.1


class SingleHeadMLPDecomposer(nn.Module):
    """Single-hidden-layer MLP baseline for decomposition.

    Input shape: ``[batch, in_channels, signal_length]``.
    Output shape: ``[batch, out_channels, signal_length]``.

    Uses a single shared head producing all components jointly.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 3,
        hidden_dim: int = 128,
        activation: Literal["relu", "gelu", "tanh"] = "relu",
        dropout: float = 0.1,
    ):
        """Initialize layers and settings."""
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.fc1 = nn.Linear(in_channels, hidden_dim)
        self.act = self._make_activation(activation)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, out_channels)

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
        """Forward pass over each time step independently."""
        if x.ndim != 3:
            raise ValueError("Expected x to have shape [batch, channels, length].")

        seq = x.transpose(1, 2)  # [B, L, Cin]
        hidden = self.fc1(seq)
        hidden = self.act(hidden)
        hidden = self.dropout(hidden)
        components = self.fc2(hidden)  # [B, L, Cout]
        return components.transpose(1, 2)

    @staticmethod
    def reconstruction(components: torch.Tensor) -> torch.Tensor:
        """Compute additive reconstruction from predicted components."""
        return components.sum(dim=1, keepdim=True)
