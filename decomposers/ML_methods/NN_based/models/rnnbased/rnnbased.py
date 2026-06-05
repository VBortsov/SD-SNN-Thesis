from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn


@dataclass
class RNNDecomposerConfig:
    """Configuration for :class:`RNNDecomposer`."""

    in_channels: int = 1
    out_channels: int = 3
    hidden_size: int = 128
    num_layers: int = 2
    bidirectional: bool = True
    dropout: float = 0.1
    cell_type: Literal["lstm", "gru"] = "lstm"


class RNNDecomposer(nn.Module):
    """Recurrent 1D signal decomposer.

    Expects input of shape ``[batch, in_channels, signal_length]`` and returns
    component predictions with shape ``[batch, out_channels, signal_length]``.

    The predicted reconstruction can be computed as
    ``components.sum(dim=1, keepdim=True)``.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 3,
        hidden_size: int = 128,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.1,
        cell_type: Literal["lstm", "gru"] = "lstm",
    ):
        """Initialize layers and settings."""
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.cell_type = cell_type.lower()

        effective_dropout = dropout if num_layers > 1 else 0.0
        rnn_cls = nn.LSTM if self.cell_type == "lstm" else nn.GRU
        if self.cell_type not in {"lstm", "gru"}:
            raise ValueError(f"Unsupported cell_type '{cell_type}'. Use 'lstm' or 'gru'.")

        self.encoder = rnn_cls(
            input_size=in_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=effective_dropout,
            bidirectional=bidirectional,
        )

        rnn_features = hidden_size * (2 if bidirectional else 1)
        self.feature_dropout = nn.Dropout(dropout)
        self.output_head = nn.Linear(rnn_features, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor shaped ``[batch, in_channels, signal_length]``.

        Returns:
            Tensor shaped ``[batch, out_channels, signal_length]``.
        """
        if x.ndim != 3:
            raise ValueError("Expected x to have shape [batch, channels, length].")

        seq = x.transpose(1, 2)  # [B, L, Cin]
        encoded, _ = self.encoder(seq)
        encoded = self.feature_dropout(encoded)

        components = self.output_head(encoded)  # [B, L, Cout]
        return components.transpose(1, 2)

    @staticmethod
    def reconstruction(components: torch.Tensor) -> torch.Tensor:
        """Compute additive reconstruction from predicted components."""
        return components.sum(dim=1, keepdim=True)


class LSTMDecomposer(RNNDecomposer):
    """Convenience wrapper for LSTM based decomposition."""

    def __init__(self, **kwargs):
        """Initialize layers and settings.
        
        Args:
            kwargs: Extra keyword arguments passed through.
        """
        super().__init__(cell_type="lstm", **kwargs)


class GRUDecomposer(RNNDecomposer):
    """Convenience wrapper for GRU based decomposition."""

    def __init__(self, **kwargs):
        """Initialize layers and settings.
        
        Args:
            kwargs: Extra keyword arguments passed through.
        """
        super().__init__(cell_type="gru", **kwargs)
