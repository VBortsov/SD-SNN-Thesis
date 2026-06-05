from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AutoencoderDecomposerConfig:
    """Configuration for :class:`AutoencoderDecomposer`."""

    in_channels: int = 1
    out_channels: int = 3
    hidden_channels: tuple[int, ...] = (16, 32, 64)
    kernel_size: int = 5
    dropout: float = 0.1


class AutoencoderDecomposer(nn.Module):
    """Convolutional autoencoder-based 1D decomposer.

    Input shape: ``[batch, in_channels, signal_length]``.
    Output shape: ``[batch, out_channels, signal_length]``.

    Component-wise additive reconstruction can be obtained with
    ``components.sum(dim=1, keepdim=True)``.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 3,
        hidden_channels: Sequence[int] = (16, 32, 64),
        depth: int | None = None,
        kernel_size: int = 5,
        dropout: float = 0.1,
    ):
        """Initialize layers and settings."""
        super().__init__()

        if depth is not None:
            if depth <= 0:
                raise ValueError("depth must be positive when provided.")
            if len(hidden_channels) < depth:
                raise ValueError("len(hidden_channels) must be >= depth.")
            hidden_channels = tuple(hidden_channels[:depth])

        if len(hidden_channels) == 0:
            raise ValueError("hidden_channels cannot be empty.")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = tuple(int(c) for c in hidden_channels)
        self.kernel_size = kernel_size

        padding = kernel_size // 2
        encoder_layers = []
        ch_in = in_channels
        for ch_out in self.hidden_channels:
            encoder_layers.extend(
                [
                    nn.Conv1d(ch_in, ch_out, kernel_size=kernel_size, stride=2, padding=padding),
                    nn.BatchNorm1d(ch_out),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                ]
            )
            ch_in = ch_out

        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        reverse_channels = list(self.hidden_channels[::-1])
        ch_in = reverse_channels[0]
        for ch_out in reverse_channels[1:]:
            decoder_layers.extend(
                [
                    nn.ConvTranspose1d(ch_in, ch_out, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm1d(ch_out),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                ]
            )
            ch_in = ch_out

        self.decoder = nn.Sequential(*decoder_layers)
        self.head = nn.Conv1d(ch_in, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the forward pass.
        
        Args:
            x: Input tensor.
        """
        if x.ndim != 3:
            raise ValueError("Expected x to have shape [batch, channels, length].")

        input_len = x.size(-1)
        z = self.encoder(x)
        decoded = self.decoder(z)

        if decoded.size(-1) != input_len:
            decoded = F.interpolate(decoded, size=input_len, mode="linear", align_corners=False)

        return self.head(decoded)

    @staticmethod
    def reconstruction(components: torch.Tensor) -> torch.Tensor:
        """Compute additive reconstruction from predicted components."""
        return components.sum(dim=1, keepdim=True)


class ConvAutoencoderDecomposer(AutoencoderDecomposer):
    """Alias for the default convolutional autoencoder decomposer."""

    pass
