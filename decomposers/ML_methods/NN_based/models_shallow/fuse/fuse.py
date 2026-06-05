from typing import Literal

import torch
import torch.nn as nn


class Fuse(nn.Module):
    """Shallow channel-fusion block for multi-scale branch features.

    This module expects concatenated branch features of shape
    ``[batch, in_channels, length]`` and projects them to a smaller shared
    representation ``[batch, out_channels, length]`` using a 1x1 Conv1D.

    The temporal axis is preserved exactly (no stride/pooling), and optional
    normalization, activation, and dropout can be applied after fusion.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 48,
        norm: Literal["groupnorm", "batchnorm", "none"] = "groupnorm",
        activation: Literal["relu", "gelu", "tanh"] = "gelu",
        dropout: float = 0.0,
        num_groups: int = 8,
    ):
        """Initialize layers and settings."""
        super().__init__()

        if in_channels <= 0:
            raise ValueError(f"in_channels must be > 0, got {in_channels}.")
        if out_channels <= 0:
            raise ValueError(f"out_channels must be > 0, got {out_channels}.")
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError(f"dropout must be in [0.0, 1.0), got {dropout}.")

        self.in_channels = in_channels
        self.out_channels = out_channels

        layers: list[nn.Module] = [
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            self._make_norm(norm=norm, channels=out_channels, num_groups=num_groups),
            self._make_activation(activation),
        ]

        if dropout > 0.0:
            layers.append(nn.Dropout(p=dropout))

        self.fuse = nn.Sequential(*layers)

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
                    f"out_channels ({channels}) must be divisible by num_groups ({num_groups}) "
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
        """Fuse concatenated branch features.

        Args:
            x: Tensor shaped ``[batch, in_channels, length]``.

        Returns:
            Tensor shaped ``[batch, out_channels, length]``.
        """
        if x.ndim != 3:
            raise ValueError("Expected x to have shape [batch, channels, length].")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected x.shape[1] == {self.in_channels}, got {x.shape[1]}."
            )

        return self.fuse(x)
