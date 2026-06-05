from typing import Literal

import torch
import torch.nn as nn


class ThreeParallelShallowBranches(nn.Module):
    """Three shallow parallel Conv1D branches for multi-scale features.

    This block expects an input tensor of shape ``[batch, in_channels, length]``.
    Each branch applies exactly one Conv1D block (Conv1D + optional normalization +
    activation + optional dropout) with its own kernel size and dilation.

    The sequence length is preserved in every branch, then branch outputs are
    concatenated along the channel axis.
    """

    def __init__(
        self,
        in_channels: int,
        branch_channels: int = 32,
        kernel_sizes: tuple[int, int, int] = (5, 7, 9),
        dilations: tuple[int, int, int] = (1, 2, 4),
        norm: Literal["groupnorm", "batchnorm", "none"] = "groupnorm",
        activation: Literal["relu", "gelu", "tanh"] = "gelu",
        dropout: float = 0.0,
        num_groups: int = 8,
    ):
        """Initialize layers and settings."""
        super().__init__()

        if in_channels <= 0:
            raise ValueError(f"in_channels must be > 0, got {in_channels}.")
        if branch_channels <= 0:
            raise ValueError(f"branch_channels must be > 0, got {branch_channels}.")
        if len(kernel_sizes) != 3:
            raise ValueError(f"kernel_sizes must contain 3 values, got {kernel_sizes}.")
        if len(dilations) != 3:
            raise ValueError(f"dilations must contain 3 values, got {dilations}.")
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError(f"dropout must be in [0.0, 1.0), got {dropout}.")

        self.in_channels = in_channels
        self.branch_channels = branch_channels
        self.kernel_sizes = kernel_sizes
        self.dilations = dilations

        self.branches = nn.ModuleList(
            [
                self._make_branch(
                    in_channels=in_channels,
                    out_channels=branch_channels,
                    kernel_size=kernel_sizes[i],
                    dilation=dilations[i],
                    norm=norm,
                    activation=activation,
                    dropout=dropout,
                    num_groups=num_groups,
                )
                for i in range(3)
            ]
        )

    def _make_branch(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        norm: Literal["groupnorm", "batchnorm", "none"],
        activation: Literal["relu", "gelu", "tanh"],
        dropout: float,
        num_groups: int,
    ) -> nn.Sequential:
        if kernel_size <= 0:
            raise ValueError(f"kernel_size must be > 0, got {kernel_size}.")
        if dilation <= 0:
            raise ValueError(f"dilation must be > 0, got {dilation}.")

        # Exact same-length output for stride=1 with symmetric padding requires
        # even total receptive extension: dilation * (kernel_size - 1) must be even.
        effective_extension = dilation * (kernel_size - 1)
        if effective_extension % 2 != 0:
            raise ValueError(
                "To preserve sequence length, dilation * (kernel_size - 1) must be even. "
                f"Got kernel_size={kernel_size}, dilation={dilation}."
            )
        padding = effective_extension // 2

        layers: list[nn.Module] = [
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=padding,
            )
        ]

        layers.append(self._make_norm(norm=norm, channels=out_channels, num_groups=num_groups))
        layers.append(self._make_activation(activation))

        if dropout > 0.0:
            layers.append(nn.Dropout(p=dropout))

        return nn.Sequential(*layers)

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
                    f"branch_channels ({channels}) must be divisible by num_groups ({num_groups}) "
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
        """Run the three parallel branches and concatenate along channels.

        Args:
            x: Input tensor with shape ``[batch, in_channels, length]``.

        Returns:
            Concatenated multi-scale features with shape
            ``[batch, 3 * branch_channels, length]``.
        """
        if x.ndim != 3:
            raise ValueError("Expected x to have shape [batch, channels, length].")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected x.shape[1] == {self.in_channels}, got {x.shape[1]}."
            )

        outputs = [branch(x) for branch in self.branches]
        return torch.cat(outputs, dim=1)
