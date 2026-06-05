from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from decomposers.ML_methods.NN_based.models_shallow.common import ThreeParallelShallowBranches


ActivationName = Literal["relu", "gelu", "tanh"]
NormName = Literal["groupnorm", "batchnorm", "none"]
FrequencyFeatureMode = Literal["none", "fft_magnitude"]


@dataclass
class AttentionStemMultiHeadMultiScaleTCNDecomposerConfig:
    """Configuration for :class:`AttentionStemMultiHeadMultiScaleTCNDecomposer`."""

    in_channels: int = 1
    out_channels: int = 3
    hidden_channels: int = 32
    branch_channels: int = 16
    fused_channels: int = 48
    tcn_channels: int = 48
    kernel_sizes: tuple[int, int, int] = (5, 7, 9)
    branch_dilations: tuple[int, int, int] = (1, 2, 4)
    tcn_dilations: tuple[int, int, int, int] = (1, 2, 4, 8)
    stem_kernel_size: int = 9
    activation: ActivationName = "gelu"
    norm: NormName = "groupnorm"
    dropout: float = 0.1
    num_groups: int = 8
    use_frequency_features: bool = False
    frequency_feature_mode: FrequencyFeatureMode = "none"
    causal: bool = False


def _validate_positive_int(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}.")


def _validate_positive_odd_int(name: str, value: int) -> None:
    if value <= 0 or value % 2 == 0:
        raise ValueError(f"{name} must be a positive odd integer, got {value}.")


def _validate_dropout(dropout: float) -> None:
    if dropout < 0.0 or dropout >= 1.0:
        raise ValueError(f"dropout must be in [0.0, 1.0), got {dropout}.")


def _resolve_group_count(channels: int, requested_groups: int) -> int:
    if requested_groups <= 0:
        raise ValueError(f"num_groups must be > 0, got {requested_groups}.")
    return math.gcd(channels, requested_groups) or 1


def _make_norm(norm: NormName, channels: int, num_groups: int) -> nn.Module:
    key = norm.lower()
    if key == "none":
        return nn.Identity()
    if key == "batchnorm":
        return nn.BatchNorm1d(channels)
    if key == "groupnorm":
        return nn.GroupNorm(
            num_groups=_resolve_group_count(channels, num_groups),
            num_channels=channels,
        )
    raise ValueError(f"Unsupported norm '{norm}'.")


def _make_activation(name: ActivationName) -> nn.Module:
    key = name.lower()
    if key == "relu":
        return nn.ReLU(inplace=True)
    if key == "gelu":
        return nn.GELU()
    if key == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation '{name}'.")


class SameLengthConv1d(nn.Module):
    """Conv1D with explicit padding so output length always matches input length."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        *,
        causal: bool = False,
        bias: bool = True,
    ):
        """Initialize layers and settings."""
        super().__init__()
        _validate_positive_int("in_channels", in_channels)
        _validate_positive_int("out_channels", out_channels)
        _validate_positive_odd_int("kernel_size", kernel_size)
        _validate_positive_int("dilation", dilation)
        self.causal = causal
        self.total_padding = dilation * (kernel_size - 1)
        symmetric_padding = self.total_padding // 2 if not causal else 0
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=symmetric_padding,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the forward pass.
        
        Args:
            x: Input tensor.
        """
        if x.ndim != 3:
            raise ValueError("Expected x to have shape [batch, channels, length].")
        if self.causal:
            return self.conv(F.pad(x, (self.total_padding, 0)))
        return self.conv(x)


class TCNResidualBlock(nn.Module):
    """Lightweight residual TCN block with two same-length dilated convolutions."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dilation: int,
        *,
        activation: ActivationName,
        norm: NormName,
        dropout: float,
        num_groups: int,
        causal: bool,
        kernel_size: int = 3,
    ):
        """Initialize layers and settings."""
        super().__init__()
        self.conv1 = SameLengthConv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            causal=causal,
            bias=False,
        )
        self.norm1 = _make_norm(norm, out_channels, num_groups)
        self.act1 = _make_activation(activation)
        self.drop1 = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        self.conv2 = SameLengthConv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            causal=causal,
            bias=False,
        )
        self.norm2 = _make_norm(norm, out_channels, num_groups)
        self.act2 = _make_activation(activation)
        self.drop2 = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        self.residual = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the forward pass.
        
        Args:
            x: Input tensor.
        """
        residual = self.residual(x)
        out = self.drop1(self.act1(self.norm1(self.conv1(x))))
        out = self.drop2(self.act2(self.norm2(self.conv2(out))))
        return out + residual


class ComponentHead(nn.Module):
    """Small component-specific output head."""

    def __init__(self, in_channels: int, activation: ActivationName, dropout: float):
        """Initialize layers and settings.
        
        Args:
            in_channels: Number of input channels.
            activation: Activation function name.
            dropout: Dropout probability.
        """
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=1),
            _make_activation(activation),
            nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity(),
            nn.Conv1d(in_channels, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the forward pass.
        
        Args:
            x: Input tensor.
        """
        return self.block(x)


class AttentionStemMultiHeadMultiScaleTCNDecomposer(nn.Module):
    """Shallow TCN-based signal decomposer with attention, multiscale branches, and per-component heads.

    The model keeps the thesis shallow-network design goal intact:
    a local attention stem, three lightweight multi-scale Conv1D branches, a
    compact 1x1 fusion stage, and a short residual TCN backbone that improves
    temporal context without growing into a deep separator.

    Input shape: ``[batch, in_channels, signal_length]``.
    Output shape: ``[batch, out_channels, signal_length]``.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 3,
        hidden_channels: int = 32,
        branch_channels: int = 16,
        fused_channels: int = 48,
        tcn_channels: int = 48,
        kernel_sizes: tuple[int, int, int] = (5, 7, 9),
        branch_dilations: tuple[int, int, int] = (1, 2, 4),
        tcn_dilations: tuple[int, int, int, int] = (1, 2, 4, 8),
        stem_kernel_size: int = 9,
        activation: ActivationName = "gelu",
        norm: NormName = "groupnorm",
        dropout: float = 0.1,
        num_groups: int = 8,
        use_frequency_features: bool = False,
        frequency_feature_mode: FrequencyFeatureMode = "none",
        causal: bool = False,
    ):
        """Initialize layers and settings."""
        super().__init__()
        _validate_positive_int("in_channels", in_channels)
        _validate_positive_int("out_channels", out_channels)
        _validate_positive_int("hidden_channels", hidden_channels)
        _validate_positive_int("branch_channels", branch_channels)
        _validate_positive_int("fused_channels", fused_channels)
        _validate_positive_int("tcn_channels", tcn_channels)
        _validate_positive_odd_int("stem_kernel_size", stem_kernel_size)
        _validate_dropout(dropout)
        if len(kernel_sizes) != 3:
            raise ValueError(f"kernel_sizes must contain 3 values, got {kernel_sizes}.")
        if len(branch_dilations) != 3:
            raise ValueError(
                f"branch_dilations must contain 3 values, got {branch_dilations}."
            )
        if len(tcn_dilations) == 0:
            raise ValueError("tcn_dilations must contain at least one value.")
        for index, kernel_size in enumerate(kernel_sizes):
            _validate_positive_odd_int(f"kernel_sizes[{index}]", kernel_size)
        for index, dilation in enumerate(branch_dilations):
            _validate_positive_int(f"branch_dilations[{index}]", dilation)
        for index, dilation in enumerate(tcn_dilations):
            _validate_positive_int(f"tcn_dilations[{index}]", dilation)

        resolved_frequency_mode = frequency_feature_mode
        if use_frequency_features and resolved_frequency_mode == "none":
            resolved_frequency_mode = "fft_magnitude"
        if resolved_frequency_mode not in ("none", "fft_magnitude"):
            raise ValueError(
                "frequency_feature_mode must be 'none' or 'fft_magnitude', "
                f"got '{frequency_feature_mode}'."
            )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.branch_channels = branch_channels
        self.num_branches = 3
        self.use_frequency_features = use_frequency_features
        self.frequency_feature_mode = resolved_frequency_mode
        self.causal = causal
        stem_in_channels = in_channels * (2 if use_frequency_features else 1)

        self.stem = nn.Sequential(
            SameLengthConv1d(
                stem_in_channels,
                hidden_channels,
                kernel_size=stem_kernel_size,
                causal=causal,
                bias=False,
            ),
            _make_norm(norm, hidden_channels, num_groups),
            _make_activation(activation),
            nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity(),
        )
        self.features = ThreeParallelShallowBranches(
            in_channels=hidden_channels,
            branch_channels=branch_channels,
            kernel_sizes=kernel_sizes,
            dilations=branch_dilations,
            norm=norm,
            activation=activation,
            dropout=dropout,
            num_groups=_resolve_group_count(branch_channels, num_groups),
        )
        self.branch_attention = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            _make_activation(activation),
            nn.Linear(hidden_channels, self.num_branches),
            nn.Softmax(dim=1),
        )
        self.fusion = nn.Sequential(
            nn.Conv1d(self.num_branches * branch_channels, fused_channels, kernel_size=1, bias=False),
            _make_norm(norm, fused_channels, num_groups),
            _make_activation(activation),
        )
        self.tcn_input = (
            nn.Identity()
            if fused_channels == tcn_channels
            else nn.Sequential(
                nn.Conv1d(fused_channels, tcn_channels, kernel_size=1, bias=False),
                _make_norm(norm, tcn_channels, num_groups),
                _make_activation(activation),
            )
        )
        self.tcn_blocks = nn.ModuleList(
            [
                TCNResidualBlock(
                    in_channels=tcn_channels,
                    out_channels=tcn_channels,
                    dilation=dilation,
                    activation=activation,
                    norm=norm,
                    dropout=dropout,
                    num_groups=num_groups,
                    causal=causal,
                )
                for dilation in tcn_dilations
            ]
        )
        self.heads = nn.ModuleList(
            [
                ComponentHead(
                    in_channels=tcn_channels,
                    activation=activation,
                    dropout=dropout,
                )
                for _ in range(out_channels)
            ]
        )

    def _augment_input(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_frequency_features:
            return x
        if self.frequency_feature_mode != "fft_magnitude":
            raise ValueError(
                "Frequency features are enabled but frequency_feature_mode is invalid: "
                f"{self.frequency_feature_mode}."
            )
        magnitude = torch.abs(torch.fft.rfft(x, dim=-1))
        resized = F.interpolate(magnitude, size=x.shape[-1], mode="linear", align_corners=False)
        return torch.cat((x, resized), dim=1)

    def branch_attention_weights(self, stem_features: torch.Tensor) -> torch.Tensor:
        """Return per-sample attention weights over the three multiscale branches."""
        if stem_features.ndim != 3:
            raise ValueError("Expected stem_features to have shape [batch, channels, length].")
        pooled = stem_features.mean(dim=2)
        return self.branch_attention(pooled)

    def apply_branch_attention(
        self,
        stem_features: torch.Tensor,
        branch_features: torch.Tensor,
    ) -> torch.Tensor:
        """Weight each branch feature map using attention computed from stem features."""
        expected_channels = self.num_branches * self.branch_channels
        if branch_features.ndim != 3:
            raise ValueError("Expected branch_features to have shape [batch, channels, length].")
        if branch_features.shape[1] != expected_channels:
            raise ValueError(
                f"Expected branch_features.shape[1] == {expected_channels}, got {branch_features.shape[1]}."
            )
        weights = self.branch_attention_weights(stem_features)
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
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Expected x.shape[1] == {self.in_channels}, got {x.shape[1]}.")

        augmented = self._augment_input(x)
        stem_features = self.stem(augmented)
        branch_features = self.features(stem_features)
        attended = self.apply_branch_attention(stem_features, branch_features)
        fused = self.tcn_input(self.fusion(attended))
        for block in self.tcn_blocks:
            fused = block(fused)
        return torch.cat([head(fused) for head in self.heads], dim=1)

    @staticmethod
    def reconstruction(components: torch.Tensor) -> torch.Tensor:
        """Return the reconstructed mixture with shape ``[batch, 1, signal_length]``."""
        return components.sum(dim=1, keepdim=True)
