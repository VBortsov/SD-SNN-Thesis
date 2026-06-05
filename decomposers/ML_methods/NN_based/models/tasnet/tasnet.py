from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TasNetDecomposerConfig:
    """Configuration for :class:`TasNetDecomposer`."""

    in_channels: int = 1
    out_channels: int = 3
    encoder_dim: int = 128
    bottleneck_dim: int = 128
    hidden_dim: int = 256
    skip_dim: int = 128
    kernel_size: int = 16
    stride: int = 8
    num_blocks: int = 8
    num_repeats: int = 3
    dilation_growth: int = 2
    norm_type: str = "gln"
    causal: bool = False
    mask_activation: str = "sigmoid"
    dropout: float = 0.0


class _ChannelwiseLayerNorm(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-8):
        super().__init__()
        self.ln = nn.LayerNorm(channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ln(x.transpose(1, 2)).transpose(1, 2)


class _GlobalLayerNorm(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-8):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=(1, 2), keepdim=True)
        var = (x - mean).pow(2).mean(dim=(1, 2), keepdim=True)
        return self.weight * (x - mean) / torch.sqrt(var + self.eps) + self.bias


class _SeparatorNorm(nn.Module):
    def __init__(self, channels: int, norm_type: str):
        super().__init__()
        norm_type = norm_type.lower()
        if norm_type == "gln":
            self.norm = _GlobalLayerNorm(channels)
        elif norm_type == "cln":
            self.norm = _ChannelwiseLayerNorm(channels)
        elif norm_type == "bn":
            self.norm = nn.BatchNorm1d(channels)
        else:
            raise ValueError(f"Unsupported norm_type '{norm_type}'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)


class TemporalBlock(nn.Module):
    """Residual temporal convolution block used by TasNet."""
    def __init__(
        self,
        channels: int,
        hidden_dim: int,
        skip_dim: int,
        kernel_size: int,
        dilation: int,
        norm_type: str,
        causal: bool,
        dropout: float,
    ):
        """Initialize layers and settings."""
        super().__init__()
        self.causal = causal
        self.kernel_size = kernel_size
        self.dilation = dilation

        self.in_conv = nn.Conv1d(channels, hidden_dim, kernel_size=1)
        self.prelu1 = nn.PReLU(hidden_dim)
        self.norm1 = _SeparatorNorm(hidden_dim, norm_type)

        if causal:
            padding = 0
        else:
            padding = ((kernel_size - 1) * dilation) // 2

        self.depthwise = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            groups=hidden_dim,
        )
        self.prelu2 = nn.PReLU(hidden_dim)
        self.norm2 = _SeparatorNorm(hidden_dim, norm_type)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.res_out = nn.Conv1d(hidden_dim, channels, kernel_size=1)
        self.skip_out = nn.Conv1d(hidden_dim, skip_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the forward pass.
        
        Args:
            x: Input tensor.
        """
        y = self.in_conv(x)
        y = self.prelu1(y)
        y = self.norm1(y)

        if self.causal:
            pad = (self.kernel_size - 1) * self.dilation
            y = F.pad(y, (pad, 0))

        y = self.depthwise(y)
        y = self.prelu2(y)
        y = self.norm2(y)
        y = self.dropout(y)

        residual = self.res_out(y)
        skip = self.skip_out(y)
        return x + residual, skip


class TasNetDecomposer(nn.Module):
    """Conv-TasNet style latent masking decomposer for generic 1D signals.

    Input shape: ``[batch, in_channels, signal_length]``.
    Output shape: ``[batch, out_channels, signal_length]``.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 3,
        encoder_dim: int = 128,
        bottleneck_dim: int = 128,
        hidden_dim: int = 256,
        skip_dim: int = 128,
        kernel_size: int = 16,
        stride: int = 8,
        num_blocks: int = 8,
        num_repeats: int = 3,
        dilation_growth: int = 2,
        norm_type: str = "gln",
        causal: bool = False,
        mask_activation: str = "sigmoid",
        dropout: float = 0.0,
    ):
        """Initialize layers and settings."""
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.encoder_dim = encoder_dim
        self.mask_activation = mask_activation.lower()

        self.encoder = nn.Conv1d(in_channels, encoder_dim, kernel_size=kernel_size, stride=stride, bias=False)
        self.bottleneck = nn.Conv1d(encoder_dim, bottleneck_dim, kernel_size=1)

        blocks = []
        for _ in range(num_repeats):
            for block_idx in range(num_blocks):
                dilation = dilation_growth ** block_idx
                blocks.append(
                    TemporalBlock(
                        channels=bottleneck_dim,
                        hidden_dim=hidden_dim,
                        skip_dim=skip_dim,
                        kernel_size=3,
                        dilation=dilation,
                        norm_type=norm_type,
                        causal=causal,
                        dropout=dropout,
                    )
                )
        self.separator_blocks = nn.ModuleList(blocks)

        self.mask_head = nn.Sequential(
            nn.PReLU(skip_dim),
            nn.Conv1d(skip_dim, out_channels * encoder_dim, kernel_size=1),
        )

        self.decoder = nn.ConvTranspose1d(
            encoder_dim,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=False,
        )

    def _activate_masks(self, mask_logits: torch.Tensor) -> torch.Tensor:
        if self.mask_activation == "sigmoid":
            return torch.sigmoid(mask_logits)
        if self.mask_activation == "relu":
            return F.relu(mask_logits)
        if self.mask_activation == "softmax":
            return torch.softmax(mask_logits, dim=1)
        raise ValueError(f"Unsupported mask_activation '{self.mask_activation}'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the forward pass.
        
        Args:
            x: Input tensor.
        """
        if x.ndim != 3:
            raise ValueError("Expected x to have shape [batch, channels, length].")

        input_len = x.size(-1)
        encoded = self.encoder(x)  # [B, N, T']
        sep = self.bottleneck(encoded)

        skip_sum = None
        for block in self.separator_blocks:
            sep, skip = block(sep)
            skip_sum = skip if skip_sum is None else skip_sum + skip

        mask_logits = self.mask_head(skip_sum)
        masks = mask_logits.view(x.size(0), self.out_channels, self.encoder_dim, encoded.size(-1))
        masks = self._activate_masks(masks)

        masked = masks * encoded.unsqueeze(1)
        decoded = self.decoder(masked.reshape(-1, self.encoder_dim, encoded.size(-1)))
        decoded = decoded.view(x.size(0), self.out_channels, self.in_channels, -1).squeeze(2)

        if decoded.size(-1) > input_len:
            decoded = decoded[..., :input_len]
        elif decoded.size(-1) < input_len:
            decoded = F.pad(decoded, (0, input_len - decoded.size(-1)))

        return decoded

    @staticmethod
    def reconstruction(components: torch.Tensor) -> torch.Tensor:
        return components.sum(dim=1, keepdim=True)


class ConvTasNetDecomposer(TasNetDecomposer):
    """Alias for TasNetDecomposer."""

    pass
