from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SepFormerDecomposerConfig:
    """Configuration for :class:`SepFormerDecomposer`."""

    in_channels: int = 1
    out_channels: int = 3
    encoder_dim: int = 128
    bottleneck_dim: int = 128
    kernel_size: int = 16
    stride: int = 8
    chunk_size: int = 100
    hop_size: int = 50
    num_sepformer_blocks: int = 2
    num_attention_heads: int = 8
    feedforward_dim: int = 256
    transformer_dropout: float = 0.1
    mask_activation: str = "sigmoid"


class DualPathTransformerBlock(nn.Module):
    """Transformer block over local chunks and chunk order."""
    def __init__(
        self,
        dim: int,
        num_heads: int,
        feedforward_dim: int,
        dropout: float,
    ):
        """Initialize layers and settings."""
        super().__init__()
        layer_args = dict(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.intra = nn.TransformerEncoderLayer(**layer_args)
        self.inter = nn.TransformerEncoderLayer(**layer_args)
        self.intra_norm = nn.LayerNorm(dim)
        self.inter_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, K, S]
        """Run the forward pass.
        
        Args:
            x: Input tensor.
        """
        b, n, k, s = x.shape

        intra_in = x.permute(0, 2, 3, 1).reshape(b * k, s, n)
        intra_out = self.intra(intra_in)
        intra_out = self.intra_norm(intra_in + intra_out)
        intra_out = intra_out.reshape(b, k, s, n).permute(0, 3, 1, 2)

        inter_in = intra_out.permute(0, 3, 2, 1).reshape(b * s, k, n)
        inter_out = self.inter(inter_in)
        inter_out = self.inter_norm(inter_in + inter_out)
        inter_out = inter_out.reshape(b, s, k, n).permute(0, 3, 2, 1)

        return inter_out


class SepFormerDecomposer(nn.Module):
    """SepFormer-style decomposer for generic 1D signals.

    Input shape: ``[batch, in_channels, signal_length]``.
    Output shape: ``[batch, out_channels, signal_length]``.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 3,
        encoder_dim: int = 128,
        bottleneck_dim: int = 128,
        kernel_size: int = 16,
        stride: int = 8,
        chunk_size: int = 100,
        hop_size: int = 50,
        num_sepformer_blocks: int = 2,
        transformer_dim: int | None = None,
        num_attention_heads: int = 8,
        feedforward_dim: int = 256,
        transformer_dropout: float = 0.1,
        positional_encoding: str = "none",
        mask_activation: str = "sigmoid",
    ):
        """Initialize layers and settings."""
        super().__init__()

        if transformer_dim is not None and transformer_dim != bottleneck_dim:
            raise ValueError("For this implementation, transformer_dim must match bottleneck_dim.")
        if positional_encoding != "none":
            raise ValueError("Only positional_encoding='none' is currently supported.")
        if hop_size <= 0 or chunk_size <= 0:
            raise ValueError("chunk_size and hop_size must be positive.")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.encoder_dim = encoder_dim
        self.chunk_size = chunk_size
        self.hop_size = hop_size
        self.mask_activation = mask_activation.lower()

        self.encoder = nn.Conv1d(in_channels, encoder_dim, kernel_size=kernel_size, stride=stride, bias=False)
        self.bottleneck = nn.Conv1d(encoder_dim, bottleneck_dim, kernel_size=1)

        self.blocks = nn.ModuleList(
            [
                DualPathTransformerBlock(
                    dim=bottleneck_dim,
                    num_heads=num_attention_heads,
                    feedforward_dim=feedforward_dim,
                    dropout=transformer_dropout,
                )
                for _ in range(num_sepformer_blocks)
            ]
        )

        self.mask_head = nn.Conv1d(bottleneck_dim, out_channels * encoder_dim, kernel_size=1)
        self.decoder = nn.ConvTranspose1d(encoder_dim, in_channels, kernel_size=kernel_size, stride=stride, bias=False)

    def _activate_masks(self, mask_logits: torch.Tensor) -> torch.Tensor:
        if self.mask_activation == "sigmoid":
            return torch.sigmoid(mask_logits)
        if self.mask_activation == "relu":
            return F.relu(mask_logits)
        if self.mask_activation == "softmax":
            return torch.softmax(mask_logits, dim=1)
        raise ValueError(f"Unsupported mask_activation '{self.mask_activation}'.")

    def _chunk(self, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        # x: [B, N, T]
        t = x.size(-1)
        if t < self.chunk_size:
            pad = self.chunk_size - t
        else:
            rem = (t - self.chunk_size) % self.hop_size
            pad = (self.hop_size - rem) % self.hop_size
        if pad > 0:
            x = F.pad(x, (0, pad))
        chunks = x.unfold(dimension=-1, size=self.chunk_size, step=self.hop_size)
        return chunks.contiguous(), pad

    def _merge(self, chunks: torch.Tensor, original_t: int, pad: int) -> torch.Tensor:
        # chunks: [B, N, K, S]
        b, n, k, s = chunks.shape
        total = (k - 1) * self.hop_size + s
        merged = chunks.new_zeros((b, n, total))
        denom = chunks.new_zeros((b, 1, total))
        for idx in range(k):
            start = idx * self.hop_size
            end = start + s
            merged[:, :, start:end] += chunks[:, :, idx, :]
            denom[:, :, start:end] += 1.0
        merged = merged / denom.clamp_min(1e-8)
        if pad > 0:
            merged = merged[..., :-pad]
        return merged[..., :original_t]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the forward pass.
        
        Args:
            x: Input tensor.
        """
        if x.ndim != 3:
            raise ValueError("Expected x to have shape [batch, channels, length].")

        input_len = x.size(-1)
        encoded = self.encoder(x)
        sep = self.bottleneck(encoded)

        chunks, pad = self._chunk(sep)
        for block in self.blocks:
            chunks = block(chunks)

        sep_out = self._merge(chunks, original_t=sep.size(-1), pad=pad)

        mask_logits = self.mask_head(sep_out)
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
