from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[5]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from decomposers.ML_methods.NN_based.models_shallow.multiscale_branches import (
    AttentionStemMultiHeadMultiScaleTCNDecomposer,
)


@dataclass
class TrainConfig:
    """Configuration container."""
    train_samples: int = 4000
    val_samples: int = 400
    test_samples: int = 400
    signal_length: int = 1024
    fs: int = 256
    use_weights: bool = True
    noise_min: float = 0.0
    noise_max: float = 0.05
    batch_size: int = 32
    epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    component_loss_weight: float = 1.0
    reconstruction_loss_weight: float = 0.5
    spectral_loss_weight: float = 0.05
    component_loss_weights: tuple[float, ...] | None = None
    mixing_penalty_weight: float = 0.0
    frequency_band_loss_weight: float = 0.0
    frequency_band_ranges: tuple[tuple[float, float], ...] | None = None
    in_channels: int = 1
    out_channels: int = 3
    hidden_channels: int = 24
    branch_channels: int = 12
    fused_channels: int = 32
    tcn_channels: int = 32
    kernel_sizes: tuple[int, int, int] = (5, 7, 9)
    branch_dilations: tuple[int, int, int] = (1, 2, 4)
    tcn_dilations: tuple[int, int, int] = (1, 2, 4)
    stem_kernel_size: int = 9
    activation: Literal["relu", "gelu", "tanh"] = "relu"
    norm: Literal["groupnorm", "batchnorm", "none"] = "batchnorm"
    dropout: float = 0.0
    num_groups: int = 8
    use_frequency_features: bool = False
    frequency_feature_mode: Literal["none", "fft_magnitude"] = "none"
    causal: bool = False
    experiment_name: str = ""
    num_workers: int = 0
    seed: int = 42
    checkpoint_every: int = 5
    permutation_invariant_eval: bool = True
    output_dir: str = "decomposers/ML_methods/NN_based/saved_models/attention_stem_multi_head_multiscale_tcn_inference_optimized"


def _component_names_from_env():
    raw = os.environ.get("NN_COMPONENT_TYPES", "").strip()
    names = [item.strip() for item in raw.split(",") if item.strip()] or ["harmonic", "amfm", "chirp"]
    counts = {name: names.count(name) for name in names}
    seen = {}
    unique = []
    for name in names:
        seen[name] = seen.get(name, 0) + 1
        unique.append(f"{name}_{seen[name]}" if counts[name] > 1 else name)
    return unique


def _parse_component_loss_weights(raw: str | None) -> tuple[float, ...] | None:
    if raw is None or not raw.strip():
        return None
    return tuple(float(value.strip()) for value in raw.split(",") if value.strip())


def _parse_frequency_band_ranges(raw: str | None) -> tuple[tuple[float, float], ...] | None:
    if raw is None or not raw.strip():
        return None
    bands = []
    for item in raw.split(","):
        low, high = item.split(":")
        bands.append((float(low.strip()), float(high.strip())))
    return tuple(bands)


COMPONENT_NAMES = _component_names_from_env()


def parse_args() -> TrainConfig:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train the inference-optimized shallow attention-stem multi-head multiscale TCN decomposer.")
    parser.add_argument("--train-samples", type=int, default=4000)
    parser.add_argument("--val-samples", type=int, default=400)
    parser.add_argument("--test-samples", type=int, default=400)
    parser.add_argument("--signal-length", type=int, default=1024)
    parser.add_argument("--fs", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--component-loss-weight", type=float, default=1.0)
    parser.add_argument("--reconstruction-loss-weight", type=float, default=0.5)
    parser.add_argument("--spectral-loss-weight", type=float, default=0.05)
    parser.add_argument("--component-loss-weights", type=str, default=None)
    parser.add_argument("--mixing-penalty-weight", type=float, default=0.0)
    parser.add_argument("--frequency-band-loss-weight", type=float, default=0.0)
    parser.add_argument("--frequency-band-ranges", type=str, default=None)
    parser.add_argument("--in-channels", type=int, default=1)
    parser.add_argument("--out-channels", type=int, default=3)
    parser.add_argument("--hidden-channels", type=int, default=24)
    parser.add_argument("--branch-channels", type=int, default=12)
    parser.add_argument("--fused-channels", type=int, default=32)
    parser.add_argument("--tcn-channels", type=int, default=32)
    parser.add_argument("--kernel-sizes", type=int, nargs=3, default=(5, 7, 9))
    parser.add_argument("--branch-dilations", type=int, nargs=3, default=(1, 2, 4))
    parser.add_argument("--tcn-dilations", type=int, nargs="+", default=(1, 2, 4))
    parser.add_argument("--stem-kernel-size", type=int, default=9)
    parser.add_argument("--norm", type=str, default="batchnorm", choices=["groupnorm", "batchnorm", "none"])
    parser.add_argument("--activation", type=str, default="relu", choices=["relu", "gelu", "tanh"])
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--num-groups", type=int, default=8)
    parser.add_argument("--experiment-name", type=str, default="")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--noise-min", type=float, default=0.0)
    parser.add_argument("--noise-max", type=float, default=0.05)
    parser.add_argument("--frequency-feature-mode", type=str, default="none", choices=["none", "fft_magnitude"])
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--use-frequency-features", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="decomposers/ML_methods/NN_based/saved_models/attention_stem_multi_head_multiscale_tcn_inference_optimized",
    )
    parser.add_argument("--no-weights", action="store_true")
    parser.add_argument("--disable-permutation-invariant-eval", action="store_true")
    args = parser.parse_args()

    return TrainConfig(
        train_samples=args.train_samples,
        val_samples=args.val_samples,
        test_samples=args.test_samples,
        signal_length=args.signal_length,
        fs=args.fs,
        use_weights=not args.no_weights,
        noise_min=args.noise_min,
        noise_max=args.noise_max,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        component_loss_weight=args.component_loss_weight,
        reconstruction_loss_weight=args.reconstruction_loss_weight,
        spectral_loss_weight=args.spectral_loss_weight,
        component_loss_weights=_parse_component_loss_weights(args.component_loss_weights),
        mixing_penalty_weight=args.mixing_penalty_weight,
        frequency_band_loss_weight=args.frequency_band_loss_weight,
        frequency_band_ranges=_parse_frequency_band_ranges(args.frequency_band_ranges),
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        hidden_channels=args.hidden_channels,
        branch_channels=args.branch_channels,
        fused_channels=args.fused_channels,
        tcn_channels=args.tcn_channels,
        kernel_sizes=tuple(args.kernel_sizes),
        branch_dilations=tuple(args.branch_dilations),
        tcn_dilations=tuple(args.tcn_dilations),
        stem_kernel_size=args.stem_kernel_size,
        norm=args.norm,
        activation=args.activation,
        dropout=args.dropout,
        num_groups=args.num_groups,
        use_frequency_features=args.use_frequency_features,
        frequency_feature_mode=args.frequency_feature_mode,
        causal=args.causal,
        experiment_name=args.experiment_name,
        num_workers=args.num_workers,
        seed=args.seed,
        checkpoint_every=args.checkpoint_every,
        permutation_invariant_eval=not args.disable_permutation_invariant_eval,
        output_dir=args.output_dir,
    )


def create_model(cfg: TrainConfig, device: torch.device) -> nn.Module:
    """Build the model instance.
    
    Args:
        cfg: Training configuration.
        device: Torch device for tensors and modules.
    """
    return AttentionStemMultiHeadMultiScaleTCNDecomposer(
        in_channels=cfg.in_channels,
        out_channels=cfg.out_channels,
        hidden_channels=cfg.hidden_channels,
        branch_channels=cfg.branch_channels,
        fused_channels=cfg.fused_channels,
        tcn_channels=cfg.tcn_channels,
        kernel_sizes=cfg.kernel_sizes,
        branch_dilations=cfg.branch_dilations,
        tcn_dilations=cfg.tcn_dilations,
        stem_kernel_size=cfg.stem_kernel_size,
        norm=cfg.norm,
        activation=cfg.activation,
        dropout=cfg.dropout,
        num_groups=cfg.num_groups,
        use_frequency_features=cfg.use_frequency_features,
        frequency_feature_mode=cfg.frequency_feature_mode,
        causal=cfg.causal,
    ).to(device)


def main() -> None:
    """Run the command-line entry point."""
    from decomposers.ML_methods.NN_based.training_common import run_training_pipeline

    cfg = parse_args()
    run_training_pipeline(
        cfg=cfg,
        project_root=PROJECT_ROOT,
        model_name="attention_stem_multi_head_multiscale_tcn_inference_optimized",
        create_model_fn=create_model,
        component_names=COMPONENT_NAMES,
        include_sample_metrics=True,
    )


if __name__ == "__main__":
    main()
