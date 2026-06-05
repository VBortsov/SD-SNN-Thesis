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
    AttentionStemMultipleHeadMultiScaleBranchesDecomposer,
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
    in_channels: int = 1
    out_channels: int = 3
    stem_channels: int = 16
    branch_channels: int = 32
    fused_channels: int = 48
    kernel_sizes: tuple[int, int, int] = (5, 7, 9)
    dilations: tuple[int, int, int] = (1, 2, 4)
    norm: Literal["groupnorm", "batchnorm", "none"] = "groupnorm"
    activation: Literal["relu", "gelu", "tanh"] = "gelu"
    dropout: float = 0.1
    num_groups: int = 8
    attention_hidden_channels: int = 16
    experiment_name: str = ""
    extra_conv_layers: int = 0
    extra_conv_kernel_size: int = 3
    extra_conv_channels: int = 0
    extra_conv_dilation: int = 1
    extra_conv_activation: Literal["relu", "gelu", "tanh"] = "gelu"
    extra_conv_norm: Literal["groupnorm", "batchnorm", "none"] = "groupnorm"
    extra_conv_dropout: float = 0.1
    extra_conv_num_groups: int = 8
    extra_conv_residual: bool = False
    num_workers: int = 0
    seed: int = 42
    checkpoint_every: int = 5
    permutation_invariant_eval: bool = True
    output_dir: str = "decomposers/ML_methods/NN_based/saved_models/attention_stem_multiple_head_multiscale_branches"


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


COMPONENT_NAMES = _component_names_from_env()


def parse_args() -> TrainConfig:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train Attention-Stem Multiple-Head MultiScale Branches decomposer.")
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
    parser.add_argument("--in-channels", type=int, default=1)
    parser.add_argument("--out-channels", type=int, default=3)
    parser.add_argument("--stem-channels", type=int, default=16)
    parser.add_argument("--branch-channels", type=int, default=32)
    parser.add_argument("--fused-channels", type=int, default=48)
    parser.add_argument("--kernel-sizes", type=int, nargs=3, default=(5, 7, 9))
    parser.add_argument("--dilations", type=int, nargs=3, default=(1, 2, 4))
    parser.add_argument("--norm", type=str, default="groupnorm", choices=["groupnorm", "batchnorm", "none"])
    parser.add_argument("--activation", type=str, default="gelu", choices=["relu", "gelu", "tanh"])
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num-groups", type=int, default=8)
    parser.add_argument("--attention-hidden-channels", type=int, default=16)
    parser.add_argument("--experiment-name", type=str, default="")
    parser.add_argument("--extra-conv-layers", type=int, default=0)
    parser.add_argument("--extra-conv-kernel-size", type=int, default=3)
    parser.add_argument("--extra-conv-channels", type=int, default=0)
    parser.add_argument("--extra-conv-dilation", type=int, default=1)
    parser.add_argument("--extra-conv-activation", type=str, default="gelu", choices=["relu", "gelu", "tanh"])
    parser.add_argument("--extra-conv-norm", type=str, default="groupnorm", choices=["groupnorm", "batchnorm", "none"])
    parser.add_argument("--extra-conv-dropout", type=float, default=0.1)
    parser.add_argument("--extra-conv-num-groups", type=int, default=8)
    parser.add_argument("--extra-conv-residual", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--noise-min", type=float, default=0.0)
    parser.add_argument("--noise-max", type=float, default=0.05)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="decomposers/ML_methods/NN_based/saved_models/attention_stem_multiple_head_multiscale_branches",
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
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        stem_channels=args.stem_channels,
        branch_channels=args.branch_channels,
        fused_channels=args.fused_channels,
        kernel_sizes=tuple(args.kernel_sizes),
        dilations=tuple(args.dilations),
        norm=args.norm,
        activation=args.activation,
        dropout=args.dropout,
        num_groups=args.num_groups,
        attention_hidden_channels=args.attention_hidden_channels,
        experiment_name=args.experiment_name,
        extra_conv_layers=args.extra_conv_layers,
        extra_conv_kernel_size=args.extra_conv_kernel_size,
        extra_conv_channels=args.extra_conv_channels,
        extra_conv_dilation=args.extra_conv_dilation,
        extra_conv_activation=args.extra_conv_activation,
        extra_conv_norm=args.extra_conv_norm,
        extra_conv_dropout=args.extra_conv_dropout,
        extra_conv_num_groups=args.extra_conv_num_groups,
        extra_conv_residual=args.extra_conv_residual,
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
    return AttentionStemMultipleHeadMultiScaleBranchesDecomposer(
        in_channels=cfg.in_channels,
        out_channels=cfg.out_channels,
        stem_channels=cfg.stem_channels,
        branch_channels=cfg.branch_channels,
        fused_channels=cfg.fused_channels,
        kernel_sizes=cfg.kernel_sizes,
        dilations=cfg.dilations,
        norm=cfg.norm,
        activation=cfg.activation,
        dropout=cfg.dropout,
        num_groups=cfg.num_groups,
        attention_hidden_channels=cfg.attention_hidden_channels,
        extra_conv_layers=cfg.extra_conv_layers,
        extra_conv_kernel_size=cfg.extra_conv_kernel_size,
        extra_conv_channels=(cfg.extra_conv_channels or None),
        extra_conv_dilation=cfg.extra_conv_dilation,
        extra_conv_activation=cfg.extra_conv_activation,
        extra_conv_norm=cfg.extra_conv_norm,
        extra_conv_dropout=cfg.extra_conv_dropout,
        extra_conv_num_groups=cfg.extra_conv_num_groups,
        extra_conv_residual=cfg.extra_conv_residual,
    ).to(device)


def main() -> None:
    """Run the command-line entry point."""
    from decomposers.ML_methods.NN_based.training_common import run_training_pipeline

    cfg = parse_args()
    run_training_pipeline(
        cfg=cfg,
        project_root=PROJECT_ROOT,
        model_name="attention_stem_multiple_head_multiscale_branches",
        create_model_fn=create_model,
        component_names=COMPONENT_NAMES,
        include_sample_metrics=True,
    )


if __name__ == "__main__":
    main()
