import argparse
import json
import os
import random
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[5]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from decomposers.ML_methods.NN_based.datasets.synthetic import SyntheticSignalDataset
from decomposers.ML_methods.NN_based.models.sepformer import SepFormerDecomposer
from evaluation.decomposition import evaluate_decomposition


@dataclass
class TrainConfig:
    """Configuration container."""
    train_samples: int = 3000
    val_samples: int = 300
    test_samples: int = 300
    signal_length: int = 1024
    fs: int = 256
    use_weights: bool = True
    noise_min: float = 0.0
    noise_max: float = 0.05
    batch_size: int = 8
    epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    component_loss_weight: float = 1.0
    reconstruction_loss_weight: float = 0.5
    spectral_loss_weight: float = 0.05
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
    experiment_name: str = ""
    num_workers: int = 0
    seed: int = 42
    checkpoint_every: int = 5
    permutation_invariant_eval: bool = True
    output_dir: str = "decomposers/ML_methods/NN_based/saved_models/sepformer"


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
    parser = argparse.ArgumentParser(description="Train SepFormer decomposer on synthetic decomposition data.")
    parser.add_argument("--train-samples", type=int, default=3000)
    parser.add_argument("--val-samples", type=int, default=300)
    parser.add_argument("--test-samples", type=int, default=300)
    parser.add_argument("--signal-length", type=int, default=1024)
    parser.add_argument("--fs", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--component-loss-weight", type=float, default=1.0)
    parser.add_argument("--reconstruction-loss-weight", type=float, default=0.5)
    parser.add_argument("--spectral-loss-weight", type=float, default=0.05)
    parser.add_argument("--in-channels", type=int, default=1)
    parser.add_argument("--out-channels", type=int, default=3)
    parser.add_argument("--encoder-dim", type=int, default=128)
    parser.add_argument("--bottleneck-dim", type=int, default=128)
    parser.add_argument("--kernel-size", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--hop-size", type=int, default=50)
    parser.add_argument("--num-sepformer-blocks", type=int, default=2)
    parser.add_argument("--num-attention-heads", type=int, default=8)
    parser.add_argument("--feedforward-dim", type=int, default=256)
    parser.add_argument("--transformer-dropout", type=float, default=0.1)
    parser.add_argument("--mask-activation", choices=["sigmoid", "relu", "softmax"], default="sigmoid")
    parser.add_argument("--experiment-name", type=str, default="")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--noise-min", type=float, default=0.0)
    parser.add_argument("--noise-max", type=float, default=0.05)
    parser.add_argument("--output-dir", type=str, default="decomposers/ML_methods/NN_based/saved_models/sepformer")
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
        encoder_dim=args.encoder_dim,
        bottleneck_dim=args.bottleneck_dim,
        kernel_size=args.kernel_size,
        stride=args.stride,
        chunk_size=args.chunk_size,
        hop_size=args.hop_size,
        num_sepformer_blocks=args.num_sepformer_blocks,
        num_attention_heads=args.num_attention_heads,
        feedforward_dim=args.feedforward_dim,
        transformer_dropout=args.transformer_dropout,
        mask_activation=args.mask_activation,
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
    return SepFormerDecomposer(
        in_channels=cfg.in_channels,
        out_channels=cfg.out_channels,
        encoder_dim=cfg.encoder_dim,
        bottleneck_dim=cfg.bottleneck_dim,
        kernel_size=cfg.kernel_size,
        stride=cfg.stride,
        chunk_size=cfg.chunk_size,
        hop_size=cfg.hop_size,
        num_sepformer_blocks=cfg.num_sepformer_blocks,
        num_attention_heads=cfg.num_attention_heads,
        feedforward_dim=cfg.feedforward_dim,
        transformer_dropout=cfg.transformer_dropout,
        mask_activation=cfg.mask_activation,
    ).to(device)


def main() -> None:
    """Run the command-line entry point."""
    from decomposers.ML_methods.NN_based.training_common import run_training_pipeline

    cfg = parse_args()
    run_training_pipeline(
        cfg=cfg,
        project_root=PROJECT_ROOT,
        model_name="sepformer",
        create_model_fn=create_model,
        component_names=COMPONENT_NAMES,
        include_sample_metrics=True,
    )


if __name__ == "__main__":
    main()
