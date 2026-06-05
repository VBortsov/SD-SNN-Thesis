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
from decomposers.ML_methods.NN_based.models.autoencoderbased.autoencoderbased import AutoencoderDecomposer
from evaluation.decomposition import evaluate_decomposition


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
    hidden_channels: tuple[int, ...] = (16, 32, 64)
    kernel_size: int = 5
    dropout: float = 0.1
    experiment_name: str = ""
    num_workers: int = 0
    seed: int = 42
    checkpoint_every: int = 5
    permutation_invariant_eval: bool = True
    output_dir: str = "decomposers/ML_methods/NN_based/saved_models/autoencoderbased"


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


def parse_hidden_channels(raw: str) -> tuple[int, ...]:
    """Parse hidden channels.
    
    Args:
        raw: Raw string value from arguments.
    """
    values = tuple(int(v.strip()) for v in raw.split(",") if v.strip())
    if not values:
        raise ValueError("hidden_channels cannot be empty.")
    return values


def parse_args() -> TrainConfig:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train autoencoder-based decomposer on synthetic signal decomposition data with checkpointing and integrated evaluation."
    )
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
    parser.add_argument("--hidden-channels", type=str, default="16,32,64")
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--experiment-name", type=str, default="")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--noise-min", type=float, default=0.0)
    parser.add_argument("--noise-max", type=float, default=0.05)
    parser.add_argument("--output-dir", type=str, default="decomposers/ML_methods/NN_based/saved_models/autoencoderbased")
    parser.add_argument("--no-weights", action="store_true", help="Disable random component weights in synthetic data.")
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
        hidden_channels=parse_hidden_channels(args.hidden_channels),
        kernel_size=args.kernel_size,
        dropout=args.dropout,
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
    model = AutoencoderDecomposer(
        in_channels=cfg.in_channels,
        out_channels=cfg.out_channels,
        hidden_channels=cfg.hidden_channels,
        kernel_size=cfg.kernel_size,
        dropout=cfg.dropout,
    )
    return model.to(device)


def main() -> None:
    """Run the command-line entry point."""
    from decomposers.ML_methods.NN_based.training_common import run_training_pipeline

    cfg = parse_args()
    run_training_pipeline(
        cfg=cfg,
        project_root=PROJECT_ROOT,
        model_name="autoencoderbased",
        create_model_fn=create_model,
        component_names=COMPONENT_NAMES,
        include_sample_metrics=True,
    )


if __name__ == "__main__":
    main()
