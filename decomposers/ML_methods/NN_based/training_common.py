from __future__ import annotations

import json
import random
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from decomposers.ML_methods.NN_based.datasets.synthetic import SyntheticSignalDataset
from evaluation.decomposition import evaluate_decomposition


class DecompositionLoss(nn.Module):
    """Composite loss for supervised signal decomposition."""

    def __init__(
        self,
        component_weight: float = 1.0,
        reconstruction_weight: float = 0.5,
        spectral_weight: float = 0.05,
        component_loss_weights: Sequence[float] | None = None,
        mixing_penalty_weight: float = 0.0,
        frequency_band_weight: float = 0.0,
        frequency_band_ranges: Sequence[Sequence[float]] | None = None,
        sample_rate: float | None = None,
    ):
        """Initialize layers and settings."""
        super().__init__()
        if component_weight < 0:
            raise ValueError(f"component_weight must be >= 0, got {component_weight}.")
        if reconstruction_weight < 0:
            raise ValueError(f"reconstruction_weight must be >= 0, got {reconstruction_weight}.")
        if spectral_weight < 0:
            raise ValueError(f"spectral_weight must be >= 0, got {spectral_weight}.")
        if mixing_penalty_weight < 0:
            raise ValueError(f"mixing_penalty_weight must be >= 0, got {mixing_penalty_weight}.")
        if frequency_band_weight < 0:
            raise ValueError(f"frequency_band_weight must be >= 0, got {frequency_band_weight}.")
        if frequency_band_weight > 0.0 and sample_rate is not None and sample_rate <= 0:
            raise ValueError(f"sample_rate must be > 0, got {sample_rate}.")
        self.component_weight = component_weight
        self.reconstruction_weight = reconstruction_weight
        self.spectral_weight = spectral_weight
        self.component_loss_weights = (
            torch.tensor(component_loss_weights, dtype=torch.float32)
            if component_loss_weights is not None
            else None
        )
        self.mixing_penalty_weight = mixing_penalty_weight
        self.frequency_band_weight = frequency_band_weight
        self.frequency_band_ranges = (
            [tuple(float(value) for value in band) for band in frequency_band_ranges]
            if frequency_band_ranges is not None
            else None
        )
        self.sample_rate = float(sample_rate) if sample_rate is not None else None
        self.l1 = nn.L1Loss()

    def _component_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        per_component = F.mse_loss(pred, target, reduction="none").mean(dim=(0, 2))
        if self.component_loss_weights is None:
            return per_component.mean()
        if self.component_loss_weights.numel() != pred.shape[1]:
            raise ValueError(
                "component_loss_weights length must match pred.shape[1]. "
                f"Got {self.component_loss_weights.numel()} and {pred.shape[1]}."
            )
        weights = self.component_loss_weights.to(device=pred.device, dtype=pred.dtype)
        return (per_component * weights).sum() / weights.sum().clamp_min(torch.finfo(pred.dtype).eps)

    @staticmethod
    def _mixing_penalty(pred: torch.Tensor) -> torch.Tensor:
        if pred.shape[1] < 2:
            return pred.new_zeros(())
        centered = pred - pred.mean(dim=-1, keepdim=True)
        normalized = F.normalize(centered, p=2, dim=-1, eps=1e-8)
        similarities = []
        for i in range(pred.shape[1]):
            for j in range(i + 1, pred.shape[1]):
                similarities.append((normalized[:, i] * normalized[:, j]).sum(dim=-1).abs().mean())
        if not similarities:
            return pred.new_zeros(())
        return torch.stack(similarities).mean()

    def _frequency_band_loss(self, pred: torch.Tensor) -> torch.Tensor:
        if (
            self.frequency_band_weight <= 0.0
            or not self.frequency_band_ranges
            or self.sample_rate is None
        ):
            return pred.new_zeros(())
        if len(self.frequency_band_ranges) != pred.shape[1]:
            raise ValueError(
                "frequency_band_ranges length must match pred.shape[1]. "
                f"Got {len(self.frequency_band_ranges)} and {pred.shape[1]}."
            )
        freqs = torch.fft.rfftfreq(pred.shape[-1], d=1.0 / self.sample_rate).to(pred.device)
        spectrum = torch.abs(torch.fft.rfft(pred, dim=-1)).pow(2)
        penalties = []
        for index, (low, high) in enumerate(self.frequency_band_ranges):
            if low < 0 or high < low:
                raise ValueError(f"Invalid frequency band range {(low, high)} at index {index}.")
            band_mask = ((freqs >= low) & (freqs <= high)).to(dtype=pred.dtype).view(1, 1, -1)
            component_energy = spectrum[:, index : index + 1]
            outside_energy = (component_energy * (1.0 - band_mask)).sum(dim=-1)
            total_energy = component_energy.sum(dim=-1).clamp_min(torch.finfo(pred.dtype).eps)
            penalties.append((outside_energy / total_energy).mean())
        return torch.stack(penalties).mean() if penalties else pred.new_zeros(())

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mixture: torch.Tensor) -> torch.Tensor:
        """Run the forward pass.
        
        Args:
            mixture: Observed mixed signal.
            pred: Predicted tensor.
            target: Target tensor.
        """
        component_loss = self._component_loss(pred, target)

        pred_sum = pred.sum(dim=1, keepdim=True)
        reconstruction_loss = F.mse_loss(pred_sum, mixture)

        pred_spectrum = torch.abs(torch.fft.rfft(pred, dim=-1))
        target_spectrum = torch.abs(torch.fft.rfft(target, dim=-1))
        spectral_loss = self.l1(pred_spectrum, target_spectrum)
        mixing_penalty = self._mixing_penalty(pred)
        frequency_band_loss = self._frequency_band_loss(pred)

        return (
            self.component_weight * component_loss
            + self.reconstruction_weight * reconstruction_loss
            + self.spectral_weight * spectral_loss
            + self.mixing_penalty_weight * mixing_penalty
            + self.frequency_band_weight * frequency_band_loss
        )


def build_decomposition_loss(cfg) -> DecompositionLoss:
    """Build decomposition loss.
    
    Args:
        cfg: Training configuration.
    """
    return DecompositionLoss(
        component_weight=float(getattr(cfg, "component_loss_weight", 1.0)),
        reconstruction_weight=float(getattr(cfg, "reconstruction_loss_weight", 0.5)),
        spectral_weight=float(getattr(cfg, "spectral_loss_weight", 0.05)),
        component_loss_weights=getattr(cfg, "component_loss_weights", None),
        mixing_penalty_weight=float(getattr(cfg, "mixing_penalty_weight", 0.0)),
        frequency_band_weight=float(getattr(cfg, "frequency_band_loss_weight", 0.0)),
        frequency_band_ranges=getattr(cfg, "frequency_band_ranges", None),
        sample_rate=getattr(cfg, "fs", None),
    )


def component_names_from_env() -> list[str]:
    """Component names from env."""
    import os

    raw = os.environ.get("NN_COMPONENT_TYPES", "").strip()
    names = [item.strip() for item in raw.split(",") if item.strip()] or ["harmonic", "amfm", "chirp"]
    counts = {name: names.count(name) for name in names}
    seen: dict[str, int] = {}
    unique = []
    for name in names:
        seen[name] = seen.get(name, 0) + 1
        unique.append(f"{name}_{seen[name]}" if counts[name] > 1 else name)
    return unique


def set_seed(seed: int) -> None:
    """Set seed.
    
    Args:
        seed: Random seed for reproducible output.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_loaders(cfg):
    """Build loaders.
    
    Args:
        cfg: Training configuration.
    """
    common = dict(
        signal_length=cfg.signal_length,
        fs=cfg.fs,
        use_weights=cfg.use_weights,
        noise_range=(cfg.noise_min, cfg.noise_max),
    )
    train_ds = SyntheticSignalDataset(num_samples=cfg.train_samples, **common)
    val_ds = SyntheticSignalDataset(num_samples=cfg.val_samples, **common)
    test_ds = SyntheticSignalDataset(num_samples=cfg.test_samples, **common)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    eval_kwargs = dict(
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, DataLoader(val_ds, **eval_kwargs), DataLoader(test_ds, **eval_kwargs)


def move_batch(batch, device: torch.device):
    """Move batch.
    
    Args:
        batch: Batch from a data loader.
        device: Torch device for tensors and modules.
    """
    x, y = batch
    return x.to(device), y.to(device)


def compute_loss(criterion: nn.Module, pred: torch.Tensor, target: torch.Tensor, mixture: torch.Tensor) -> torch.Tensor:
    """Compute loss.
    
    Args:
        mixture: Observed mixed signal.
    """
    try:
        return criterion(pred, target, mixture)
    except TypeError:
        return criterion(pred, target)


@torch.no_grad()
def evaluate_loss(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
    """Evaluate loss."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    for batch in loader:
        x, y = move_batch(batch, device)
        loss = compute_loss(criterion, model(x), y, x)
        total_loss += loss.item() * x.size(0)
        total_samples += x.size(0)
    return total_loss / max(total_samples, 1)


@torch.no_grad()
def evaluate_decomposition_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    permutation_invariant: bool,
    component_names: list[str],
) -> dict[str, dict[str, float]]:
    """Evaluate decomposition loader."""
    model.eval()
    reports = []
    for batch in loader:
        x, y = move_batch(batch, device)
        pred = model(x)
        x_np = x.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()
        pred_np = pred.detach().cpu().numpy()
        for i in range(y_np.shape[0]):
            reports.append(
                evaluate_decomposition(
                    y_true=y_np[i],
                    y_pred=pred_np[i],
                    observed_mixture=x_np[i].squeeze(),
                    permutation_invariant=permutation_invariant,
                )
            )

    empty = {"macro_average": {}, "clean_sum_metrics": {}, "observed_mixture_metrics": {}, "mixture_metrics": {}, "components": {}}
    if not reports:
        return empty

    metric_names = list(reports[0]["macro_average"].keys())
    summary = {
        "macro_average": {metric: float(np.mean([r["macro_average"][metric] for r in reports])) for metric in metric_names},
        "clean_sum_metrics": {metric: float(np.mean([r["clean_sum_metrics"][metric] for r in reports])) for metric in metric_names},
        "observed_mixture_metrics": {metric: float(np.mean([r["observed_mixture_metrics"][metric] for r in reports])) for metric in metric_names},
        "components": {},
    }
    summary["mixture_metrics"] = summary["clean_sum_metrics"]
    for idx, name in enumerate(component_names[: len(reports[0]["component_metrics"])]):
        summary["components"][name] = {
            metric: float(np.mean([r["component_metrics"][idx][metric] for r in reports]))
            for metric in metric_names
        }
    return summary


@torch.no_grad()
def infer_single_batch_metrics(model: nn.Module, loader: DataLoader, device: torch.device, permutation_invariant: bool) -> dict[str, float]:
    """Infer single batch metrics."""
    for batch in loader:
        x, y = move_batch(batch, device)
        pred = model(x)
        report = evaluate_decomposition(
            y_true=y[0].detach().cpu().numpy(),
            y_pred=pred[0].detach().cpu().numpy(),
            observed_mixture=x[0].detach().cpu().numpy().squeeze(),
            permutation_invariant=permutation_invariant,
        )
        return report["macro_average"]
    return {}


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.Module, device: torch.device) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_samples = 0
    for batch_idx, batch in enumerate(loader, start=1):
        x, y = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        loss = compute_loss(criterion, model(x), y, x)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        total_samples += x.size(0)
        if batch_idx == 1 or batch_idx % 10 == 0:
            print(f"  batch {batch_idx:04d} | loss={loss.item():.6f}", flush=True)
    return total_loss / max(total_samples, 1)


def save_json(path: Path, data: dict[str, object]) -> None:
    """Write a JSON payload to disk.
    
    Args:
        path: File or directory path.
        data: Serializable data payload.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def checkpoint_payload(model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, cfg, train_loss: float, val_loss: float, val_summary: dict) -> dict[str, object]:
    """Checkpoint payload."""
    return {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_summary": val_summary,
        "config": asdict(cfg),
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "model_class": model.__class__.__name__,
    }


@torch.no_grad()
def estimate_model_stats(model: nn.Module, signal_length: int, device: torch.device) -> dict[str, float]:
    """Estimate model stats.
    
    Args:
        model: Model instance to run or inspect.
        signal_length: Input signal length.
        device: Torch device for tensors and modules.
    """
    parameter_count = float(sum(parameter.numel() for parameter in model.parameters()))
    model.eval()
    in_channels = int(getattr(model, "in_channels", 1))
    dummy = torch.zeros(1, in_channels, signal_length, device=device)
    for _ in range(2):
        _ = model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    timings = []
    for _ in range(5):
        started = time.perf_counter()
        _ = model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        timings.append((time.perf_counter() - started) * 1000.0)
    return {
        "parameter_count": parameter_count,
        "inference_time_ms": float(np.mean(timings)),
    }


def print_epoch_summary(epoch: int, train_loss: float, val_loss: float, val_summary: dict) -> None:
    """Print epoch summary."""
    macro = val_summary.get("macro_average", {})
    mix = val_summary.get("mixture_metrics", {})
    corr = macro.get("corr", float("nan"))
    snr = macro.get("snr_db", float("nan"))
    mix_mse = mix.get("mse", float("nan"))
    print(
        f"epoch {epoch:03d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f} "
        f"| val_corr={corr:.6f} | val_snr_db={snr:.6f} | mixture_mse={mix_mse:.6f}",
        flush=True,
    )


def run_training_pipeline(
    *,
    cfg,
    project_root: Path,
    model_name: str,
    create_model_fn: Callable[[object, torch.device], nn.Module],
    component_names: list[str] | None = None,
    include_sample_metrics: bool = True,
) -> dict[str, object]:
    """Run training pipeline."""
    component_names = component_names or component_names_from_env()
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = (project_root / cfg.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    training_started = datetime.utcnow()

    print(f"Project root: {project_root}", flush=True)
    print(f"Using device: {device}", flush=True)
    print(f"Saving outputs to: {output_dir}", flush=True)

    train_loader, val_loader, test_loader = build_loaders(cfg)
    model = create_model_fn(cfg, device)
    model_stats = estimate_model_stats(model, int(getattr(cfg, "signal_length", 1024)), device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    criterion = build_decomposition_loss(cfg)

    best_val_loss = float("inf")
    history: list[dict] = []
    best_checkpoint_path = output_dir / f"{model_name}_best.pt"
    last_checkpoint_path = output_dir / f"{model_name}_last.pt"
    best_weights_path = output_dir / f"{model_name}_best_weights_only.pt"
    last_weights_path = output_dir / f"{model_name}_last_weights_only.pt"

    save_json(output_dir / "train_config.json", asdict(cfg))

    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate_loss(model, val_loader, criterion, device)
        val_summary = evaluate_decomposition_loader(
            model,
            val_loader,
            device,
            cfg.permutation_invariant_eval,
            component_names,
        )
        epoch_record = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_summary": val_summary}
        history.append(epoch_record)
        print_epoch_summary(epoch, train_loss, val_loss, val_summary)

        payload = checkpoint_payload(model, optimizer, epoch, cfg, train_loss, val_loss, val_summary)
        torch.save(payload, last_checkpoint_path)
        torch.save(model.state_dict(), last_weights_path)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(payload, best_checkpoint_path)
            torch.save(model.state_dict(), best_weights_path)
            print(f"  -> saved new best checkpoint: {best_checkpoint_path.name}", flush=True)

        if cfg.checkpoint_every > 0 and epoch % cfg.checkpoint_every == 0:
            periodic_path = output_dir / f"{model_name}_epoch_{epoch:03d}.pt"
            torch.save(payload, periodic_path)
            print(f"  -> saved periodic checkpoint: {periodic_path.name}", flush=True)

        save_json(output_dir / "training_history.json", {"history": history})
        save_json(output_dir / "latest_metrics.json", {"latest_epoch": epoch, "best_val_loss": best_val_loss, "current": epoch_record})

    print("\nLoading best checkpoint for final test evaluation...", flush=True)
    best_state = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(best_state["model_state_dict"])
    test_loss = evaluate_loss(model, test_loader, criterion, device)
    test_summary = evaluate_decomposition_loader(
        model,
        test_loader,
        device,
        cfg.permutation_invariant_eval,
        component_names,
    )
    final_report = {
        "model_name": str(model_name),
        "best_checkpoint": str(best_checkpoint_path),
        "weights_for_future_evaluation": str(best_weights_path),
        "test_loss": test_loss,
        "test_summary": test_summary,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "training_metadata": {
            "model_name": str(model_name),
            "output_dir_name": output_dir.name,
            "started_utc": training_started.isoformat() + "Z",
            "finished_utc": datetime.utcnow().isoformat() + "Z",
            "elapsed_seconds": (datetime.utcnow() - training_started).total_seconds(),
            "train_samples": int(getattr(cfg, "train_samples", 0)),
            "val_samples": int(getattr(cfg, "val_samples", 0)),
            "test_samples": int(getattr(cfg, "test_samples", 0)),
            "total_samples": int(getattr(cfg, "train_samples", 0))
            + int(getattr(cfg, "val_samples", 0))
            + int(getattr(cfg, "test_samples", 0)),
            "epochs": int(getattr(cfg, "epochs", 0)),
            "batch_size": int(getattr(cfg, "batch_size", 0)),
            "signal_length": int(getattr(cfg, "signal_length", 0)),
            "fs": int(getattr(cfg, "fs", 0)),
            "component_loss_weight": float(getattr(cfg, "component_loss_weight", 1.0)),
            "reconstruction_loss_weight": float(getattr(cfg, "reconstruction_loss_weight", 0.5)),
            "spectral_loss_weight": float(getattr(cfg, "spectral_loss_weight", 0.05)),
            "mixing_penalty_weight": float(getattr(cfg, "mixing_penalty_weight", 0.0)),
            "frequency_band_loss_weight": float(getattr(cfg, "frequency_band_loss_weight", 0.0)),
            "parameter_count": model_stats["parameter_count"],
            "inference_time_ms": model_stats["inference_time_ms"],
            "extra_conv_layers": int(getattr(cfg, "extra_conv_layers", 0)),
            "extra_conv_kernel_size": int(getattr(cfg, "extra_conv_kernel_size", 0)),
            "extra_conv_channels": int(getattr(cfg, "extra_conv_channels", 0)),
            "extra_conv_dilation": int(getattr(cfg, "extra_conv_dilation", 0)),
            "extra_conv_dropout": float(getattr(cfg, "extra_conv_dropout", 0.0)),
            "extra_conv_num_groups": int(getattr(cfg, "extra_conv_num_groups", 0)),
            "extra_conv_residual": bool(getattr(cfg, "extra_conv_residual", False)),
            "extra_conv_activation": str(getattr(cfg, "extra_conv_activation", "")),
            "extra_conv_norm": str(getattr(cfg, "extra_conv_norm", "")),
            "experiment_name": str(getattr(cfg, "experiment_name", "")),
        },
    }
    if include_sample_metrics:
        final_report["sample_macro_average"] = infer_single_batch_metrics(model, test_loader, device, cfg.permutation_invariant_eval)

    save_json(output_dir / "final_test_report.json", final_report)
    print("\nFinal test metrics", flush=True)
    print(json.dumps(final_report, indent=2), flush=True)
    print("\nTraining complete.", flush=True)
    print(f"Best checkpoint: {best_checkpoint_path}", flush=True)
    print(f"Weights-only file: {best_weights_path}", flush=True)
    return final_report
