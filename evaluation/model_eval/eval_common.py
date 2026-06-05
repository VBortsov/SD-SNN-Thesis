from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from decomposers.ML_methods.NN_based.datasets.synthetic import SyntheticSignalDataset
from decomposers.ML_methods.NN_based.experiment_catalog import ModelExperimentSpec
from decomposers.ML_methods.NN_based.models import create_model
from decomposers.ML_methods.NN_based.training_common import (
    component_names_from_env,
    evaluate_decomposition_loader,
    evaluate_loss,
)


def _load_state(model: nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=False)


def evaluate_experiment(
    spec: ModelExperimentSpec,
    project_root: Path,
    *,
    signal_length: int = 1024,
    fs: int = 256,
    num_test_samples: int = 300,
    batch_size: int = 32,
    permutation_invariant: bool = True,
) -> dict[str, object]:
    """Evaluate experiment."""
    checkpoint_path = project_root / spec.default_checkpoint
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    component_names = component_names_from_env()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = SyntheticSignalDataset(
        num_samples=num_test_samples,
        signal_length=signal_length,
        fs=fs,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model = create_model(spec.model_key, in_channels=1, out_channels=len(component_names)).to(device)
    _load_state(model, checkpoint_path, device)

    criterion = nn.MSELoss()
    test_loss = evaluate_loss(model, loader, criterion, device)
    test_summary = evaluate_decomposition_loader(
        model,
        loader,
        device,
        permutation_invariant,
        component_names,
    )
    report_path = project_root / spec.report
    previous_report = {}
    if report_path.exists():
        try:
            previous_report = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception:
            previous_report = {}
    report = {
        "model_name": previous_report.get("model_name", spec.key),
        "best_checkpoint": str(checkpoint_path),
        "test_loss": float(test_loss),
        "test_summary": test_summary,
        "component_names": component_names,
        "training_metadata": previous_report.get("training_metadata", {}),
        "timestamp": previous_report.get("timestamp", ""),
        "eval_timestamp": datetime.utcnow().isoformat() + "Z",
        "eval_config": {
            "signal_length": signal_length,
            "fs": fs,
            "num_test_samples": num_test_samples,
            "batch_size": batch_size,
            "permutation_invariant": permutation_invariant,
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    macro = test_summary.get("macro_average", {})
    print(f"\n[{spec.key}] Test MSE loss: {test_loss:.6f}")
    print(f"[{spec.key}] Macro corr: {float(macro.get('corr', np.nan)):.6f}")
    print(f"[{spec.key}] Saved report: {report_path}")
    return report
