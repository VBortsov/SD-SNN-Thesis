from __future__ import annotations

import inspect
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from decomposers.ML_methods.NN_based.models import MODEL_REGISTRY, create_model
from evaluation.decomposition import evaluate_decomposition, reorder_prediction
from app.services.paths import REPO_ROOT


def _resolve_checkpoint_path(checkpoint_path: str | None) -> Path | None:
    if not checkpoint_path:
        return None
    path = Path(checkpoint_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def _load_train_config(checkpoint_path: Path | None) -> dict:
    if checkpoint_path is None:
        return {}
    config_path = checkpoint_path.parent / "train_config.json"
    if not config_path.exists():
        return {}
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def load_train_config(checkpoint_path: str | None) -> dict:
    """Load the saved training config.
    
    Args:
        checkpoint_path: Checkpoint file to load.
    """
    return _load_train_config(_resolve_checkpoint_path(checkpoint_path))


def _filtered_model_kwargs(model_key: str, out_channels: int, train_config: dict) -> dict:
    cls = MODEL_REGISTRY[model_key.lower()]
    signature = inspect.signature(cls.__init__)
    accepts_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values())

    model_kwargs = {
        "in_channels": 1,
        "out_channels": out_channels,
    }
    model_kwargs.update(train_config or {})

    if accepts_kwargs:
        return model_kwargs

    valid_names = {
        name
        for name, param in signature.parameters.items()
        if name != "self" and param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    return {key: value for key, value in model_kwargs.items() if key in valid_names}


def load_model(model_key: str, out_channels: int, checkpoint_path: str | None = None) -> tuple[torch.nn.Module, str]:
    """Load a model and its training config.
    
    Args:
        model_key: Registry key for the model.
        checkpoint_path: Checkpoint file to load.
        out_channels: Number of output component channels.
    """
    checkpoint = _resolve_checkpoint_path(checkpoint_path)
    train_config = _load_train_config(checkpoint)
    requested_out_channels = int(train_config.get("out_channels", out_channels))
    model = create_model(model_key, **_filtered_model_kwargs(model_key, requested_out_channels, train_config))
    if not checkpoint_path:
        return model, "No checkpoint selected."

    path = checkpoint
    if not path.exists():
        return model, f"Checkpoint not found: {path}"

    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and "model_state_dict" in payload:
        state = payload["model_state_dict"]
    elif isinstance(payload, dict) and "state_dict" in payload:
        state = payload["state_dict"]
    else:
        state = payload
    missing, unexpected = model.load_state_dict(state, strict=False)
    msg = f"Loaded checkpoint: {path}"
    if missing:
        msg += f" | missing keys={len(missing)}"
    if unexpected:
        msg += f" | unexpected keys={len(unexpected)}"
    return model, msg


def run_inference(model: torch.nn.Module, mixture: np.ndarray) -> tuple[np.ndarray, float]:
    """Run model inference on one signal.
    
    Args:
        mixture: Observed mixed signal.
        model: Model instance to run or inspect.
    """
    x = torch.tensor(mixture, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    model.eval()
    start = perf_counter()
    with torch.inference_mode():
        y_pred = model(x).squeeze(0).detach().cpu().numpy()
    elapsed_ms = (perf_counter() - start) * 1_000.0
    return y_pred, elapsed_ms


def evaluate_prediction(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mixture: np.ndarray,
    permutation_invariant: bool = True,
) -> dict:
    """Score predicted components against targets.
    
    Args:
        y_true: Target component signals.
        y_pred: Predicted component signals.
        mixture: Observed mixed signal.
    """
    report = evaluate_decomposition(
        y_true=y_true,
        y_pred=y_pred,
        observed_mixture=mixture,
        permutation_invariant=permutation_invariant,
    )
    aligned = reorder_prediction(y_pred, report["permutation"]) if permutation_invariant else y_pred
    report["aligned_prediction"] = aligned
    return report


def count_parameters(model: torch.nn.Module) -> int:
    """Count parameters.
    
    Args:
        model: Model instance to run or inspect.
    """
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))
