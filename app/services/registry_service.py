from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

from app.services.paths import MODEL_REGISTRY_PATH, SAVED_MODELS_DIR, REPO_ROOT, ensure_app_dirs
from decomposers.ML_methods.NN_based.experiment_catalog import MODEL_EXPERIMENTS


@dataclass
class ModelRegistryEntry:
    """Structured registry or manifest entry."""
    key: str
    model_key: str
    display_name: str
    family: str
    depth_label: str
    num_layers: int | None
    notes: str
    default_checkpoint: str
    enabled: bool = True


DEFAULT_REGISTRY: list[ModelRegistryEntry] = [
    ModelRegistryEntry(
        key=spec.key,
        model_key=spec.model_key,
        display_name=spec.display_name,
        family=spec.family,
        depth_label=spec.depth_label,
        num_layers=None,
        notes="",
        default_checkpoint=spec.default_checkpoint,
        enabled=spec.enabled,
    )
    for spec in MODEL_EXPERIMENTS.values()
]


def _to_dicts(entries: Iterable[ModelRegistryEntry]) -> list[dict]:
    return [asdict(item) for item in entries]


def ensure_registry_file() -> None:
    """Ensure registry file."""
    ensure_app_dirs()
    if not MODEL_REGISTRY_PATH.exists():
        payload = {"models": _to_dicts(DEFAULT_REGISTRY)}
        MODEL_REGISTRY_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_registry() -> list[dict]:
    """Load model registry entries."""
    ensure_registry_file()
    payload = json.loads(MODEL_REGISTRY_PATH.read_text(encoding="utf-8"))
    models = payload.get("models", [])
    if not isinstance(models, list):
        return _to_dicts(DEFAULT_REGISTRY)
    existing_by_key = {
        str(row.get("key", "")).strip(): row
        for row in models
        if str(row.get("key", "")).strip()
    }
    merged_models: list[dict] = []
    changed = False

    for default in _to_dicts(DEFAULT_REGISTRY):
        key = str(default.get("key", "")).strip()
        current = existing_by_key.pop(key, None)
        if current is None:
            merged_models.append(default)
            changed = True
            continue

        merged = dict(current)
        for field, value in default.items():
            if field not in merged:
                merged[field] = value
                changed = True
        merged_models.append(merged)

    if existing_by_key:
        merged_models.extend(existing_by_key.values())

    merged_keys = [str(row.get("key", "")).strip() for row in merged_models]
    current_keys = [str(row.get("key", "")).strip() for row in models]
    if changed or merged_keys != current_keys or len(merged_models) != len(models):
        save_registry(merged_models)
    return merged_models


def save_registry(models: list[dict]) -> None:
    """Save model registry entries.
    
    Args:
        models: Model registry entries.
    """
    ensure_registry_file()
    MODEL_REGISTRY_PATH.write_text(json.dumps({"models": models}, indent=2), encoding="utf-8")


def validate_registry(models: list[dict]) -> list[str]:
    """Validate registry.
    
    Args:
        models: Model registry entries.
    """
    errors: list[str] = []
    keys = [str(row.get("key", "")).strip() for row in models]
    nonempty = [k for k in keys if k]
    if len(nonempty) != len(set(nonempty)):
        errors.append("Duplicate model keys are not allowed.")
    for idx, row in enumerate(models):
        depth_label = str(row.get("depth_label", "")).strip().lower()
        if depth_label not in {"shallow", "deep"}:
            errors.append(f"Row {idx + 1}: depth_label must be 'shallow' or 'deep'.")
        if not str(row.get("model_key", "")).strip():
            errors.append(f"Row {idx + 1}: model_key is required.")
    return errors


def discover_checkpoints(model_folder_name: str) -> list[str]:
    """Discover checkpoints.
    
    Args:
        model_folder_name: Name used for lookup or display.
    """
    folder = SAVED_MODELS_DIR / model_folder_name
    if not folder.exists():
        return []
    checkpoints = sorted(folder.glob("*.pt"))
    return [str(p.relative_to(REPO_ROOT)) for p in checkpoints]


def absolute_path_from_repo(relative_or_absolute: str) -> Path:
    """Return the absolute path from repo.
    
    Args:
        relative_or_absolute: Relative repository path or absolute path.
    """
    p = Path(relative_or_absolute)
    if p.is_absolute():
        return p
    return (REPO_ROOT / p).resolve()
