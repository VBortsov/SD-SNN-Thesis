from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from app.services.paths import RUN_HISTORY_PATH, SAMPLE_ANALYSIS_PATH, ensure_app_dirs


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def load_runs() -> list[dict]:
    """Load runs."""
    ensure_app_dirs()
    payload = _read_json(RUN_HISTORY_PATH, {"runs": []})
    runs = payload.get("runs", [])
    return runs if isinstance(runs, list) else []


def save_runs(runs: list[dict]) -> None:
    """Save runs.
    
    Args:
        runs: Run records to persist.
    """
    ensure_app_dirs()
    RUN_HISTORY_PATH.write_text(json.dumps({"runs": runs}, indent=2), encoding="utf-8")


def create_run_record(
    model_name: str,
    depth_label: str,
    data_settings: dict,
    hyperparams: dict,
    notes: str = "",
) -> dict:
    """Create run record."""
    return {
        "run_id": uuid.uuid4().hex[:12],
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": model_name,
        "depth_label": depth_label,
        "data_settings": data_settings,
        "hyperparameters": hyperparams,
        "best_validation_metric": None,
        "test_summary": {},
        "checkpoint_path": "",
        "notes": notes,
        "favorite": False,
        "status": "started",
    }


def append_run(run: dict) -> None:
    """Append run.
    
    Args:
        run: Run record to persist.
    """
    runs = load_runs()
    runs.append(run)
    save_runs(runs)


def update_run(run_id: str, updates: dict) -> bool:
    """Update run.
    
    Args:
        run_id: Run identifier.
        updates: Fields to update.
    """
    runs = load_runs()
    updated = False
    for idx, run in enumerate(runs):
        if run.get("run_id") == run_id:
            runs[idx] = {**run, **updates}
            updated = True
            break
    if updated:
        save_runs(runs)
    return updated


def delete_run(run_id: str) -> bool:
    """Delete run.
    
    Args:
        run_id: Run identifier.
    """
    runs = load_runs()
    before = len(runs)
    runs = [run for run in runs if run.get("run_id") != run_id]
    if len(runs) != before:
        save_runs(runs)
        return True
    return False


def toggle_favorite(run_id: str) -> bool:
    """Toggle favorite.
    
    Args:
        run_id: Run identifier.
    """
    runs = load_runs()
    changed = False
    for run in runs:
        if run.get("run_id") == run_id:
            run["favorite"] = not bool(run.get("favorite", False))
            changed = True
            break
    if changed:
        save_runs(runs)
    return changed


def load_sample_analysis() -> list[dict]:
    """Load sample analysis."""
    ensure_app_dirs()
    payload = _read_json(SAMPLE_ANALYSIS_PATH, {"samples": []})
    samples = payload.get("samples", [])
    return samples if isinstance(samples, list) else []


def append_sample_analysis(sample: dict) -> None:
    """Append sample analysis.
    
    Args:
        sample: Sample-analysis record.
    """
    samples = load_sample_analysis()
    samples.append(sample)
    SAMPLE_ANALYSIS_PATH.write_text(json.dumps({"samples": samples}, indent=2), encoding="utf-8")

