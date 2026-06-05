from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
APP_ROOT = REPO_ROOT / "app"
APP_CONFIG_DIR = APP_ROOT / "config"
APP_DATA_DIR = APP_ROOT / "data"
APP_EXPORTS_DIR = APP_ROOT / "exports"
SAVED_MODELS_DIR = REPO_ROOT / "decomposers" / "ML_methods" / "NN_based" / "saved_models"
REAL_DATA_DIR = REPO_ROOT / "RealData"

RUN_HISTORY_PATH = APP_DATA_DIR / "runs_history.json"
SAMPLE_ANALYSIS_PATH = APP_DATA_DIR / "sample_analysis.json"
MODEL_REGISTRY_PATH = APP_CONFIG_DIR / "model_registry.json"


def ensure_app_dirs() -> None:
    """Ensure app dirs."""
    APP_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    APP_DATA_DIR.mkdir(parents=True, exist_ok=True)
    APP_EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
