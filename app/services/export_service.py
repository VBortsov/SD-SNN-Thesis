from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from app.services.paths import APP_EXPORTS_DIR, ensure_app_dirs


def _stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def export_dataframe(df: pd.DataFrame, base_name: str) -> Path:
    """Export dataframe.
    
    Args:
        df: Input dataframe.
        base_name: Name used for lookup or display.
    """
    ensure_app_dirs()
    path = APP_EXPORTS_DIR / f"{base_name}_{_stamp()}.csv"
    df.to_csv(path, index=False)
    return path


def export_json(payload: dict, base_name: str) -> Path:
    """Export json.
    
    Args:
        payload: Data to write.
        base_name: Name used for lookup or display.
    """
    ensure_app_dirs()
    path = APP_EXPORTS_DIR / f"{base_name}_{_stamp()}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def export_figure(fig, base_name: str) -> Path:
    """Export figure.
    
    Args:
        fig: Matplotlib figure to export.
        base_name: Name used for lookup or display.
    """
    ensure_app_dirs()
    path = APP_EXPORTS_DIR / f"{base_name}_{_stamp()}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    return path


def create_export_bundle_dir(base_name: str) -> Path:
    """Create export bundle dir.
    
    Args:
        base_name: Name used for lookup or display.
    """
    ensure_app_dirs()
    path = APP_EXPORTS_DIR / f"{base_name}_{_stamp()}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def export_dataframe_to_dir(df: pd.DataFrame, path: Path) -> Path:
    """Export dataframe to dir.
    
    Args:
        df: Input dataframe.
        path: File or directory path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def export_json_to_dir(payload: dict, path: Path) -> Path:
    """Export json to dir.
    
    Args:
        path: File or directory path.
        payload: Data to write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def export_figure_to_dir(fig, path: Path, dpi: int = 300) -> Path:
    """Export figure to dir.
    
    Args:
        path: File or directory path.
        fig: Matplotlib figure to export.
        dpi: Export resolution.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path
