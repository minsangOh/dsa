"""Application-specific filesystem utilities."""
from __future__ import annotations

import os
import sys
from pathlib import Path


APP_NAME = "Neam Coin Auto Trading Bot"


def get_app_storage_dir() -> Path:
    """Return a writable directory for persistent app data."""
    if sys.platform == "darwin":
        base_dir = Path.home() / "Library" / "Application Support"
    elif sys.platform.startswith("win"):
        base = os.environ.get("APPDATA")
        base_dir = Path(base) if base else Path.home()
    else:
        base = os.environ.get("XDG_DATA_HOME")
        base_dir = Path(base) if base else Path.home() / ".local" / "share"

    storage_dir = base_dir / APP_NAME
    try:
        storage_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"Error: Could not create storage directory at {storage_dir}. {e}", file=sys.stderr)
        sys.exit(1)
    return storage_dir


def resolve_data_path(file_name: str) -> Path:
    """Return the fully qualified path for a data file."""
    return get_app_storage_dir() / file_name
