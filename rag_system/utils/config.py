"""
Shared configuration loader for config.yaml.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent / "config.yaml"


def load_config(config_path: Path | str | None = None) -> Dict[str, Any]:
    """Load YAML config from *config_path* (defaults to repo-root config.yaml)."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    try:
        import yaml

        if path.exists():
            with open(path, encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
    except Exception:
        pass
    return {}
