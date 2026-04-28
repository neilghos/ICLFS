from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


@lru_cache(maxsize=4)
def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config at {path} must be a YAML mapping.")
    return data


def get_augmentor_config(config_path: str | Path | None = None) -> dict[str, Any]:
    config = load_config(config_path)
    augmentor = config.get("augmentor", {})
    if not isinstance(augmentor, dict):
        raise ValueError("config.yaml: 'augmentor' must be a mapping.")
    return augmentor
