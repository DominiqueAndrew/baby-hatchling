"""YAML configuration helpers."""
from __future__ import annotations

import pathlib
from typing import Any, Dict

import yaml


def load_config(path: str | pathlib.Path) -> Dict[str, Any]:
    """Loads a YAML config file into a nested dict with string keys."""

    with open(path, "r", encoding="utf8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):  # pragma: no cover - defensive
        raise ValueError(f"Config at {path} is not a mapping")
    return cfg


def merge_overrides(cfg: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merges overrides into cfg (in-place)."""

    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(cfg.get(key), dict):
            merge_overrides(cfg[key], value)
        else:
            cfg[key] = value
    return cfg
