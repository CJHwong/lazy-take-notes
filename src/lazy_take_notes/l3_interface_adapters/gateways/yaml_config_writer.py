"""Gateway: YAML configuration writer — persists config dict back to disk."""

from __future__ import annotations

from pathlib import Path

import yaml

from lazy_take_notes.l3_interface_adapters.gateways.paths import CONFIG_DIR, DEFAULT_CONFIG_PATHS
from lazy_take_notes.l3_interface_adapters.gateways.yaml_config_loader import deep_merge


def config_file_path() -> Path:
    """Return the canonical config file path (first entry in DEFAULT_CONFIG_PATHS)."""
    return DEFAULT_CONFIG_PATHS[0]


def write_config(data: dict, path: Path | None = None) -> Path:
    """Merge *data* into the config file (or create it) and write back.

    *path* is the file given with --config; without it, the default location is used.
    Returns the path that was written.
    """
    if path is None:
        path = _resolve_existing_or_default()
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    existing = _read_existing(path)
    deep_merge(existing, data)
    path.write_text(
        yaml.dump(existing, default_flow_style=False, allow_unicode=True, sort_keys=False),
        encoding='utf-8',
    )
    return path


def _resolve_existing_or_default() -> Path:
    """Find the first existing config file, or fall back to the canonical default."""
    for candidate in DEFAULT_CONFIG_PATHS:
        if candidate.exists():
            return candidate
    return DEFAULT_CONFIG_PATHS[0]


def _read_existing(path: Path) -> dict:
    """Read existing YAML config, returning empty dict if file doesn't exist or is empty."""
    if not path.exists():
        return {}
    content = yaml.safe_load(path.read_text(encoding='utf-8'))
    return content if isinstance(content, dict) else {}
