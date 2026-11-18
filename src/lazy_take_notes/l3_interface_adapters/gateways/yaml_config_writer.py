"""Gateway: YAML configuration writer — persists config dict back to disk."""

from __future__ import annotations

import os
from pathlib import Path

import yaml

from lazy_take_notes.l3_interface_adapters.gateways.paths import CONFIG_DIR, DEFAULT_CONFIG_PATHS
from lazy_take_notes.l3_interface_adapters.gateways.yaml_config_loader import deep_merge


def config_file_path() -> Path:
    """Return the canonical config file path (first entry in DEFAULT_CONFIG_PATHS)."""
    return DEFAULT_CONFIG_PATHS[0]


def write_config(data: dict, path: Path | None = None, owned: tuple[tuple[str, ...], ...] = ()) -> Path:
    """Merge *data* into the config file (or create it) and write back.

    *path* is the file given with --config; without it, the default location is used.
    Each key path in *owned* belongs to the form: its value in *data* replaces the
    file's value, and a key missing from *data* is deleted. A merge alone would
    keep a language override or a limit the user removed. Every other key, such
    as a plugin's settings, merges as before.
    Returns the path that was written.
    """
    if path is None:
        path = _resolve_existing_or_default()
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    existing = _read_existing(path)
    deep_merge(existing, data)
    for keys in owned:
        _replace(existing, data, keys)
    text = yaml.dump(existing, default_flow_style=False, allow_unicode=True, sort_keys=False)
    # The file can hold an API key, so only the user may read it. Writing in place
    # keeps a symlinked config.yaml (dotfiles) a symlink.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, 'w', encoding='utf-8') as handle:
        handle.write(text)
    path.chmod(0o600)  # a file from an older version was created readable by everyone
    return path


def _replace(existing: dict, data: dict, keys: tuple[str, ...]) -> None:
    *parents, last = keys
    target, source = existing, data
    for key in parents:
        target = target.get(key)
        source = source.get(key) if isinstance(source, dict) else None
        if not isinstance(target, dict):
            return
    if isinstance(source, dict) and last in source:
        target[last] = source[last]
    else:
        target.pop(last, None)


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
