"""Shared path constants for configuration and user templates."""

from __future__ import annotations

from pathlib import Path

CONFIG_DIR = Path('~/.config/lazy-take-notes').expanduser()
USER_TEMPLATES_DIR = CONFIG_DIR / 'templates'

DEFAULT_CONFIG_PATHS = [
    CONFIG_DIR / 'config.yaml',
    CONFIG_DIR / 'config.yml',
]
