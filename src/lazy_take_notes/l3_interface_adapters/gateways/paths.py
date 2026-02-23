"""Shared path constants for configuration and user templates."""

from __future__ import annotations

from platformdirs import user_config_path

CONFIG_DIR = user_config_path('lazy-take-notes')
USER_TEMPLATES_DIR = CONFIG_DIR / 'templates'

CONSENT_NOTICED_PATH = CONFIG_DIR / '.consent_noticed'

DEFAULT_CONFIG_PATHS = [
    CONFIG_DIR / 'config.yaml',
    CONFIG_DIR / 'config.yml',
]
