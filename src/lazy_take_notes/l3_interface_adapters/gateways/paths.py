"""Shared path constants for configuration and user templates."""

from __future__ import annotations

from platformdirs import user_config_path

CONFIG_DIR = user_config_path('lazy-take-notes')
USER_TEMPLATES_DIR = CONFIG_DIR / 'templates'

CONSENT_NOTICED_PATH = CONFIG_DIR / '.consent_noticed'
BUILTIN_TEMPLATES_NOTICED_PATH = CONFIG_DIR / '.builtin_templates_noticed'

PLUGINS_YAML = CONFIG_DIR / 'plugins.yaml'
PLUGINS_TXT = CONFIG_DIR / 'plugins.txt'

DEFAULT_CONFIG_PATHS = [
    CONFIG_DIR / 'config.yaml',
    CONFIG_DIR / 'config.yml',
]
