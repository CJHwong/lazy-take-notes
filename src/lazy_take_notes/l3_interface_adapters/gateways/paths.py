"""Shared path constants for configuration, templates, and the model cache."""

from __future__ import annotations

from pathlib import Path

from platformdirs import user_cache_path, user_config_path

CONFIG_DIR = user_config_path('lazy-take-notes')
USER_TEMPLATES_DIR = CONFIG_DIR / 'templates'

CACHE_DIR = user_cache_path('lazy-take-notes')
LEGACY_NEMOTRON_STORE = Path.home() / '.cache' / 'nemotron-diarization'

CONSENT_NOTICED_PATH = CONFIG_DIR / '.consent_noticed'
BUILTIN_TEMPLATES_NOTICED_PATH = CONFIG_DIR / '.builtin_templates_noticed'

PLUGINS_YAML = CONFIG_DIR / 'plugins.yaml'
PLUGINS_TXT = CONFIG_DIR / 'plugins.txt'

DEFAULT_CONFIG_PATHS = [
    CONFIG_DIR / 'config.yaml',
    CONFIG_DIR / 'config.yml',
]
