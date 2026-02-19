"""Gateway: YAML configuration loader — implements ConfigLoader port."""

from __future__ import annotations

from pathlib import Path

import yaml

from lazy_take_notes.l1_entities.config import AppConfig
from lazy_take_notes.l3_interface_adapters.gateways.paths import DEFAULT_CONFIG_PATHS


class YamlConfigLoader:
    """Loads AppConfig from YAML files with merge and override support."""

    def load(
        self,
        config_path: str | None = None,
        overrides: dict | None = None,
    ) -> AppConfig:
        data = _load_data(config_path, overrides)
        return AppConfig.model_validate(data)

    def load_raw(
        self,
        config_path: str | None = None,
        overrides: dict | None = None,
    ) -> dict:
        """Return the merged YAML data as a raw dict (before Pydantic validation)."""
        return _load_data(config_path, overrides)


def _load_data(
    config_path: str | None = None,
    overrides: dict | None = None,
) -> dict:
    """Resolve, read, and merge YAML config into a plain dict."""
    data: dict = {}
    if config_path is not None:
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f'Config file not found: {path}')
        data = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
    else:
        for default_path in DEFAULT_CONFIG_PATHS:
            if default_path.exists():
                data = yaml.safe_load(default_path.read_text(encoding='utf-8')) or {}
                break
    if overrides:
        deep_merge(data, overrides)
    return data


def deep_merge(base: dict, override: dict) -> dict:
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            deep_merge(base[key], value)
        else:
            base[key] = value
    return base
