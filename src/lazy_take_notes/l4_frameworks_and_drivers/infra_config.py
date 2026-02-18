"""Infrastructure provider configs — lives in L4, not domain."""

from __future__ import annotations

import copy

from pydantic import BaseModel, Field

from lazy_take_notes.l1_entities.config import AppConfig

APP_CONFIG_DEFAULTS: dict = {
    'transcription': {
        'model': 'large-v3-turbo-q8_0',
        'models': {'zh': 'breeze-q8'},
        'chunk_duration': 25.0,
        'overlap': 1.0,
        'silence_threshold': 0.01,
        'pause_duration': 1.5,
    },
    'digest': {
        'model': 'gpt-oss:120b-cloud',
        'min_lines': 15,
        'min_interval': 60.0,
        'compact_token_threshold': 100_000,
    },
    'interactive': {
        'model': 'gpt-oss:20b-cloud',
    },
    'template': 'default_zh_tw',
    'output': {
        'directory': './output',
        'save_audio': True,
    },
}


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* (mutates base)."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def build_app_config(raw: dict) -> AppConfig:
    """Merge *raw* user overrides on top of defaults, then validate."""
    merged = copy.deepcopy(APP_CONFIG_DEFAULTS)
    deep_merge(merged, raw)
    return AppConfig.model_validate(merged)


class OllamaProviderConfig(BaseModel):
    host: str = 'http://localhost:11434'


class InfraConfig(BaseModel):
    """Groups all provider-specific settings outside the domain layer."""

    ollama: OllamaProviderConfig = Field(default_factory=OllamaProviderConfig)
