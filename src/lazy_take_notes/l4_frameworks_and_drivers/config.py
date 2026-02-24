"""Infrastructure provider configs — lives in L4, not domain."""

from __future__ import annotations

import copy

from pydantic import BaseModel, Field

from lazy_take_notes.l1_entities.config import AppConfig
from lazy_take_notes.l3_interface_adapters.gateways.yaml_config_loader import deep_merge

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
    'output': {
        'directory': './output',
        'save_audio': True,
    },
}


def build_app_config(raw: dict) -> AppConfig:
    """Merge *raw* user overrides on top of defaults, then validate."""
    merged = copy.deepcopy(APP_CONFIG_DEFAULTS)
    deep_merge(merged, raw)
    return AppConfig.model_validate(merged)


class OllamaProviderConfig(BaseModel):
    host: str = 'http://localhost:11434'


class OpenAIProviderConfig(BaseModel):
    api_key: str | None = None  # None → SDK reads OPENAI_API_KEY env
    base_url: str = 'https://api.openai.com/v1'


class InfraConfig(BaseModel):
    """Groups all provider-specific settings outside the domain layer."""

    llm_provider: str = 'ollama'  # 'ollama' | 'openai'
    ollama: OllamaProviderConfig = Field(default_factory=OllamaProviderConfig)
    openai: OpenAIProviderConfig = Field(default_factory=OpenAIProviderConfig)
