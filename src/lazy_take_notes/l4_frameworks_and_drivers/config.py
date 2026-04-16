"""Infrastructure provider configs — lives in L4, not domain."""

from __future__ import annotations

import copy

from pydantic import BaseModel, ConfigDict, Field

from lazy_take_notes.l1_entities.config import AppConfig
from lazy_take_notes.l3_interface_adapters.gateways.yaml_config_loader import deep_merge

APP_CONFIG_DEFAULTS: dict = {
    'transcription': {
        'model': 'hf://ggerganov/whisper.cpp/ggml-large-v3-turbo-q8_0.bin',
        'models': {'zh': 'hf://alan314159/Breeze-ASR-25-whispercpp/ggml-model-q8_0.bin'},
        'chunk_duration': 25.0,
        'overlap': 1.0,
        'silence_threshold': 0.01,
        'pause_duration': 1.5,
    },
    'digest': {
        'model': 'gpt-oss:20b',
        'min_lines': 15,
        'min_interval': 60.0,
        'compact_token_threshold': 100_000,
    },
    'interactive': {
        'model': 'gpt-oss:20b',
    },
    'output': {
        'directory': './output',
        'save_audio': True,
        'save_notes_history': True,
        'save_context': True,
        'save_debug_log': False,
        'auto_label': True,
    },
    'recognition_hints': [],
}


def build_app_config(raw: dict) -> AppConfig:
    """Merge *raw* user overrides on top of defaults, then validate.

    Known short aliases in transcription model fields are expanded to
    ``hf://`` URIs so users always see the full HuggingFace origin.
    """
    merged = copy.deepcopy(APP_CONFIG_DEFAULTS)
    deep_merge(merged, raw)
    _expand_transcription_aliases(merged)
    return AppConfig.model_validate(merged)


def _expand_transcription_aliases(cfg: dict) -> None:
    """Expand known short model aliases to hf:// URIs in-place."""
    from lazy_take_notes.l3_interface_adapters.gateways.hf_model_resolver import (  # noqa: PLC0415 -- deferred: avoid import at module level
        expand_model_alias,
    )

    trans = cfg.get('transcription', {})
    if 'model' in trans:
        trans['model'] = expand_model_alias(trans['model'])
    for locale, name in trans.get('models', {}).items():
        trans['models'][locale] = expand_model_alias(name)


class OllamaProviderConfig(BaseModel):
    host: str = 'http://localhost:11434'


class OpenAIProviderConfig(BaseModel):
    api_key: str | None = None  # None → SDK reads OPENAI_API_KEY env
    base_url: str = 'https://api.openai.com/v1'


class InfraConfig(BaseModel):
    """Groups all provider-specific settings outside the domain layer.

    Plugin providers can store custom config under arbitrary keys
    (e.g. ``claude_code: {digest_model: sonnet}``). These are accessible
    via ``infra.model_extra``.
    """

    model_config = ConfigDict(extra='allow')

    llm_provider: str = 'ollama'  # 'ollama' | 'openai' | plugin-registered name
    transcription_provider: str = 'whisper-cpp'
    theme: str = 'textual-dark'  # Textual built-in theme name
    show_builtin_templates: bool = True
    ollama: OllamaProviderConfig = Field(default_factory=OllamaProviderConfig)
    openai: OpenAIProviderConfig = Field(default_factory=OpenAIProviderConfig)


DEFAULT_THEME = 'textual-dark'


def load_theme() -> str:
    """Read the saved theme from config.yaml, defaulting to DEFAULT_THEME."""
    from lazy_take_notes.l3_interface_adapters.gateways.yaml_config_loader import (  # noqa: PLC0415 -- deferred: avoid circular import at module level
        YamlConfigLoader,
    )

    try:
        raw = YamlConfigLoader().load()
        return raw.get('theme', DEFAULT_THEME)
    except FileNotFoundError:
        return DEFAULT_THEME
