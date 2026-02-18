"""Tests for configuration Pydantic models — schema validation only."""

import pytest
from pydantic import ValidationError

from lazy_take_notes.l1_entities.config import (
    AppConfig,
    DigestConfig,
    InteractiveConfig,
    OutputConfig,
    TranscriptionConfig,
)


class TestTranscriptionConfig:
    def test_valid(self):
        cfg = TranscriptionConfig(
            model='m',
            chunk_duration=10.0,
            overlap=1.0,
            silence_threshold=0.01,
            pause_duration=1.5,
        )
        assert cfg.model == 'm'

    def test_models_defaults_to_empty(self):
        cfg = TranscriptionConfig(
            model='m',
            chunk_duration=10.0,
            overlap=1.0,
            silence_threshold=0.01,
            pause_duration=1.5,
        )
        assert cfg.models == {}

    def test_missing_field_raises(self):
        with pytest.raises(ValidationError):
            TranscriptionConfig(model='m')  # type: ignore[call-arg]

    def test_wrong_type_raises(self):
        with pytest.raises(ValidationError):
            TranscriptionConfig(
                model='m',
                chunk_duration='not_a_float',  # type: ignore[invalid-argument-type]
                overlap=1.0,
                silence_threshold=0.01,
                pause_duration=1.5,
            )


class TestModelForLocale:
    def _cfg(self, **kwargs) -> TranscriptionConfig:
        defaults = dict(
            model='default-model', chunk_duration=10.0, overlap=1.0, silence_threshold=0.01, pause_duration=1.5
        )
        defaults.update(kwargs)
        return TranscriptionConfig(**defaults)  # type: ignore[invalid-argument-type]

    def test_exact_match(self):
        cfg = self._cfg(models={'zh-tw': 'breeze-q8'})
        assert cfg.model_for_locale('zh-TW') == 'breeze-q8'

    def test_prefix_match(self):
        cfg = self._cfg(models={'zh': 'breeze-q8'})
        assert cfg.model_for_locale('zh-TW') == 'breeze-q8'

    def test_fallback_to_default(self):
        cfg = self._cfg(models={'zh': 'breeze-q8'})
        assert cfg.model_for_locale('en') == 'default-model'

    def test_exact_takes_priority_over_prefix(self):
        cfg = self._cfg(models={'zh': 'generic-zh', 'zh-tw': 'breeze-q8'})
        assert cfg.model_for_locale('zh-TW') == 'breeze-q8'

    def test_empty_models_falls_back(self):
        cfg = self._cfg()
        assert cfg.model_for_locale('en') == 'default-model'


class TestDigestConfig:
    def test_valid(self):
        cfg = DigestConfig(
            model='m',
            min_lines=10,
            min_interval=30.0,
            compact_token_threshold=50_000,
        )
        assert cfg.min_lines == 10

    def test_missing_field_raises(self):
        with pytest.raises(ValidationError):
            DigestConfig(model='m', min_lines=10)  # type: ignore[call-arg]


class TestInteractiveConfig:
    def test_valid(self):
        cfg = InteractiveConfig(model='m')
        assert cfg.model == 'm'

    def test_missing_field_raises(self):
        with pytest.raises(ValidationError):
            InteractiveConfig()  # type: ignore[call-arg]


class TestOutputConfig:
    def test_valid(self):
        cfg = OutputConfig(directory='./out', save_audio=False)
        assert cfg.save_audio is False

    def test_missing_field_raises(self):
        with pytest.raises(ValidationError):
            OutputConfig(directory='./out')  # type: ignore[call-arg]


class TestAppConfig:
    def test_valid(self):
        cfg = AppConfig(
            transcription=TranscriptionConfig(
                model='m',
                chunk_duration=10.0,
                overlap=1.0,
                silence_threshold=0.01,
                pause_duration=1.5,
            ),
            digest=DigestConfig(
                model='m',
                min_lines=10,
                min_interval=30.0,
                compact_token_threshold=50_000,
            ),
            interactive=InteractiveConfig(model='m'),
            template='default_en',
            output=OutputConfig(directory='./out', save_audio=True),
        )
        assert cfg.template == 'default_en'

    def test_missing_top_level_field_raises(self):
        with pytest.raises(ValidationError):
            AppConfig(template='x')  # type: ignore[call-arg]

    def test_no_defaults(self):
        """AppConfig has no defaults — bare construction must fail."""
        with pytest.raises(ValidationError):
            AppConfig()  # type: ignore[call-arg]
