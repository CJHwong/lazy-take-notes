"""Tests for L4 infra config defaults and build_app_config factory."""

from __future__ import annotations

from unittest.mock import patch

from lazy_take_notes.l4_frameworks_and_drivers.config import (
    APP_CONFIG_DEFAULTS,
    InfraConfig,
    build_app_config,
    deep_merge,
    load_theme,
)


class TestBuildAppConfig:
    def test_defaults_produce_valid_config(self):
        cfg = build_app_config({})
        assert cfg.transcription.model == 'hf://ggerganov/whisper.cpp/ggml-large-v3-turbo-q8_0.bin'
        assert cfg.transcription.models == {'zh': 'hf://alan314159/Breeze-ASR-25-whispercpp/ggml-model-q8_0.bin'}
        assert cfg.transcription.chunk_duration == 25.0
        assert cfg.transcription.overlap == 1.0
        assert cfg.transcription.silence_threshold == 0.01
        assert cfg.transcription.pause_duration == 1.5
        assert cfg.digest.model == 'gpt-oss:20b'
        assert cfg.digest.min_lines == 15
        assert cfg.digest.min_interval == 60.0
        assert cfg.digest.compact_token_threshold == 100_000
        assert cfg.interactive.model == 'gpt-oss:20b'
        assert cfg.output.directory == './output'
        assert cfg.output.save_audio is True
        assert cfg.output.save_notes_history is True
        assert cfg.output.save_context is True
        assert cfg.output.save_debug_log is False

    def test_user_overrides_take_precedence(self):
        cfg = build_app_config(
            {
                'transcription': {'model': 'custom-model', 'models': {'ja': 'ja-model'}},
                'digest': {'min_lines': 5},
            }
        )
        assert cfg.transcription.model == 'custom-model'
        assert cfg.transcription.models == {
            'zh': 'hf://alan314159/Breeze-ASR-25-whispercpp/ggml-model-q8_0.bin',
            'ja': 'ja-model',
        }
        assert cfg.transcription.chunk_duration == 25.0  # default preserved
        assert cfg.digest.min_lines == 5
        assert cfg.digest.model == 'gpt-oss:20b'  # default preserved

    def test_old_alias_expanded_to_hf_uri(self):
        """Old short aliases in user config get auto-converted to hf:// URIs."""
        cfg = build_app_config({'transcription': {'model': 'large-v3-turbo-q8_0', 'models': {'zh': 'breeze-q8'}}})
        assert cfg.transcription.model == 'hf://ggerganov/whisper.cpp/ggml-large-v3-turbo-q8_0.bin'
        assert cfg.transcription.models['zh'] == 'hf://alan314159/Breeze-ASR-25-whispercpp/ggml-model-q8_0.bin'

    def test_unknown_model_name_passes_through(self):
        """Non-alias model names (custom paths, unknown names) are left unchanged."""
        cfg = build_app_config({'transcription': {'model': 'my-custom-model', 'models': {'ja': '/abs/path.bin'}}})
        assert cfg.transcription.model == 'my-custom-model'
        assert cfg.transcription.models['ja'] == '/abs/path.bin'

    def test_hf_uri_not_double_expanded(self):
        """hf:// URIs are not modified by expansion."""
        uri = 'hf://my-org/my-repo/my-model.bin'
        cfg = build_app_config({'transcription': {'model': uri}})
        assert cfg.transcription.model == uri

    def test_build_does_not_mutate_defaults(self):
        import copy

        snapshot = copy.deepcopy(APP_CONFIG_DEFAULTS)
        build_app_config({'transcription': {'model': 'mutant'}})
        assert APP_CONFIG_DEFAULTS == snapshot


class TestDeepMerge:
    def test_nested_merge(self):
        base = {'a': {'x': 1, 'y': 2}, 'b': 3}
        override = {'a': {'y': 99, 'z': 100}, 'c': 4}
        result = deep_merge(base, override)
        assert result == {'a': {'x': 1, 'y': 99, 'z': 100}, 'b': 3, 'c': 4}

    def test_override_replaces_non_dict(self):
        base = {'a': 'old'}
        result = deep_merge(base, {'a': 'new'})
        assert result['a'] == 'new'

    def test_override_dict_over_non_dict(self):
        base = {'a': 'scalar'}
        result = deep_merge(base, {'a': {'nested': True}})
        assert result['a'] == {'nested': True}


class TestInfraConfigTheme:
    def test_default_theme(self):
        cfg = InfraConfig()
        assert cfg.theme == 'textual-dark'

    def test_custom_theme(self):
        cfg = InfraConfig(theme='textual-light')
        assert cfg.theme == 'textual-light'


class TestInfraConfigTranscriptionProvider:
    def test_default_transcription_provider(self):
        cfg = InfraConfig()
        assert cfg.transcription_provider == 'whisper-cpp'

    def test_custom_transcription_provider(self):
        cfg = InfraConfig(transcription_provider='my-stt-plugin')
        assert cfg.transcription_provider == 'my-stt-plugin'


class TestLoadTheme:
    @patch(
        'lazy_take_notes.l3_interface_adapters.gateways.yaml_config_loader.YamlConfigLoader',
    )
    def test_returns_theme_from_config(self, mock_loader_cls):
        mock_loader_cls.return_value.load.return_value = {'theme': 'nord'}
        assert load_theme() == 'nord'

    @patch(
        'lazy_take_notes.l3_interface_adapters.gateways.yaml_config_loader.YamlConfigLoader',
    )
    def test_returns_default_when_key_missing(self, mock_loader_cls):
        mock_loader_cls.return_value.load.return_value = {'llm_provider': 'ollama'}
        assert load_theme() == 'textual-dark'

    @patch(
        'lazy_take_notes.l3_interface_adapters.gateways.yaml_config_loader.YamlConfigLoader',
    )
    def test_returns_default_on_file_not_found(self, mock_loader_cls):
        mock_loader_cls.return_value.load.side_effect = FileNotFoundError
        assert load_theme() == 'textual-dark'
