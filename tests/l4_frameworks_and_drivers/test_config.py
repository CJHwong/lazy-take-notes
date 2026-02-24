"""Tests for L4 infra config defaults and build_app_config factory."""

from __future__ import annotations

from lazy_take_notes.l4_frameworks_and_drivers.config import (
    APP_CONFIG_DEFAULTS,
    build_app_config,
    deep_merge,
)


class TestBuildAppConfig:
    def test_defaults_produce_valid_config(self):
        cfg = build_app_config({})
        assert cfg.transcription.model == 'large-v3-turbo-q8_0'
        assert cfg.transcription.models == {'zh': 'breeze-q8'}
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

    def test_user_overrides_take_precedence(self):
        cfg = build_app_config(
            {
                'transcription': {'model': 'custom-model', 'models': {'ja': 'ja-model'}},
                'digest': {'min_lines': 5},
            }
        )
        assert cfg.transcription.model == 'custom-model'
        assert cfg.transcription.models == {'zh': 'breeze-q8', 'ja': 'ja-model'}
        assert cfg.transcription.chunk_duration == 25.0  # default preserved
        assert cfg.digest.min_lines == 5
        assert cfg.digest.model == 'gpt-oss:20b'  # default preserved

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
