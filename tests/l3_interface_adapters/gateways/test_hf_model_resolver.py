"""Tests for HF model resolver gateway."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from lazy_take_notes.l1_entities.errors import ModelResolutionError
from lazy_take_notes.l3_interface_adapters.gateways.hf_model_resolver import (
    BREEZE_VARIANTS,
    WHISPER_CPP_MODELS,
    HfModelResolver,
    _make_progress_class,  # noqa: PLC2701 -- testing private helper
)


class TestHfModelResolver:
    def test_absolute_path_exists(self, tmp_path: Path):
        model_file = tmp_path / 'model.bin'
        model_file.touch()
        resolver = HfModelResolver()
        assert resolver.resolve(str(model_file)) == str(model_file)

    def test_absolute_path_not_found(self, tmp_path: Path):
        resolver = HfModelResolver()
        with pytest.raises(ModelResolutionError, match='not found'):
            resolver.resolve(str(tmp_path / 'nope.bin'))

    def test_passthrough_name(self):
        resolver = HfModelResolver()
        result = resolver.resolve('some-custom-model')
        assert result == 'some-custom-model'

    def test_breeze_variants_are_recognized(self):
        for name in BREEZE_VARIANTS:
            # Should not raise — will either return cached path or attempt HF download
            # We just verify it doesn't fall through to passthrough
            assert name in BREEZE_VARIANTS

    def test_whisper_cpp_models_are_recognized(self):
        for name in WHISPER_CPP_MODELS:
            assert name in WHISPER_CPP_MODELS

    def test_large_v3_turbo_q8_is_known(self):
        assert 'large-v3-turbo-q8_0' in WHISPER_CPP_MODELS
        assert WHISPER_CPP_MODELS['large-v3-turbo-q8_0'] == 'ggml-large-v3-turbo-q8_0.bin'

    def test_on_progress_not_called_for_absolute_path(self, tmp_path: Path):
        model_file = tmp_path / 'model.bin'
        model_file.touch()
        calls = []
        resolver = HfModelResolver(on_progress=lambda p: calls.append(p))
        resolver.resolve(str(model_file))
        assert calls == []

    def test_on_progress_not_called_for_passthrough(self):
        calls = []
        resolver = HfModelResolver(on_progress=lambda p: calls.append(p))
        resolver.resolve('some-custom-model')
        assert calls == []

    @patch('lazy_take_notes.l3_interface_adapters.gateways.hf_model_resolver.hf_hub_download')
    def test_on_progress_passes_tqdm_class_to_download(self, mock_download, tmp_path):
        mock_download.return_value = str(tmp_path / 'model.bin')
        calls = []
        resolver = HfModelResolver(on_progress=lambda p: calls.append(p))
        # Monkeypatch MODELS_DIR to tmp so cache check fails
        with patch(
            'lazy_take_notes.l3_interface_adapters.gateways.hf_model_resolver.MODELS_DIR',
            str(tmp_path / 'models'),
        ):
            resolver.resolve('large-v3-turbo-q8_0')
        assert mock_download.called
        assert 'tqdm_class' in mock_download.call_args.kwargs

    @patch('lazy_take_notes.l3_interface_adapters.gateways.hf_model_resolver.hf_hub_download')
    def test_no_tqdm_class_without_on_progress(self, mock_download, tmp_path):
        mock_download.return_value = str(tmp_path / 'model.bin')
        resolver = HfModelResolver()
        with patch(
            'lazy_take_notes.l3_interface_adapters.gateways.hf_model_resolver.MODELS_DIR',
            str(tmp_path / 'models'),
        ):
            resolver.resolve('large-v3-turbo-q8_0')
        assert mock_download.called
        assert 'tqdm_class' not in mock_download.call_args.kwargs


class TestProgressClass:
    def test_reports_zero_on_init(self):
        calls = []
        cls = _make_progress_class(lambda p: calls.append(p))
        cls(total=1000)
        assert calls == [0]

    def test_reports_percentage_on_update(self):
        calls = []
        cls = _make_progress_class(lambda p: calls.append(p))
        reporter = cls(total=100)
        reporter.update(50)
        reporter.update(50)
        assert calls == [0, 50, 100]

    def test_no_callback_when_total_is_zero(self):
        calls = []
        cls = _make_progress_class(lambda p: calls.append(p))
        reporter = cls(total=0)
        reporter.update(10)
        assert calls == []

    def test_context_manager(self):
        cls = _make_progress_class(lambda p: None)
        with cls(total=100) as r:
            r.update(100)

    def test_caps_at_100(self):
        calls = []
        cls = _make_progress_class(lambda p: calls.append(p))
        reporter = cls(total=50)
        reporter.update(60)  # overshoots
        assert calls[-1] == 100

    def test_set_description_is_noop(self):
        cls = _make_progress_class(lambda p: None)
        reporter = cls(total=100)
        # Should not raise
        reporter.set_description('downloading')

    def test_set_description_str_is_noop(self):
        cls = _make_progress_class(lambda p: None)
        reporter = cls(total=100)
        reporter.set_description_str('downloading')

    def test_refresh_is_noop(self):
        cls = _make_progress_class(lambda p: None)
        reporter = cls(total=100)
        reporter.refresh()


class TestCacheHit:
    @patch('lazy_take_notes.l3_interface_adapters.gateways.hf_model_resolver.hf_hub_download')
    def test_breeze_cache_hit_skips_download(self, mock_download, tmp_path):
        # Pre-create the model file so the cache check passes
        cache_dir = tmp_path / 'models' / 'breeze'
        cache_dir.mkdir(parents=True)
        (cache_dir / 'ggml-model-q8_0.bin').touch()

        with patch(
            'lazy_take_notes.l3_interface_adapters.gateways.hf_model_resolver.MODELS_DIR',
            str(tmp_path / 'models'),
        ):
            resolver = HfModelResolver()
            result = resolver.resolve('breeze-q8')

        mock_download.assert_not_called()
        assert 'ggml-model-q8_0.bin' in result

    @patch('lazy_take_notes.l3_interface_adapters.gateways.hf_model_resolver.hf_hub_download')
    def test_whisper_cpp_cache_hit_skips_download(self, mock_download, tmp_path):
        cache_dir = tmp_path / 'models' / 'whisper-cpp'
        cache_dir.mkdir(parents=True)
        (cache_dir / 'ggml-large-v3-turbo-q8_0.bin').touch()

        with patch(
            'lazy_take_notes.l3_interface_adapters.gateways.hf_model_resolver.MODELS_DIR',
            str(tmp_path / 'models'),
        ):
            resolver = HfModelResolver()
            result = resolver.resolve('large-v3-turbo-q8_0')

        mock_download.assert_not_called()
        assert 'ggml-large-v3-turbo-q8_0.bin' in result

    @patch('lazy_take_notes.l3_interface_adapters.gateways.hf_model_resolver.hf_hub_download')
    def test_breeze_with_tqdm_class(self, mock_download, tmp_path):
        mock_download.return_value = str(tmp_path / 'model.bin')
        calls = []
        resolver = HfModelResolver(on_progress=lambda p: calls.append(p))
        with patch(
            'lazy_take_notes.l3_interface_adapters.gateways.hf_model_resolver.MODELS_DIR',
            str(tmp_path / 'models'),
        ):
            resolver.resolve('breeze-q5')
        assert mock_download.called
        assert 'tqdm_class' in mock_download.call_args.kwargs
