"""Tests for YAML config loader gateway."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from lazy_take_notes.l3_interface_adapters.gateways.yaml_config_loader import YamlConfigLoader


class TestYamlConfigLoader:
    def test_load_from_yaml(self, sample_config_yaml: Path):
        loader = YamlConfigLoader()
        cfg = loader.load(str(sample_config_yaml))
        assert cfg.transcription.model == 'breeze-q5'
        assert cfg.transcription.chunk_duration == 8.0
        assert cfg.digest.model == 'llama3:8b'
        assert cfg.digest.min_lines == 10

    def test_load_nonexistent_raises(self, tmp_path: Path):
        loader = YamlConfigLoader()
        with pytest.raises(FileNotFoundError):
            loader.load(str(tmp_path / 'nonexistent.yaml'))

    def test_load_with_overrides(self, sample_config_yaml: Path):
        loader = YamlConfigLoader()
        cfg = loader.load(
            str(sample_config_yaml),
            overrides={'digest': {'min_lines': 99}},
        )
        assert cfg.digest.min_lines == 99
        assert cfg.digest.model == 'llama3:8b'

    def test_empty_yaml_raises_validation_error(self, tmp_path: Path):
        p = tmp_path / 'empty.yaml'
        p.write_text('', encoding='utf-8')
        loader = YamlConfigLoader()
        with pytest.raises(ValidationError):
            loader.load(str(p))

    def test_load_raw_returns_dict(self, sample_config_yaml: Path):
        loader = YamlConfigLoader()
        raw = loader.load_raw(str(sample_config_yaml))
        assert isinstance(raw, dict)
        assert raw['transcription']['model'] == 'breeze-q5'
        assert raw['digest']['min_lines'] == 10

    def test_load_raw_with_overrides(self, sample_config_yaml: Path):
        loader = YamlConfigLoader()
        raw = loader.load_raw(
            str(sample_config_yaml),
            overrides={'digest': {'min_lines': 99}},
        )
        assert raw['digest']['min_lines'] == 99

    def test_load_raw_preserves_extra_keys(self, tmp_path: Path):
        p = tmp_path / 'with_infra.yaml'
        p.write_text(
            'ollama:\n  host: "http://my-server:11434"\ntemplate: "default_en"\n',
            encoding='utf-8',
        )
        loader = YamlConfigLoader()
        raw = loader.load_raw(str(p))
        assert raw['ollama']['host'] == 'http://my-server:11434'
        assert raw['template'] == 'default_en'


class TestDefaultConfigResolution:
    """Tests for default config directory resolution."""

    _MINIMAL_CONFIG = 'template: "default_en"\n'

    def test_loads_from_default_config_dir(self, tmp_path: Path, monkeypatch):
        config_dir = tmp_path / 'lazy-take-notes'
        config_dir.mkdir()
        (config_dir / 'config.yaml').write_text(self._MINIMAL_CONFIG, encoding='utf-8')

        import lazy_take_notes.l3_interface_adapters.gateways.yaml_config_loader as mod

        monkeypatch.setattr(
            mod,
            'DEFAULT_CONFIG_PATHS',
            [
                config_dir / 'config.yaml',
                config_dir / 'config.yml',
            ],
        )

        loader = YamlConfigLoader()
        raw = loader.load_raw()
        assert raw['template'] == 'default_en'

    def test_no_default_config_returns_empty(self, tmp_path: Path, monkeypatch):
        nonexistent = tmp_path / 'nonexistent'

        import lazy_take_notes.l3_interface_adapters.gateways.yaml_config_loader as mod

        monkeypatch.setattr(
            mod,
            'DEFAULT_CONFIG_PATHS',
            [
                nonexistent / 'config.yaml',
                nonexistent / 'config.yml',
            ],
        )

        loader = YamlConfigLoader()
        raw = loader.load_raw()
        assert raw == {}
