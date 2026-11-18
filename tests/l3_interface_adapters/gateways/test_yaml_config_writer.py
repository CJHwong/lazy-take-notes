"""Tests for YAML config writer gateway."""

from __future__ import annotations

import stat
import sys
from pathlib import Path

import pytest
import yaml

from lazy_take_notes.l3_interface_adapters.gateways import yaml_config_writer as mod
from lazy_take_notes.l3_interface_adapters.gateways.yaml_config_writer import write_config


class TestWriteConfig:
    def test_creates_new_file(self, tmp_path: Path, monkeypatch):
        config_dir = tmp_path / 'lazy-take-notes'
        config_dir.mkdir()
        default_path = config_dir / 'config.yaml'

        monkeypatch.setattr(mod, 'CONFIG_DIR', config_dir)
        monkeypatch.setattr(mod, 'DEFAULT_CONFIG_PATHS', [default_path])

        path = write_config({'digest': {'model': 'test-model'}})
        assert path == default_path
        assert path.exists()

        written = yaml.safe_load(path.read_text(encoding='utf-8'))
        assert written['digest']['model'] == 'test-model'

    def test_merges_with_existing(self, tmp_path: Path, monkeypatch):
        config_dir = tmp_path / 'lazy-take-notes'
        config_dir.mkdir()
        default_path = config_dir / 'config.yaml'
        default_path.write_text(
            yaml.dump({'digest': {'model': 'old-model', 'min_lines': 10}}),
            encoding='utf-8',
        )

        monkeypatch.setattr(mod, 'CONFIG_DIR', config_dir)
        monkeypatch.setattr(mod, 'DEFAULT_CONFIG_PATHS', [default_path])

        write_config({'digest': {'model': 'new-model'}})
        written = yaml.safe_load(default_path.read_text(encoding='utf-8'))
        assert written['digest']['model'] == 'new-model'
        assert written['digest']['min_lines'] == 10  # preserved

    def test_uses_existing_yml_extension(self, tmp_path: Path, monkeypatch):
        """If config.yml exists (not .yaml), write to that file."""
        config_dir = tmp_path / 'lazy-take-notes'
        config_dir.mkdir()
        yaml_path = config_dir / 'config.yaml'
        yml_path = config_dir / 'config.yml'
        yml_path.write_text(yaml.dump({'output': {'directory': './old'}}), encoding='utf-8')

        monkeypatch.setattr(mod, 'CONFIG_DIR', config_dir)
        monkeypatch.setattr(mod, 'DEFAULT_CONFIG_PATHS', [yaml_path, yml_path])

        path = write_config({'output': {'directory': './new'}})
        assert path == yml_path
        written = yaml.safe_load(yml_path.read_text(encoding='utf-8'))
        assert written['output']['directory'] == './new'

    def test_creates_config_dir_if_missing(self, tmp_path: Path, monkeypatch):
        config_dir = tmp_path / 'new-dir' / 'lazy-take-notes'
        default_path = config_dir / 'config.yaml'

        monkeypatch.setattr(mod, 'CONFIG_DIR', config_dir)
        monkeypatch.setattr(mod, 'DEFAULT_CONFIG_PATHS', [default_path])

        write_config({'transcription': {'model': 'test'}})
        assert default_path.exists()

    def test_empty_yaml_treated_as_empty_dict(self, tmp_path: Path, monkeypatch):
        config_dir = tmp_path / 'lazy-take-notes'
        config_dir.mkdir()
        default_path = config_dir / 'config.yaml'
        default_path.write_text('', encoding='utf-8')

        monkeypatch.setattr(mod, 'CONFIG_DIR', config_dir)
        monkeypatch.setattr(mod, 'DEFAULT_CONFIG_PATHS', [default_path])

        write_config({'digest': {'model': 'x'}})
        written = yaml.safe_load(default_path.read_text(encoding='utf-8'))
        assert written['digest']['model'] == 'x'


class TestExplicitPath:
    def test_writes_the_given_file_and_leaves_the_default_alone(self, tmp_path: Path, monkeypatch):
        default_path = tmp_path / 'cfg' / 'config.yaml'
        monkeypatch.setattr(mod, 'CONFIG_DIR', default_path.parent)
        monkeypatch.setattr(mod, 'DEFAULT_CONFIG_PATHS', [default_path])
        custom = tmp_path / 'elsewhere' / 'team.yaml'
        custom.parent.mkdir()
        custom.write_text(yaml.dump({'llm_provider': 'openai', 'digest': {'model': 'a'}}), encoding='utf-8')

        assert write_config({'digest': {'model': 'b'}}, path=custom) == custom
        assert yaml.safe_load(custom.read_text(encoding='utf-8')) == {
            'llm_provider': 'openai',
            'digest': {'model': 'b'},
        }
        assert not default_path.exists()


@pytest.mark.skipif(sys.platform == 'win32', reason='POSIX file modes')
class TestOnlyTheUserCanRead:
    """config.yaml can hold an API key."""

    def test_a_new_file_is_private(self, tmp_path: Path):
        path = tmp_path / 'config.yaml'
        write_config({'openai': {'api_key': 'k'}}, path)
        assert stat.S_IMODE(path.stat().st_mode) == 0o600

    def test_an_older_readable_file_becomes_private(self, tmp_path: Path):
        path = tmp_path / 'config.yaml'
        path.write_text('{}')
        path.chmod(0o644)
        write_config({'theme': 'nord'}, path)
        assert stat.S_IMODE(path.stat().st_mode) == 0o600

    def test_a_symlinked_file_stays_a_symlink(self, tmp_path: Path):
        target = tmp_path / 'dotfiles' / 'config.yaml'
        target.parent.mkdir()
        target.write_text('{}')
        link = tmp_path / 'config.yaml'
        link.symlink_to(target)
        write_config({'theme': 'nord'}, link)
        assert link.is_symlink()
        assert yaml.safe_load(target.read_text())['theme'] == 'nord'
