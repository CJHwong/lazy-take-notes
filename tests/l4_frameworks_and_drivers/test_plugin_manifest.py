"""Tests for plugin manifest (plugins.yaml + plugins.txt management)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from lazy_take_notes.l4_frameworks_and_drivers.plugin_manifest import (
    add_plugin,
    load_plugins,
    parse_spec_name,
    remove_plugin,
    save_plugins,
    validate_spec,
)


class TestParseSpecName:
    def test_plain_name(self):
        assert parse_spec_name('my-plugin') == 'my-plugin'

    def test_git_spec(self):
        assert parse_spec_name('my-plugin @ git+https://github.com/user/repo.git') == 'my-plugin'

    def test_version_pin(self):
        assert parse_spec_name('my-plugin>=1.0') == 'my-plugin'

    def test_exact_version(self):
        assert parse_spec_name('my-plugin==2.0.0') == 'my-plugin'

    def test_whitespace_stripped(self):
        assert parse_spec_name('  my-plugin  ') == 'my-plugin'


class TestLoadPlugins:
    def test_empty_when_no_file(self, tmp_path: Path):
        assert load_plugins(tmp_path) == []

    def test_reads_yaml(self, tmp_path: Path):
        (tmp_path / 'plugins.yaml').write_text('plugins:\n  - plugin-a\n  - plugin-b\n')
        assert load_plugins(tmp_path) == ['plugin-a', 'plugin-b']

    def test_empty_yaml(self, tmp_path: Path):
        (tmp_path / 'plugins.yaml').write_text('')
        assert load_plugins(tmp_path) == []

    def test_yaml_with_no_plugins_key(self, tmp_path: Path):
        (tmp_path / 'plugins.yaml').write_text('other: value\n')
        assert load_plugins(tmp_path) == []


class TestSavePlugins:
    def test_creates_both_files(self, tmp_path: Path):
        specs = ['plugin-a @ git+https://example.com', 'plugin-b']
        save_plugins(specs, tmp_path)

        assert (tmp_path / 'plugins.yaml').exists()
        assert (tmp_path / 'plugins.txt').exists()

        reloaded = load_plugins(tmp_path)
        assert reloaded == specs

        lines = (tmp_path / 'plugins.txt').read_text().strip().splitlines()
        assert lines == specs

    def test_empty_list_clears_files(self, tmp_path: Path):
        save_plugins(['something'], tmp_path)
        save_plugins([], tmp_path)

        assert load_plugins(tmp_path) == []
        assert not (tmp_path / 'plugins.txt').read_text()

    def test_creates_parent_dirs(self, tmp_path: Path):
        deep = tmp_path / 'a' / 'b'
        save_plugins(['x'], deep)
        assert load_plugins(deep) == ['x']


class TestAddPlugin:
    def test_add_new_plugin(self, tmp_path: Path):
        err = add_plugin('plugin-a', config_dir=tmp_path, skip_validation=True)
        assert err is None
        assert load_plugins(tmp_path) == ['plugin-a']

    def test_duplicate_add_is_idempotent(self, tmp_path: Path):
        add_plugin('plugin-a', config_dir=tmp_path, skip_validation=True)
        add_plugin('plugin-a', config_dir=tmp_path, skip_validation=True)
        assert load_plugins(tmp_path) == ['plugin-a']

    def test_duplicate_by_name_with_different_spec(self, tmp_path: Path):
        add_plugin('plugin-a @ git+https://v1', config_dir=tmp_path, skip_validation=True)
        add_plugin('plugin-a @ git+https://v2', config_dir=tmp_path, skip_validation=True)
        assert load_plugins(tmp_path) == ['plugin-a @ git+https://v1']

    def test_add_multiple_different_plugins(self, tmp_path: Path):
        add_plugin('plugin-a', config_dir=tmp_path, skip_validation=True)
        add_plugin('plugin-b', config_dir=tmp_path, skip_validation=True)
        assert load_plugins(tmp_path) == ['plugin-a', 'plugin-b']

    def test_validation_failure_does_not_write(self, tmp_path: Path):
        with patch(
            'lazy_take_notes.l4_frameworks_and_drivers.plugin_manifest.validate_spec',
            return_value=(False, 'package not found'),
        ):
            err = add_plugin('nonexistent-pkg', config_dir=tmp_path)
        assert err is not None
        assert 'package not found' in err
        assert load_plugins(tmp_path) == []

    def test_plugins_txt_regenerated_on_add(self, tmp_path: Path):
        add_plugin('plugin-a', config_dir=tmp_path, skip_validation=True)
        add_plugin('plugin-b', config_dir=tmp_path, skip_validation=True)
        lines = (tmp_path / 'plugins.txt').read_text().strip().splitlines()
        assert lines == ['plugin-a', 'plugin-b']


class TestRemovePlugin:
    def test_remove_existing(self, tmp_path: Path):
        add_plugin('plugin-a', config_dir=tmp_path, skip_validation=True)
        add_plugin('plugin-b', config_dir=tmp_path, skip_validation=True)

        removed = remove_plugin('plugin-a', config_dir=tmp_path)
        assert removed is True
        assert load_plugins(tmp_path) == ['plugin-b']

    def test_remove_nonexistent(self, tmp_path: Path):
        removed = remove_plugin('plugin-a', config_dir=tmp_path)
        assert removed is False

    def test_plugins_txt_regenerated_on_remove(self, tmp_path: Path):
        add_plugin('plugin-a', config_dir=tmp_path, skip_validation=True)
        add_plugin('plugin-b', config_dir=tmp_path, skip_validation=True)
        remove_plugin('plugin-a', config_dir=tmp_path)
        lines = (tmp_path / 'plugins.txt').read_text().strip().splitlines()
        assert lines == ['plugin-b']


class TestValidateSpec:
    def test_success(self):
        with patch(
            'lazy_take_notes.l4_frameworks_and_drivers.plugin_manifest.subprocess.run',
        ) as mock_run:
            mock_run.return_value.returncode = 0
            ok, err = validate_spec('my-plugin')
        assert ok is True
        assert not err

    def test_failure(self):
        with patch(
            'lazy_take_notes.l4_frameworks_and_drivers.plugin_manifest.subprocess.run',
        ) as mock_run:
            mock_run.return_value.returncode = 1
            mock_run.return_value.stderr = 'No matching distribution'
            ok, err = validate_spec('nonexistent')
        assert ok is False
        assert 'No matching distribution' in err

    def test_uvx_not_found(self):
        with patch(
            'lazy_take_notes.l4_frameworks_and_drivers.plugin_manifest.subprocess.run',
            side_effect=FileNotFoundError,
        ):
            ok, err = validate_spec('my-plugin')
        assert ok is False
        assert 'uvx not found' in err

    def test_timeout(self):
        import subprocess as sp

        with patch(
            'lazy_take_notes.l4_frameworks_and_drivers.plugin_manifest.subprocess.run',
            side_effect=sp.TimeoutExpired('uvx', 120),
        ):
            ok, err = validate_spec('my-plugin')
        assert ok is False
        assert 'timed out' in err
