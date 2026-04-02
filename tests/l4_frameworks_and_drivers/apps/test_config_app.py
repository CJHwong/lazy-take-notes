"""Tests for ConfigApp — configuration editor TUI."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from lazy_take_notes.l4_frameworks_and_drivers.apps.config import (
    ConfigApp,
    _dict_to_inline,  # noqa: PLC2701 -- testing internal helper
    _inline_to_dict,  # noqa: PLC2701 -- testing internal helper
    _resolve_editor,  # noqa: PLC2701 -- testing internal helper
    _to_float,  # noqa: PLC2701 -- testing internal helper
    _to_int,  # noqa: PLC2701 -- testing internal helper
)

# ── Monkeypatch helpers ──────────────────────────────────────────────────────

_CONFIG_MOD = 'lazy_take_notes.l4_frameworks_and_drivers.apps.config'


def _patch_config_paths(monkeypatch, config_dir: Path, config_path: Path):
    """Point both the loader and writer at a temporary config directory."""
    import lazy_take_notes.l3_interface_adapters.gateways.yaml_config_loader as loader_mod
    import lazy_take_notes.l3_interface_adapters.gateways.yaml_config_writer as writer_mod

    paths = [config_path]
    monkeypatch.setattr(loader_mod, 'DEFAULT_CONFIG_PATHS', paths)
    monkeypatch.setattr(writer_mod, 'DEFAULT_CONFIG_PATHS', paths)
    monkeypatch.setattr(writer_mod, 'CONFIG_DIR', config_dir)


def _make_app(tmp_path: Path, monkeypatch, *, config_data: dict | None = None) -> tuple[ConfigApp, Path]:
    """Create a ConfigApp with isolated config paths. Returns (app, config_path)."""
    config_dir = tmp_path / 'cfg'
    config_dir.mkdir()
    config_path = config_dir / 'config.yaml'
    if config_data is not None:
        config_path.write_text(yaml.dump(config_data), encoding='utf-8')
    _patch_config_paths(monkeypatch, config_dir, config_path)
    return ConfigApp(), config_path


# ── Unit tests for helpers ───────────────────────────────────────────────────


class TestHelpers:
    def test_to_float_valid(self):
        assert _to_float('3.14') == pytest.approx(3.14)

    def test_to_float_invalid(self):
        assert _to_float('nope') == 0.0

    def test_to_int_valid(self):
        assert _to_int('42') == 42

    def test_to_int_invalid(self):
        assert _to_int('abc') == 0

    def test_dict_to_inline(self):
        assert _dict_to_inline({'zh': 'breeze', 'en': 'large'}) == 'zh: breeze, en: large'

    def test_dict_to_inline_empty(self):
        assert not _dict_to_inline({})

    def test_inline_to_dict(self):
        assert _inline_to_dict('zh: breeze, en: large') == {'zh': 'breeze', 'en': 'large'}

    def test_inline_to_dict_empty(self):
        assert _inline_to_dict('') == {}

    def test_inline_to_dict_malformed_skips(self):
        assert _inline_to_dict('no-colon, ok: yes') == {'ok': 'yes'}


class TestResolveEditor:
    def test_visual_env_var(self, monkeypatch):
        monkeypatch.setenv('VISUAL', 'code --wait')
        monkeypatch.delenv('EDITOR', raising=False)
        assert _resolve_editor() == ['code', '--wait']

    def test_editor_env_var(self, monkeypatch):
        monkeypatch.delenv('VISUAL', raising=False)
        monkeypatch.setenv('EDITOR', 'vim')
        assert _resolve_editor() == ['vim']

    def test_visual_takes_precedence(self, monkeypatch):
        monkeypatch.setenv('VISUAL', 'emacs')
        monkeypatch.setenv('EDITOR', 'vim')
        assert _resolve_editor() == ['emacs']

    def test_platform_fallback(self, monkeypatch):
        monkeypatch.delenv('VISUAL', raising=False)
        monkeypatch.delenv('EDITOR', raising=False)
        # On macOS, should fall back to ['open', '-t']
        # On any platform, we just check it returns a list (not None)
        result = _resolve_editor()
        assert isinstance(result, list)
        assert len(result) >= 1


# ── TUI integration tests ───────────────────────────────────────────────────


class TestConfigAppComposition:
    @pytest.mark.asyncio
    async def test_has_required_widgets(self, tmp_path, monkeypatch):
        app, _ = _make_app(tmp_path, monkeypatch)
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app.query_one('#cfg-header')
            assert app.query_one('#cfg-footer')
            assert app.query_one('#cfg-tabs')
            assert app.query_one('#cfg-llm-provider')
            assert app.query_one('#cfg-digest-model')
            assert app.query_one('#cfg-output-dir')

    @pytest.mark.asyncio
    async def test_loads_defaults_when_no_config(self, tmp_path, monkeypatch):
        from textual.widgets import Input

        app, _ = _make_app(tmp_path, monkeypatch)
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app.query_one('#cfg-digest-model', Input).value == 'gpt-oss:20b'

    @pytest.mark.asyncio
    async def test_loads_existing_config(self, tmp_path, monkeypatch):
        from textual.widgets import Input

        app, _ = _make_app(tmp_path, monkeypatch, config_data={'digest': {'model': 'custom-model'}})
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app.query_one('#cfg-digest-model', Input).value == 'custom-model'

    @pytest.mark.asyncio
    async def test_loads_existing_infra_config(self, tmp_path, monkeypatch):
        from textual.widgets import Input

        app, _ = _make_app(
            tmp_path,
            monkeypatch,
            config_data={'llm_provider': 'openai', 'openai': {'api_key': 'sk-test', 'base_url': 'https://example.com'}},
        )
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app.query_one('#cfg-openai-api-key', Input).value == 'sk-test'

    @pytest.mark.asyncio
    async def test_invalid_infra_falls_back_to_defaults(self, tmp_path, monkeypatch):
        """Malformed infra keys in config.yaml should not crash — fall back to InfraConfig defaults."""
        from textual.widgets import Select

        # llm_provider expects a string; giving it a nested dict triggers ValidationError
        app, _ = _make_app(
            tmp_path,
            monkeypatch,
            config_data={'llm_provider': {'nested': 'bad'}},
        )
        async with app.run_test() as pilot:
            await pilot.pause()
            # Should fall back to default 'ollama'
            assert app.query_one('#cfg-llm-provider', Select).value == 'ollama'


class TestConfigAppSave:
    @pytest.mark.asyncio
    async def test_ctrl_s_saves_config(self, tmp_path, monkeypatch):
        app, config_path = _make_app(tmp_path, monkeypatch)
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press('ctrl+s')
            await pilot.pause()

            assert config_path.exists()
            written = yaml.safe_load(config_path.read_text(encoding='utf-8'))
            assert 'digest' in written
            assert written['digest']['model'] == 'gpt-oss:20b'

    @pytest.mark.asyncio
    async def test_saved_flag_set_on_save(self, tmp_path, monkeypatch):
        app, _ = _make_app(tmp_path, monkeypatch)
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app._saved is False
            await pilot.press('ctrl+s')
            await pilot.pause()
            assert app._saved is True

    @pytest.mark.asyncio
    async def test_save_includes_api_key_when_set(self, tmp_path, monkeypatch):
        from textual.widgets import Input

        app, config_path = _make_app(tmp_path, monkeypatch)
        async with app.run_test() as pilot:
            await pilot.pause()
            app.query_one('#cfg-openai-api-key', Input).value = 'sk-secret'
            await pilot.press('ctrl+s')
            await pilot.pause()

        written = yaml.safe_load(config_path.read_text(encoding='utf-8'))
        assert written['openai']['api_key'] == 'sk-secret'

    @pytest.mark.asyncio
    async def test_save_with_recognition_hints(self, tmp_path, monkeypatch):
        from textual.widgets import TextArea

        app, config_path = _make_app(tmp_path, monkeypatch)
        async with app.run_test() as pilot:
            await pilot.pause()
            app.query_one('#cfg-recognition-hints', TextArea).text = 'Alice\nBob\nCharlie'
            await pilot.press('ctrl+s')
            await pilot.pause()

        written = yaml.safe_load(config_path.read_text(encoding='utf-8'))
        assert written['recognition_hints'] == ['Alice', 'Bob', 'Charlie']

    @pytest.mark.asyncio
    async def test_save_validation_error_notifies(self, tmp_path, monkeypatch):
        app, _ = _make_app(tmp_path, monkeypatch)
        async with app.run_test() as pilot:
            await pilot.pause()
            # Set an invalid value (model empty string still validates, but we can
            # break it by patching build_app_config to raise)
            with patch(f'{_CONFIG_MOD}.build_app_config', side_effect=ValueError('bad')):
                await pilot.press('ctrl+s')
                await pilot.pause()
            assert app._saved is False


class TestConfigAppTestConnection:
    @pytest.mark.asyncio
    async def test_t_key_validation_error_notifies(self, tmp_path, monkeypatch):
        """If InfraConfig validation fails, _test_connection notifies and returns early."""
        from pydantic import ValidationError as PydanticValidationError

        app, _ = _make_app(tmp_path, monkeypatch)
        async with app.run_test() as pilot:
            await pilot.pause()
            app.query_one('#cfg-tabs').focus()
            await pilot.pause()
            with patch(
                f'{_CONFIG_MOD}.InfraConfig.model_validate',
                side_effect=PydanticValidationError.from_exception_data('InfraConfig', []),
            ):
                await pilot.press('t')
                await pilot.pause()
            # Should not crash — notification shown instead

    @pytest.mark.asyncio
    async def test_t_key_tests_ollama_connection_ok(self, tmp_path, monkeypatch):
        app, _ = _make_app(tmp_path, monkeypatch)
        mock_client = MagicMock()
        mock_client.check_connectivity.return_value = (True, '')

        async with app.run_test() as pilot:
            await pilot.pause()
            # Focus the list view (not an Input) so 't' triggers the handler
            app.query_one('#cfg-tabs').focus()
            await pilot.pause()
            with patch(
                'lazy_take_notes.l3_interface_adapters.gateways.ollama_llm_client.OllamaLLMClient',
                return_value=mock_client,
            ):
                await pilot.press('t')
                await pilot.pause()
            mock_client.check_connectivity.assert_called_once()

    @pytest.mark.asyncio
    async def test_t_key_tests_openai_connection_fail(self, tmp_path, monkeypatch):
        from textual.widgets import Select

        app, _ = _make_app(tmp_path, monkeypatch)
        mock_client = MagicMock()
        mock_client.check_connectivity.return_value = (False, 'timeout')

        async with app.run_test() as pilot:
            await pilot.pause()
            # Switch to openai provider
            app.query_one('#cfg-llm-provider', Select).value = 'openai'
            app.query_one('#cfg-tabs').focus()
            await pilot.pause()
            with patch(
                'lazy_take_notes.l3_interface_adapters.gateways.openai_llm_client.OpenAICompatLLMClient',
                return_value=mock_client,
            ):
                await pilot.press('t')
                await pilot.pause()
            mock_client.check_connectivity.assert_called_once()

    @pytest.mark.asyncio
    async def test_t_key_ignored_when_input_focused(self, tmp_path, monkeypatch):
        from textual.widgets import Input

        app, _ = _make_app(tmp_path, monkeypatch)
        async with app.run_test() as pilot:
            await pilot.pause()
            # Focus an Input widget — 't' should type into input, not trigger test
            app.query_one('#cfg-ollama-host', Input).focus()
            await pilot.pause()
            with patch.object(app, '_test_connection') as mock_test:
                await pilot.press('t')
                await pilot.pause()
                mock_test.assert_not_called()


class TestConfigAppRawEditor:
    @pytest.mark.asyncio
    async def test_e_key_opens_editor(self, tmp_path, monkeypatch):
        app, config_path = _make_app(tmp_path, monkeypatch)
        # Write a config so it exists
        config_path.write_text(yaml.dump({'digest': {'model': 'test'}}), encoding='utf-8')

        async with app.run_test() as pilot:
            await pilot.pause()
            app.query_one('#cfg-tabs').focus()
            await pilot.pause()
            with (
                patch(f'{_CONFIG_MOD}._resolve_editor', return_value=['cat']),
                patch(f'{_CONFIG_MOD}.subprocess.run') as mock_run,
                patch.object(app, 'suspend', side_effect=lambda: _NullContext()),
            ):
                await pilot.press('e')
                await pilot.pause()
                mock_run.assert_called_once()

    @pytest.mark.asyncio
    async def test_e_key_creates_config_if_missing(self, tmp_path, monkeypatch):
        import lazy_take_notes.l3_interface_adapters.gateways.yaml_config_writer as writer_mod

        app, config_path = _make_app(tmp_path, monkeypatch)
        # Also patch paths.CONFIG_DIR for the _open_raw_editor branch
        monkeypatch.setattr(
            'lazy_take_notes.l3_interface_adapters.gateways.paths.CONFIG_DIR',
            config_path.parent,
        )
        monkeypatch.setattr(writer_mod, 'CONFIG_DIR', config_path.parent)

        async with app.run_test() as pilot:
            await pilot.pause()
            app.query_one('#cfg-tabs').focus()
            await pilot.pause()
            with (
                patch(f'{_CONFIG_MOD}._resolve_editor', return_value=['cat']),
                patch(f'{_CONFIG_MOD}.subprocess.run'),
                patch.object(app, 'suspend', side_effect=lambda: _NullContext()),
                patch(f'{_CONFIG_MOD}.config_file_path', return_value=config_path),
            ):
                await pilot.press('e')
                await pilot.pause()
            # Config file should have been created with defaults
            assert config_path.exists()

    @pytest.mark.asyncio
    async def test_e_key_no_editor_notifies(self, tmp_path, monkeypatch):
        app, _ = _make_app(tmp_path, monkeypatch)
        async with app.run_test() as pilot:
            await pilot.pause()
            app.query_one('#cfg-tabs').focus()
            await pilot.pause()
            with patch(f'{_CONFIG_MOD}._resolve_editor', return_value=None):
                await pilot.press('e')
                await pilot.pause()
                # Should not crash — notification shown instead

    @pytest.mark.asyncio
    async def test_e_key_suspend_not_supported(self, tmp_path, monkeypatch):
        from textual.app import SuspendNotSupported

        app, config_path = _make_app(tmp_path, monkeypatch)
        config_path.write_text(yaml.dump({}), encoding='utf-8')

        async with app.run_test() as pilot:
            await pilot.pause()
            app.query_one('#cfg-tabs').focus()
            await pilot.pause()
            with (
                patch(f'{_CONFIG_MOD}._resolve_editor', return_value=['vim']),
                patch.object(app, 'suspend', side_effect=SuspendNotSupported()),
            ):
                await pilot.press('e')
                await pilot.pause()
                # Should not crash — notification shown instead

    @pytest.mark.asyncio
    async def test_e_key_ignored_when_input_focused(self, tmp_path, monkeypatch):
        from textual.widgets import Input

        app, _ = _make_app(tmp_path, monkeypatch)
        async with app.run_test() as pilot:
            await pilot.pause()
            app.query_one('#cfg-ollama-host', Input).focus()
            await pilot.pause()
            with patch.object(app, '_open_raw_editor') as mock_edit:
                await pilot.press('e')
                await pilot.pause()
                mock_edit.assert_not_called()


class TestConfigAppRepopulate:
    @pytest.mark.asyncio
    async def test_repopulate_updates_all_fields(self, tmp_path, monkeypatch):
        from textual.widgets import Input, TextArea

        app, _ = _make_app(tmp_path, monkeypatch)
        async with app.run_test() as pilot:
            await pilot.pause()
            # Mutate internal state and repopulate
            app._raw['digest']['model'] = 'repopulated-model'
            app._raw['recognition_hints'] = ['Alice', 'Bob']
            app._repopulate_fields()
            await pilot.pause()
            assert app.query_one('#cfg-digest-model', Input).value == 'repopulated-model'
            assert app.query_one('#cfg-recognition-hints', TextArea).text == 'Alice\nBob'


class TestConfigAppAppearance:
    @pytest.mark.asyncio
    async def test_appearance_tab_exists(self, tmp_path, monkeypatch):
        app, _ = _make_app(tmp_path, monkeypatch)
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app.query_one('#tab-appearance')
            assert app.query_one('#cfg-theme')

    @pytest.mark.asyncio
    async def test_theme_select_has_available_themes(self, tmp_path, monkeypatch):
        from textual.widgets import Select

        app, _ = _make_app(tmp_path, monkeypatch)
        async with app.run_test() as pilot:
            await pilot.pause()
            select = app.query_one('#cfg-theme', Select)
            option_values = {str(v) for _, v in select._options}
            assert 'textual-dark' in option_values
            assert 'nord' in option_values

    @pytest.mark.asyncio
    async def test_theme_applied_on_mount(self, tmp_path, monkeypatch):
        app, _ = _make_app(tmp_path, monkeypatch, config_data={'theme': 'nord'})
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app.theme == 'nord'

    @pytest.mark.asyncio
    async def test_theme_select_change_previews(self, tmp_path, monkeypatch):
        from textual.widgets import Select

        app, _ = _make_app(tmp_path, monkeypatch)
        async with app.run_test() as pilot:
            await pilot.pause()
            app.query_one('#cfg-theme', Select).value = 'nord'
            await pilot.pause()
            assert app.theme == 'nord'

    @pytest.mark.asyncio
    async def test_recording_preview_renders(self, tmp_path, monkeypatch):
        from lazy_take_notes.l4_frameworks_and_drivers.apps.config import RecordingPreview

        app, _ = _make_app(tmp_path, monkeypatch)
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app.query_one(RecordingPreview)

    @pytest.mark.asyncio
    async def test_save_includes_theme(self, tmp_path, monkeypatch):
        app, config_path = _make_app(tmp_path, monkeypatch)
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press('ctrl+s')
            await pilot.pause()

        written = yaml.safe_load(config_path.read_text(encoding='utf-8'))
        assert written['theme'] == 'textual-dark'

    @pytest.mark.asyncio
    async def test_repopulate_updates_theme(self, tmp_path, monkeypatch):
        from textual.widgets import Select

        app, _ = _make_app(tmp_path, monkeypatch)
        async with app.run_test() as pilot:
            await pilot.pause()
            app._infra.theme = 'nord'
            app._repopulate_fields()
            await pilot.pause()
            assert app.query_one('#cfg-theme', Select).value == 'nord'


class TestConfigAppQuit:
    @pytest.mark.asyncio
    async def test_escape_exits(self, tmp_path, monkeypatch):
        app, _ = _make_app(tmp_path, monkeypatch)
        async with app.run_test() as pilot:
            with patch.object(app, 'exit') as mock_exit:
                await pilot.press('escape')
                await pilot.pause()
                mock_exit.assert_called_once()


# ── Test utilities ───────────────────────────────────────────────────────────


class _NullContext:  # noqa: N801 -- intentionally lowercase-ish for readability as context manager
    """Minimal context manager that does nothing (replaces app.suspend)."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass
