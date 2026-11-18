"""Tests for ConfigApp — configuration editor TUI."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml
from textual.widgets import Static

from lazy_take_notes.l4_frameworks_and_drivers.apps.config.app import (
    ConfigApp,
    _DiscardChangesModal,  # noqa: PLC2701 -- testing internal modal
)
from lazy_take_notes.l4_frameworks_and_drivers.apps.config.fields import (
    dict_to_inline,
    inline_to_dict,
    resolve_editor,
    to_float,
    to_int,
)
from lazy_take_notes.l4_frameworks_and_drivers.apps.config.models_tab import ModelChoiceRow, ModelChooser
from lazy_take_notes.l4_frameworks_and_drivers.apps.config.sections import LanguageRow, model_uri

# ── Monkeypatch helpers ──────────────────────────────────────────────────────

_CONFIG_MOD = 'lazy_take_notes.l4_frameworks_and_drivers.apps.config.app'


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
    def testto_float_valid(self):
        assert to_float('3.14') == pytest.approx(3.14)

    def testto_float_invalid(self):
        assert to_float('nope') == 0.0

    def testto_int_valid(self):
        assert to_int('42') == 42

    def testto_int_invalid(self):
        assert to_int('abc') == 0

    def testdict_to_inline(self):
        assert dict_to_inline({'zh': 'breeze', 'en': 'large'}) == 'zh: breeze, en: large'

    def testdict_to_inline_empty(self):
        assert not dict_to_inline({})

    def testinline_to_dict(self):
        assert inline_to_dict('zh: breeze, en: large') == {'zh': 'breeze', 'en': 'large'}

    def testinline_to_dict_empty(self):
        assert inline_to_dict('') == {}

    def testinline_to_dict_malformed_skips(self):
        assert inline_to_dict('no-colon, ok: yes') == {'ok': 'yes'}


class TestResolveEditor:
    def test_visual_env_var(self, monkeypatch):
        monkeypatch.setenv('VISUAL', 'code --wait')
        monkeypatch.delenv('EDITOR', raising=False)
        assert resolve_editor() == ['code', '--wait']

    def test_editor_env_var(self, monkeypatch):
        monkeypatch.delenv('VISUAL', raising=False)
        monkeypatch.setenv('EDITOR', 'vim')
        assert resolve_editor() == ['vim']

    def test_visual_takes_precedence(self, monkeypatch):
        monkeypatch.setenv('VISUAL', 'emacs')
        monkeypatch.setenv('EDITOR', 'vim')
        assert resolve_editor() == ['emacs']

    def test_platform_fallback(self, monkeypatch):
        monkeypatch.delenv('VISUAL', raising=False)
        monkeypatch.delenv('EDITOR', raising=False)
        # On macOS, should fall back to ['open', '-t']
        # On any platform, we just check it returns a list (not None)
        result = resolve_editor()
        assert isinstance(result, list)
        assert len(result) >= 1


# ── TUI integration tests ───────────────────────────────────────────────────


class TestConfigAppComposition:
    @pytest.mark.asyncio
    async def test_has_required_widgets(self, tmp_path, monkeypatch):
        app, _ = _make_app(tmp_path, monkeypatch)
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app.query_one('#cfg-title')
            assert app.query_one('#cfg-path')
            assert app.query_one('#cfg-footer')
            assert app.query_one('#cfg-nav')
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
    async def test_clear_api_key_writes_null(self, tmp_path, monkeypatch):
        app, config_path = _make_app(
            tmp_path,
            monkeypatch,
            config_data={'openai': {'api_key': 'sk-old'}},
        )
        async with app.run_test() as pilot:
            await pilot.pause()
            app._clear_api_key = True
            await pilot.press('ctrl+s')
            await pilot.pause()

        written = yaml.safe_load(config_path.read_text(encoding='utf-8'))
        assert written['openai']['api_key'] is None

    @pytest.mark.asyncio
    async def test_bad_number_blocks_save(self, tmp_path, monkeypatch):
        from textual.widgets import Input

        app, config_path = _make_app(tmp_path, monkeypatch)
        async with app.run_test() as pilot:
            await pilot.pause()
            app.query_one('#cfg-trans-chunk', Input).value = 'abc'

            assert 'Chunk duration must be a number.' in app._numeric_errors()
            await pilot.press('ctrl+s')
            await pilot.pause()

        assert not config_path.exists()

    @pytest.mark.asyncio
    async def test_save_includes_input_device(self, tmp_path, monkeypatch):
        from textual.widgets import Select

        app, config_path = _make_app(tmp_path, monkeypatch)
        async with app.run_test() as pilot:
            await pilot.pause()
            select = app.query_one('#cfg-audio-device', Select)
            select.set_options([('System default', '__default__'), ('Test mic', 'Test mic')])
            select.value = 'Test mic'
            await pilot.press('ctrl+s')
            await pilot.pause()

        written = yaml.safe_load(config_path.read_text(encoding='utf-8'))
        assert written['transcription']['input_device'] == 'Test mic'

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
            app.query_one('#cfg-nav').focus()
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
            app.query_one('#cfg-nav').focus()
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
            app.query_one('#cfg-nav').focus()
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
            app.query_one('#cfg-nav').focus()
            await pilot.pause()
            with (
                patch(f'{_CONFIG_MOD}.resolve_editor', return_value=['cat']),
                patch(f'{_CONFIG_MOD}.subprocess.run') as mock_run,
                patch.object(app.app, 'suspend', side_effect=lambda: _NullContext()),
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
            app.query_one('#cfg-nav').focus()
            await pilot.pause()
            with (
                patch(f'{_CONFIG_MOD}.resolve_editor', return_value=['cat']),
                patch(f'{_CONFIG_MOD}.subprocess.run'),
                patch.object(app.app, 'suspend', side_effect=lambda: _NullContext()),
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
            app.query_one('#cfg-nav').focus()
            await pilot.pause()
            with patch(f'{_CONFIG_MOD}.resolve_editor', return_value=None):
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
            app.query_one('#cfg-nav').focus()
            await pilot.pause()
            with (
                patch(f'{_CONFIG_MOD}.resolve_editor', return_value=['vim']),
                patch.object(app.app, 'suspend', side_effect=SuspendNotSupported()),
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
            assert app.query_one('#section-appearance')
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
            assert app.app.theme == 'nord'

    @pytest.mark.asyncio
    async def test_theme_select_change_previews(self, tmp_path, monkeypatch):
        from textual.widgets import Select

        app, _ = _make_app(tmp_path, monkeypatch)
        async with app.run_test() as pilot:
            await pilot.pause()
            app.query_one('#cfg-theme', Select).value = 'nord'
            await pilot.pause()
            assert app.app.theme == 'nord'

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


class TestConfigAppTranscriptionProvider:
    @pytest.mark.asyncio
    async def test_transcription_provider_select_exists(self, tmp_path, monkeypatch):
        from textual.widgets import Select

        app, _ = _make_app(tmp_path, monkeypatch)
        async with app.run_test() as pilot:
            await pilot.pause()
            select = app.query_one('#cfg-trans-provider', Select)
            assert select.value == 'whisper-cpp'

    @pytest.mark.asyncio
    async def test_save_roundtrips_transcription_provider(self, tmp_path, monkeypatch):
        app, config_path = _make_app(tmp_path, monkeypatch)
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press('ctrl+s')
            await pilot.pause()

        written = yaml.safe_load(config_path.read_text(encoding='utf-8'))
        assert written['transcription_provider'] == 'whisper-cpp'

    @pytest.mark.asyncio
    async def test_repopulate_updates_transcription_provider(self, tmp_path, monkeypatch):
        from textual.widgets import Select

        app, _ = _make_app(tmp_path, monkeypatch)
        async with app.run_test() as pilot:
            await pilot.pause()
            app._infra.transcription_provider = 'custom-engine'
            app._repopulate_fields()
            await pilot.pause()
            assert app.query_one('#cfg-trans-provider', Select).value == 'custom-engine'

    @pytest.mark.asyncio
    async def test_loads_config_with_custom_provider(self, tmp_path, monkeypatch):
        from textual.widgets import Select

        app, _ = _make_app(
            tmp_path,
            monkeypatch,
            config_data={'transcription_provider': 'custom-stt'},
        )
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app.query_one('#cfg-trans-provider', Select).value == 'custom-stt'

    @pytest.mark.asyncio
    async def test_plugin_provider_hides_model_fields(self, tmp_path, monkeypatch):
        from textual.widgets import Select

        app, _ = _make_app(tmp_path, monkeypatch)
        async with app.run_test() as pilot:
            await pilot.pause()
            # Default whisper-cpp: model fields visible
            assert app.query_one('#models-transcription-fields').display is True
            assert app.query_one('#plugin-trans-model-note').display is False

            # Simulate switching to a plugin provider that manages models
            select = app.query_one('#cfg-trans-provider', Select)
            select.set_options([('whisper-cpp', 'whisper-cpp'), ('plugin-stt', 'plugin-stt')])
            with patch(
                f'{_CONFIG_MOD}.plugin_manages_models',
                return_value=True,
            ):
                select.value = 'plugin-stt'
                await pilot.pause()

            assert app.query_one('#models-transcription-fields').display is False
            assert app.query_one('#plugin-trans-model-note').display is True


class TestConfigAppQuit:
    @pytest.mark.asyncio
    async def test_escape_exits(self, tmp_path, monkeypatch):
        app, _ = _make_app(tmp_path, monkeypatch)
        async with app.run_test() as pilot:
            with patch.object(app, 'dismiss') as mock_exit:
                await pilot.press('escape')
                await pilot.pause()
                mock_exit.assert_called_once()

    @pytest.mark.asyncio
    async def test_escape_with_unsaved_changes_warns(self, tmp_path, monkeypatch):
        from textual.widgets import Input

        app, _ = _make_app(tmp_path, monkeypatch)
        async with app.run_test() as pilot:
            await pilot.pause()
            app.query_one('#cfg-digest-model', Input).value = 'changed-model'
            await pilot.pause()
            with patch.object(app, 'dismiss') as mock_exit:
                await pilot.press('escape')
                await pilot.pause()
                mock_exit.assert_not_called()
                assert isinstance(app.app.screen, _DiscardChangesModal)

    @pytest.mark.asyncio
    async def test_discard_exits(self, tmp_path, monkeypatch):
        from textual.widgets import Input

        app, _ = _make_app(tmp_path, monkeypatch)
        async with app.run_test() as pilot:
            await pilot.pause()
            app.query_one('#cfg-digest-model', Input).value = 'changed-model'
            await pilot.pause()
            with patch.object(app, 'dismiss') as mock_exit:
                await pilot.press('escape')
                await pilot.pause()
                await pilot.click('#discard-confirm')
                await pilot.pause()
                mock_exit.assert_called_once()

    @pytest.mark.asyncio
    async def test_enter_keeps_editing(self, tmp_path, monkeypatch):
        # Enter answers the focused button. The safe one has focus, so a reflex Enter loses nothing.
        from textual.widgets import Input

        app, _ = _make_app(tmp_path, monkeypatch)
        async with app.run_test() as pilot:
            await pilot.pause()
            app.query_one('#cfg-digest-model', Input).value = 'changed-model'
            await pilot.pause()
            with patch.object(app, 'dismiss') as mock_exit:
                await pilot.press('escape')
                await pilot.pause()
                await pilot.press('enter')
                await pilot.pause()
                mock_exit.assert_not_called()
                assert not isinstance(app.app.screen, _DiscardChangesModal)

    @pytest.mark.asyncio
    async def test_tab_then_enter_discards(self, tmp_path, monkeypatch):
        from textual.widgets import Input

        app, _ = _make_app(tmp_path, monkeypatch)
        async with app.run_test() as pilot:
            await pilot.pause()
            app.query_one('#cfg-digest-model', Input).value = 'changed-model'
            await pilot.pause()
            with patch.object(app, 'dismiss') as mock_exit:
                await pilot.press('escape')
                await pilot.pause()
                await pilot.press('tab', 'enter')
                await pilot.pause()
                mock_exit.assert_called_once()

    @pytest.mark.asyncio
    async def test_keep_editing_stays(self, tmp_path, monkeypatch):
        from textual.widgets import Input

        app, _ = _make_app(tmp_path, monkeypatch)
        async with app.run_test() as pilot:
            await pilot.pause()
            app.query_one('#cfg-digest-model', Input).value = 'changed-model'
            await pilot.pause()
            with patch.object(app, 'dismiss') as mock_exit:
                await pilot.press('escape')
                await pilot.pause()
                await pilot.press('escape')
                await pilot.pause()
                mock_exit.assert_not_called()
                assert not isinstance(app.app.screen, _DiscardChangesModal)

    @pytest.mark.asyncio
    async def test_escape_after_save_exits_without_warning(self, tmp_path, monkeypatch):
        from textual.widgets import Input

        app, _ = _make_app(tmp_path, monkeypatch)
        async with app.run_test() as pilot:
            await pilot.pause()
            app.query_one('#cfg-digest-model', Input).value = 'changed-model'
            await pilot.pause()
            await pilot.press('ctrl+s')
            await pilot.pause()
            with patch.object(app, 'dismiss') as mock_exit:
                await pilot.press('escape')
                await pilot.pause()
                mock_exit.assert_called_once()

    @pytest.mark.asyncio
    async def test_reverted_edit_is_not_dirty(self, tmp_path, monkeypatch):
        from textual.widgets import Input

        app, _ = _make_app(tmp_path, monkeypatch)
        async with app.run_test() as pilot:
            await pilot.pause()
            field = app.query_one('#cfg-digest-model', Input)
            original = field.value
            field.value = 'changed-model'
            await pilot.pause()
            field.value = original
            await pilot.pause()
            with patch.object(app, 'dismiss') as mock_exit:
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


class TestDiarizationSettings:
    @pytest.mark.asyncio
    async def test_every_diarization_field_is_on_the_page(self, tmp_path, monkeypatch):
        from textual.widgets import Input, Select, Switch

        app, _ = _make_app(tmp_path, monkeypatch)
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app.query_one('#cfg-diarization-enabled', Switch)
            assert len(app.query('#cfg-diarization-binary')) == 0
            assert len(app.query('#cfg-diarization-model')) == 0
            assert app.query_one('#cfg-diarization-device', Select)
            assert app.query_one('#cfg-diarization-interval', Input)
            # digest.max_lines was the one setting missing from the page.
            assert app.query_one('#cfg-digest-max-lines', Input)

    @pytest.mark.asyncio
    async def test_save_writes_the_diarization_block(self, tmp_path, monkeypatch):
        from textual.widgets import Input, Select, Switch

        app, config_path = _make_app(tmp_path, monkeypatch)
        async with app.run_test() as pilot:
            await pilot.pause()
            app.query_one('#cfg-diarization-enabled', Switch).value = True
            app.query_one('#cfg-diarization-device', Select).value = 'cpu'
            app.query_one('#cfg-diarization-interval', Input).value = '30'
            await pilot.press('ctrl+s')
            await pilot.pause()

        written = yaml.safe_load(config_path.read_text(encoding='utf-8'))
        assert written['diarization'] == {
            'enabled': True,
            'binary': '',
            'model': '',
            'device': 'cpu',
            'interval_s': 30.0,
        }

    @pytest.mark.asyncio
    async def test_existing_values_are_loaded_into_the_form(self, tmp_path, monkeypatch):
        from textual.widgets import Input, Switch

        app, config_path = _make_app(
            tmp_path,
            monkeypatch,
            config_data={
                'diarization': {
                    'enabled': True,
                    'binary': '/opt/nemo-speech',
                    'model': '/models/diar.gguf',
                    'device': 'metal',
                    'interval_s': 45,
                },
                'digest': {
                    'model': 'm',
                    'min_lines': 10,
                    'min_interval': 60,
                    'compact_token_threshold': 1000,
                    'max_lines': 40,
                },
            },
        )
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app.query_one('#cfg-diarization-enabled', Switch).value is True
            assert app.query_one('#cfg-diarization-interval', Input).value == '45'
            assert app.query_one('#cfg-digest-max-lines', Input).value == '40'
            await pilot.press('ctrl+s')
            await pilot.pause()

        written = yaml.safe_load(config_path.read_text(encoding='utf-8'))
        assert written['diarization']['binary'] == '/opt/nemo-speech'
        assert written['diarization']['model'] == '/models/diar.gguf'

    @pytest.mark.asyncio
    async def test_save_blocked_when_enabled_without_audio_saving(self, tmp_path, monkeypatch):
        from textual.widgets import Switch

        app, config_path = _make_app(tmp_path, monkeypatch)
        async with app.run_test() as pilot:
            await pilot.pause()
            app.query_one('#cfg-diarization-enabled', Switch).value = True
            app.query_one('#cfg-output-save-audio', Switch).value = False
            await pilot.press('ctrl+s')
            await pilot.pause()

        assert app._saved is False
        assert not config_path.exists()


class TestNewSettingsShape:
    @pytest.mark.asyncio
    async def test_nav_order_follows_change_frequency(self, tmp_path, monkeypatch):
        from textual.widgets import ListItem

        app, _ = _make_app(tmp_path, monkeypatch)
        async with app.run_test() as pilot:
            await pilot.pause()
            order = [item.id for item in app.query(ListItem)]
            assert order == [
                'nav-models',
                'nav-digest',
                'nav-audio',
                'nav-sessions',
                'nav-appearance',
            ]

    @pytest.mark.asyncio
    async def test_default_model_radio_saves_the_uri(self, tmp_path, monkeypatch):
        from lazy_take_notes.l4_frameworks_and_drivers.apps.config.models_tab import ModelChooser

        app, config_path = _make_app(tmp_path, monkeypatch)
        async with app.run_test(size=(140, 60)) as pilot:
            await pilot.pause()
            app.query_one(ModelChooser).select('breeze25-q8')
            await pilot.pause()
            await pilot.press('ctrl+s')
            await pilot.pause()

        written = yaml.safe_load(config_path.read_text(encoding='utf-8'))
        assert written['transcription']['model'].startswith('hf://alan314159/')

    @pytest.mark.asyncio
    async def test_digest_advanced_is_not_folded(self, tmp_path, monkeypatch):
        from textual.widgets import Input

        app, _ = _make_app(tmp_path, monkeypatch)
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app.query_one('#cfg-digest-compact', Input)
            assert len(app.query('#cfg-digest-advanced')) == 0


def _toasts(app: ConfigApp) -> list[str]:
    return [str(n.message) for n in app.app._notifications]


class TestConfigAppFaults:
    @pytest.mark.asyncio
    async def test_save_write_failure_notifies(self, tmp_path, monkeypatch):
        app, _ = _make_app(tmp_path, monkeypatch)
        async with app.run_test() as pilot:
            await pilot.pause()
            with patch(f'{_CONFIG_MOD}.write_config', side_effect=PermissionError('read-only')):
                await pilot.press('ctrl+s')
                await pilot.pause()
            assert app._saved is False
            assert any('Cannot save' in t and 'read-only' in t for t in _toasts(app))

    @pytest.mark.asyncio
    async def test_editor_binary_missing_notifies(self, tmp_path, monkeypatch):
        app, config_path = _make_app(tmp_path, monkeypatch, config_data={'digest': {'model': 'test'}})
        async with app.run_test() as pilot:
            await pilot.pause()
            app.query_one('#cfg-nav').focus()
            await pilot.pause()
            with (
                patch(f'{_CONFIG_MOD}.resolve_editor', return_value=['no-such-editor']),
                patch(f'{_CONFIG_MOD}.config_file_path', return_value=config_path),
                patch(f'{_CONFIG_MOD}.subprocess.run', side_effect=FileNotFoundError('no-such-editor')),
                patch.object(app.app, 'suspend', side_effect=lambda: _NullContext()),
            ):
                await pilot.press('e')
                await pilot.pause()
            assert any('Cannot open editor' in t and 'no-such-editor' in t for t in _toasts(app))

    @pytest.mark.asyncio
    async def test_editor_leaves_broken_yaml_notifies(self, tmp_path, monkeypatch):
        app, config_path = _make_app(tmp_path, monkeypatch, config_data={'digest': {'model': 'test'}})

        def _break_file(*_args, **_kwargs):
            config_path.write_text('digest: [unclosed\n', encoding='utf-8')

        async with app.run_test() as pilot:
            await pilot.pause()
            app.query_one('#cfg-nav').focus()
            await pilot.pause()
            with (
                patch(f'{_CONFIG_MOD}.resolve_editor', return_value=['vim']),
                patch(f'{_CONFIG_MOD}.config_file_path', return_value=config_path),
                patch(f'{_CONFIG_MOD}.subprocess.run', side_effect=_break_file),
                patch.object(app.app, 'suspend', side_effect=lambda: _NullContext()),
            ):
                await pilot.press('e')
                await pilot.pause()
            assert any('Cannot reload' in t for t in _toasts(app))

    @pytest.mark.asyncio
    async def test_editor_cannot_create_config_notifies(self, tmp_path, monkeypatch):
        app, config_path = _make_app(tmp_path, monkeypatch)
        monkeypatch.setattr('lazy_take_notes.l3_interface_adapters.gateways.paths.CONFIG_DIR', config_path.parent)
        async with app.run_test() as pilot:
            await pilot.pause()
            app.query_one('#cfg-nav').focus()
            await pilot.pause()
            with (
                patch(f'{_CONFIG_MOD}.resolve_editor', return_value=['vim']),
                patch(f'{_CONFIG_MOD}.config_file_path', return_value=config_path),
                patch.object(Path, 'write_text', side_effect=PermissionError('read-only')),
                patch(f'{_CONFIG_MOD}.subprocess.run') as mock_run,
            ):
                await pilot.press('e')
                await pilot.pause()
            mock_run.assert_not_called()
            assert any('Cannot create' in t and 'read-only' in t for t in _toasts(app))


class TestCustomConfigPath:
    """With --config, Settings reads, shows, saves and opens that file, not the default."""

    def _setup(self, tmp_path, monkeypatch) -> tuple[ConfigApp, Path, Path]:
        _app, default_path = _make_app(tmp_path, monkeypatch, config_data={'digest': {'model': 'default-model'}})
        custom = tmp_path / 'team.yaml'
        custom.write_text(yaml.dump({'digest': {'model': 'team-model'}}), encoding='utf-8')
        return ConfigApp(config_path=str(custom)), custom, default_path

    def test_it_loads_the_custom_file(self, tmp_path, monkeypatch):
        app, _custom, _default = self._setup(tmp_path, monkeypatch)
        assert app._raw['digest']['model'] == 'team-model'

    @pytest.mark.asyncio
    async def test_it_shows_and_saves_the_custom_file(self, tmp_path, monkeypatch):
        app, custom, default_path = self._setup(tmp_path, monkeypatch)
        default_before = default_path.read_text(encoding='utf-8')
        async with app.run_test() as pilot:
            await pilot.pause()
            assert str(app.query_one('#cfg-path', Static).render()) == str(custom)
            app.action_save_config()
            await pilot.pause()
        assert yaml.safe_load(custom.read_text(encoding='utf-8'))['digest']['model'] == 'team-model'
        assert 'output' in yaml.safe_load(custom.read_text(encoding='utf-8'))  # the save landed here
        assert default_path.read_text(encoding='utf-8') == default_before

    @pytest.mark.asyncio
    async def test_the_raw_editor_opens_the_custom_file(self, tmp_path, monkeypatch):
        app, custom, _default = self._setup(tmp_path, monkeypatch)
        async with app.run_test() as pilot:
            await pilot.pause()
            app.query_one('#cfg-nav').focus()
            await pilot.pause()
            with (
                patch(f'{_CONFIG_MOD}.resolve_editor', return_value=['cat']),
                patch(f'{_CONFIG_MOD}.subprocess.run') as run,
                patch.object(app.app, 'suspend', side_effect=lambda: _NullContext()),
            ):
                await pilot.press('e')
                await pilot.pause()
        assert run.call_args.args[0] == ['cat', str(custom)]


class TestBrokenConfigOnOpen:
    @pytest.mark.asyncio
    async def test_settings_open_with_defaults_and_say_why(self, tmp_path, monkeypatch):
        config_dir = tmp_path / 'cfg'
        config_dir.mkdir()
        config_path = config_dir / 'config.yaml'
        config_path.write_text('digest: [unclosed\n', encoding='utf-8')
        _patch_config_paths(monkeypatch, config_dir, config_path)

        app = ConfigApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            assert any('cannot be read' in toast and '[e]' in toast for toast in _toasts(app))
            assert app.app.is_running
        assert config_path.read_text(encoding='utf-8') == 'digest: [unclosed\n'  # never overwritten on open


def _written(config_path: Path) -> dict:
    return yaml.safe_load(config_path.read_text(encoding='utf-8'))


class TestRemovingAValue:
    """A value the user removes in the form is removed from config.yaml too."""

    @pytest.mark.asyncio
    async def test_removing_a_language_override_removes_it_from_the_file(self, tmp_path, monkeypatch):
        from textual.widgets import Button

        data = {'transcription': {'models': {'ja': 'hf://ggerganov/whisper.cpp/ggml-large-v3-turbo-q8_0.bin'}}}
        app, config_path = _make_app(tmp_path, monkeypatch, config_data=data)
        async with app.run_test(size=(140, 50)) as pilot:
            await pilot.pause()
            row = next(row for row in app.query(LanguageRow) if row.locale == 'ja')
            row.query_one(Button).press()
            await pilot.pause()
            await pilot.press('ctrl+s')
            await pilot.pause()
        assert 'ja' not in _written(config_path)['transcription']['models']

    @pytest.mark.asyncio
    async def test_a_removed_built_in_language_override_stays_removed(self, tmp_path, monkeypatch):
        from textual.widgets import Button

        app, config_path = _make_app(tmp_path, monkeypatch, config_data={'digest': {'min_lines': 15}})
        async with app.run_test(size=(140, 50)) as pilot:
            await pilot.pause()
            row = next(row for row in app.query(LanguageRow) if row.locale == 'zh')
            row.query_one(Button).press()
            await pilot.pause()
            await pilot.press('ctrl+s')
            await pilot.pause()
        from lazy_take_notes.l4_frameworks_and_drivers.config import build_app_config

        written = _written(config_path)
        assert written['transcription']['models']['zh'] is None
        assert 'zh' not in build_app_config(written).transcription.models
        assert 'zh-min-nan' in build_app_config(written).transcription.models

        reopened = ConfigApp()  # the same patched config path
        async with reopened.run_test(size=(140, 50)) as pilot:
            await pilot.pause()
            assert [row.locale for row in reopened.query(LanguageRow)] == ['zh-min-nan']

    @pytest.mark.asyncio
    async def test_clearing_force_after_removes_it_from_the_file(self, tmp_path, monkeypatch):
        from textual.widgets import Input

        app, config_path = _make_app(tmp_path, monkeypatch, config_data={'digest': {'max_lines': 40}})
        async with app.run_test(size=(140, 50)) as pilot:
            await pilot.pause()
            app.query_one('#cfg-digest-max-lines', Input).value = ''
            await pilot.pause()
            await pilot.press('ctrl+s')
            await pilot.pause()
        assert 'max_lines' not in _written(config_path)['digest']

    @pytest.mark.asyncio
    async def test_keys_the_form_does_not_show_survive_a_save(self, tmp_path, monkeypatch):
        data = {'my_plugin': {'token': 'x'}, 'transcription': {'models': {'ja': 'large-v3'}, 'extra': 1}}
        app, config_path = _make_app(tmp_path, monkeypatch, config_data=data)
        async with app.run_test(size=(140, 50)) as pilot:
            await pilot.pause()
            await pilot.press('ctrl+s')
            await pilot.pause()
        written = _written(config_path)
        assert written['my_plugin'] == {'token': 'x'}
        assert written['transcription']['extra'] == 1


class TestCustomModelPath:
    @pytest.mark.asyncio
    async def test_a_missing_file_is_not_saved(self, tmp_path, monkeypatch):
        from textual.widgets import Input

        app, config_path = _make_app(tmp_path, monkeypatch)
        async with app.run_test(size=(140, 50)) as pilot:
            await pilot.pause()
            app.query_one('#cfg-trans-custom-path', Input).value = str(tmp_path / 'nope.bin')
            await pilot.pause()
            await pilot.press('ctrl+s')
            await pilot.pause()
            assert not config_path.exists()
            assert any('nope.bin' in toast for toast in _toasts(app))

    @pytest.mark.asyncio
    async def test_clicking_a_model_row_replaces_the_custom_path(self, tmp_path, monkeypatch):
        from textual.widgets import Input

        custom = tmp_path / 'mine.bin'
        custom.write_bytes(b'x')
        app, config_path = _make_app(tmp_path, monkeypatch, config_data={'transcription': {'model': str(custom)}})
        async with app.run_test(size=(140, 50)) as pilot:
            await pilot.pause()
            assert app.query_one('#cfg-trans-custom-path', Input).value == str(custom)
            row = app.query(ModelChoiceRow).first()
            row.post_message(ModelChooser.Selected(row.asset.id))
            await pilot.pause()
            assert not app.query_one('#cfg-trans-custom-path', Input).value
            await pilot.press('ctrl+s')
            await pilot.pause()
        assert _written(config_path)['transcription']['model'] == model_uri(row.asset)

    @pytest.mark.asyncio
    async def test_clearing_the_path_without_a_model_asks_for_one(self, tmp_path, monkeypatch):
        from textual.widgets import Input

        custom = tmp_path / 'mine.bin'
        custom.write_bytes(b'x')
        app, config_path = _make_app(tmp_path, monkeypatch, config_data={'transcription': {'model': str(custom)}})
        async with app.run_test(size=(140, 50)) as pilot:
            await pilot.pause()
            app.query_one('#cfg-trans-custom-path', Input).value = ''
            await pilot.pause()
            await pilot.press('ctrl+s')
            await pilot.pause()
            assert any('Choose a speech model' in toast for toast in _toasts(app))
        assert _written(config_path)['transcription']['model'] == str(custom)


class TestMessagesUseThePageWords:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ('field', 'label'),
        [
            ('#cfg-digest-min-lines', 'After lines'),
            ('#cfg-digest-min-interval', 'Wait seconds'),
            ('#cfg-digest-max-lines', 'Force after'),
            ('#cfg-digest-compact', 'Compaction threshold'),
            ('#cfg-diarization-interval', 'Update every'),
        ],
    )
    async def test_a_bad_number_names_the_field_as_the_page_does(self, tmp_path, monkeypatch, field, label):
        from textual.widgets import Input

        app, _ = _make_app(tmp_path, monkeypatch)
        async with app.run_test(size=(140, 50)) as pilot:
            await pilot.pause()
            app.query_one(field, Input).value = 'abc'
            assert f'{label} must be a number.' in app._numeric_errors()

    @pytest.mark.asyncio
    async def test_force_after_help_names_the_right_field(self, tmp_path, monkeypatch):
        from textual.widgets import Static

        app, _ = _make_app(tmp_path, monkeypatch)
        async with app.run_test(size=(140, 50)) as pilot:
            await pilot.pause()
            texts = [str(widget.render()) for widget in app.query(Static)]
            assert any('Blank means twice After lines.' in text for text in texts)


class TestApiKeyField:
    @pytest.mark.asyncio
    async def test_without_a_stored_key_there_is_nothing_to_clear(self, tmp_path, monkeypatch):
        from textual.widgets import Button, Input

        monkeypatch.delenv('OPENAI_API_KEY', raising=False)
        app, _ = _make_app(tmp_path, monkeypatch, config_data={'llm_provider': 'openai'})
        async with app.run_test(size=(140, 50)) as pilot:
            await pilot.pause()
            assert app.query_one('#cfg-openai-clear-key', Button).display is False
            assert app.query_one('#cfg-openai-api-key', Input).placeholder == 'No key stored'

    @pytest.mark.asyncio
    async def test_a_stored_key_can_be_cleared(self, tmp_path, monkeypatch):
        from textual.widgets import Button

        app, config_path = _make_app(
            tmp_path, monkeypatch, config_data={'llm_provider': 'openai', 'openai': {'api_key': 'test-key'}}
        )
        async with app.run_test(size=(140, 50)) as pilot:
            await pilot.pause()
            button = app.query_one('#cfg-openai-clear-key', Button)
            assert button.display is True
            button.press()
            await pilot.pause()
            await pilot.press('ctrl+s')
            await pilot.pause()
            assert button.display is False
        assert _written(config_path)['openai']['api_key'] is None


class TestAiTest:
    """Test AI checks what a session start checks, without freezing Settings."""

    async def _press_test(self, app, pilot, client):
        with patch(
            'lazy_take_notes.l4_frameworks_and_drivers.container.DependencyContainer.resolve_llm_client',
            return_value=client,
        ):
            app.query_one('#cfg-ai-test').press()
            await pilot.pause()
            started = list(_toasts(app))
            await app.workers.wait_for_complete()
            await pilot.pause()
        return started

    @pytest.mark.asyncio
    async def test_it_says_it_is_testing_at_once(self, tmp_path, monkeypatch):
        from tests.conftest import FakeLLMClient

        app, _ = _make_app(tmp_path, monkeypatch)
        async with app.run_test(size=(140, 50)) as pilot:
            await pilot.pause()
            started = await self._press_test(app, pilot, FakeLLMClient())
            assert any('Testing' in toast for toast in started)
            assert any(toast.startswith('AI works') for toast in _toasts(app))

    @pytest.mark.asyncio
    async def test_a_missing_summary_model_fails(self, tmp_path, monkeypatch):
        from tests.conftest import FakeLLMClient

        client = FakeLLMClient()
        client._missing_models = ['gpt-oss:20b']
        app, _ = _make_app(tmp_path, monkeypatch)
        async with app.run_test(size=(140, 50)) as pilot:
            await pilot.pause()
            await self._press_test(app, pilot, client)
            toasts = _toasts(app)
            assert not any(toast.startswith('AI works') for toast in toasts)
            assert any('Model not available: gpt-oss:20b' in toast for toast in toasts)

    @pytest.mark.asyncio
    async def test_a_failure_reads_as_one_sentence_then_the_fix(self, tmp_path, monkeypatch):
        from tests.conftest import FakeLLMClient

        client = FakeLLMClient()
        client._connectivity = (False, 'Authentication failed: Incorrect API key provided.')
        app, _ = _make_app(tmp_path, monkeypatch)
        async with app.run_test(size=(140, 50)) as pilot:
            await pilot.pause()
            await self._press_test(app, pilot, client)
            errors = [str(n.message) for n in app.app._notifications if n.severity == 'error']
            assert errors == [
                'Cannot reach the AI provider: Authentication failed: Incorrect API key provided. '
                'Start Ollama (ollama serve), or check ollama.host in Settings > Models.'
            ]


class TestBackToThePageList:
    @pytest.mark.asyncio
    async def test_esc_in_a_page_returns_to_the_list_then_leaves(self, tmp_path, monkeypatch):
        from textual.widgets import Input, ListView

        app, _ = _make_app(tmp_path, monkeypatch)
        async with app.run_test(size=(140, 50)) as pilot:
            await pilot.pause()
            app.query_one('#cfg-digest-model', Input).focus()
            await pilot.pause()
            with patch.object(app, 'dismiss') as mock_exit:
                await pilot.press('escape')
                await pilot.pause()
                mock_exit.assert_not_called()
                assert app.app.focused is app.query_one('#cfg-nav', ListView)
                await pilot.press('escape')
                await pilot.pause()
                mock_exit.assert_called_once()

    @pytest.mark.asyncio
    async def test_esc_with_a_dropdown_open_only_closes_it(self, tmp_path, monkeypatch):
        from textual.widgets import Select

        app, _ = _make_app(tmp_path, monkeypatch)
        async with app.run_test(size=(140, 50)) as pilot:
            await pilot.pause()
            select = app.query_one('#cfg-llm-provider', Select)
            select.focus()
            select.expanded = True
            await pilot.pause()
            with patch.object(app, 'dismiss') as mock_exit:
                await pilot.press('escape')
                await pilot.pause()
                mock_exit.assert_not_called()
                assert select.expanded is False
                assert app.app.focused is select
