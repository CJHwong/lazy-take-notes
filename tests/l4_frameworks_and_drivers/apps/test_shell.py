"""Tests for ShellApp routing between the picker and mode screens."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from lazy_take_notes.l4_frameworks_and_drivers.apps.shell import ShellApp

_CLI = 'lazy_take_notes.l4_frameworks_and_drivers.cli_helpers'
_PICKERS = 'lazy_take_notes.l4_frameworks_and_drivers.pickers'
_APPS = 'lazy_take_notes.l4_frameworks_and_drivers.apps'


class TestShellRouting:
    def test_config_pushes_config_screen(self):
        shell = ShellApp(kiosk=True)
        with (
            patch(f'{_APPS}.config.ConfigApp') as config_app,
            patch.object(shell, 'push_screen') as push,
        ):
            shell._open_mode('config')  # noqa: SLF001 -- routing unit test
        push.assert_called_once()
        assert push.call_args.args[0] is config_app.return_value

    def test_view_pushes_session_picker(self):
        shell = ShellApp(kiosk=True)
        with (
            patch(f'{_CLI}.load_config', return_value=(MagicMock(), MagicMock(), MagicMock())),
            patch(f'{_PICKERS}.session_picker.SessionPicker') as picker,
            patch.object(shell, 'push_screen') as push,
        ):
            shell._open_mode('view')  # noqa: SLF001 -- routing unit test
        push.assert_called_once()
        assert push.call_args.args[0] is picker.return_value

    def test_view_selection_pushes_view_screen(self, tmp_path):
        shell = ShellApp(kiosk=True)
        with (
            patch(f'{_APPS}.view.ViewApp') as view_app,
            patch.object(shell, 'push_screen') as push,
        ):
            shell._on_session_selected(tmp_path)  # noqa: SLF001 -- routing unit test
        push.assert_called_once()
        assert push.call_args.args[0] is view_app.return_value

    def test_view_cancel_finishes(self):
        shell = ShellApp(kiosk=True)
        with patch.object(shell, '_show_welcome') as welcome:
            shell._on_session_selected(None)  # noqa: SLF001 -- routing unit test
        welcome.assert_called_once()

    def test_record_pushes_template_picker(self):
        shell = ShellApp(kiosk=True)
        config, infra, loader = MagicMock(), MagicMock(), MagicMock()
        infra.show_builtin_templates = True
        with (
            patch(f'{_CLI}.load_config', return_value=(config, infra, loader)),
            patch(f'{_PICKERS}.template_picker.TemplatePicker') as picker,
            patch.object(shell, 'push_screen') as push,
        ):
            shell._open_record()  # noqa: SLF001 -- routing unit test
        assert shell._record_state == (config, infra, loader)  # noqa: SLF001 -- routing unit test
        push.assert_called_once()
        assert push.call_args.args[0] is picker.return_value

    def test_record_template_builds_and_pushes_record_screen(self, tmp_path):
        shell = ShellApp(kiosk=True)
        config, infra, loader = MagicMock(), MagicMock(), MagicMock()
        loader.load.return_value = MagicMock()
        shell._record_state = (config, infra, loader)  # noqa: SLF001 -- routing unit test
        screen = MagicMock()
        with (
            patch(f'{_CLI}.build_record_session', return_value=(screen, tmp_path)) as build,
            patch('lazy_take_notes.l4_frameworks_and_drivers.keep_awake.keep_awake') as awake,
            patch.object(shell, 'push_screen') as push,
        ):
            shell._on_record_template('default_en')  # noqa: SLF001 -- routing unit test
        build.assert_called_once()
        push.assert_called_once()
        assert push.call_args.args[0] is screen
        assert shell.session_artifacts is not None
        assert shell.session_artifacts.session_dir == tmp_path
        awake.return_value.__enter__.assert_called_once()

    def test_create_template_pushes_builder(self):
        shell = ShellApp(kiosk=True)
        with (
            patch(f'{_APPS}.template_builder.TemplateBuilderApp') as builder,
            patch.object(shell, 'push_screen') as push,
        ):
            shell._on_record_template('__create_template__')  # noqa: SLF001 -- routing unit test
        push.assert_called_once()
        assert push.call_args.args[0] is builder.return_value

    def test_finish_exits_in_single_shot(self):
        shell = ShellApp(kiosk=False, initial_mode='config')
        with patch.object(shell, 'exit') as mock_exit:
            shell._finished()  # noqa: SLF001 -- routing unit test
        mock_exit.assert_called_once()

    def test_finish_reopens_welcome_in_kiosk(self):
        shell = ShellApp(kiosk=True)
        with (
            patch(f'{_PICKERS}.welcome_picker.WelcomePicker') as picker,
            patch.object(shell, 'push_screen') as push,
        ):
            shell._finished()  # noqa: SLF001 -- routing unit test
        push.assert_called_once()
        assert push.call_args.args[0] is picker.return_value

    def test_welcome_cancel_exits_shell(self):
        shell = ShellApp(kiosk=True)
        with patch.object(shell, 'exit') as mock_exit:
            shell._on_mode_selected(None)  # noqa: SLF001 -- routing unit test
        mock_exit.assert_called_once()
