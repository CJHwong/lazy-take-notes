"""Tests for ShellApp routing between the picker and mode screens."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import yaml
from pydantic import BaseModel, ValidationError

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

    def test_record_template_runs_the_preflight_first(self):
        from lazy_take_notes.l4_frameworks_and_drivers.apps.preflight_screen import PreflightScreen

        shell = ShellApp(kiosk=True)
        config, infra, loader = MagicMock(), MagicMock(), MagicMock()
        shell._record_state = (config, infra, loader)  # noqa: SLF001 -- routing unit test
        with (
            patch(f'{_CLI}.build_preflight_plan') as plan,
            patch(f'{_CLI}.build_record_session') as build,
            patch.object(shell, 'push_screen') as push,
        ):
            shell._on_record_template('default_en')  # noqa: SLF001 -- routing unit test
        build.assert_not_called()  # no session folder before the checks pass
        assert isinstance(push.call_args.args[0], PreflightScreen)
        assert push.call_args.args[0]._plan is plan.return_value  # noqa: SLF001 -- routing unit test

    def test_a_passed_preflight_builds_and_pushes_the_record_screen(self, tmp_path):
        shell = ShellApp(kiosk=True)
        shell._record_state = (MagicMock(), MagicMock(), MagicMock())  # noqa: SLF001 -- routing unit test
        plan, report, screen = MagicMock(), MagicMock(), MagicMock()
        with (
            patch(f'{_CLI}.build_record_session', return_value=(screen, tmp_path)) as build,
            patch('lazy_take_notes.l4_frameworks_and_drivers.keep_awake.keep_awake') as awake,
            patch.object(shell, 'push_screen') as push,
        ):
            shell._after_preflight(MagicMock(), plan, report)  # noqa: SLF001 -- routing unit test
        kwargs = build.call_args.kwargs
        assert kwargs['preflight'] is report
        assert kwargs['transcriber'] is plan.transcriber  # the model loaded by preflight is reused
        assert kwargs['llm_client'] is plan.llm_client
        assert push.call_args.args[0] is screen
        assert shell.session_artifacts is not None
        assert shell.session_artifacts.session_dir == tmp_path
        awake.return_value.__enter__.assert_called_once()

    def test_back_from_preflight_releases_the_model_and_returns(self):
        shell = ShellApp(kiosk=True)
        shell._record_state = (MagicMock(), MagicMock(), MagicMock())  # noqa: SLF001 -- routing unit test
        plan = MagicMock()
        with (
            patch(f'{_CLI}.build_record_session') as build,
            patch.object(shell, '_finished') as finished,
        ):
            shell._after_preflight(MagicMock(), plan, None)  # noqa: SLF001 -- routing unit test
        build.assert_not_called()
        plan.transcriber.close.assert_called_once()
        finished.assert_called_once()

    def test_create_template_pushes_builder(self):
        shell = ShellApp(kiosk=True, config_path='/custom/config.yaml')
        with (
            patch(f'{_APPS}.template_builder.TemplateBuilderApp') as builder,
            patch.object(shell, 'push_screen') as push,
        ):
            shell._on_record_template('__create_template__')  # noqa: SLF001 -- routing unit test
        push.assert_called_once()
        assert push.call_args.args[0] is builder.return_value
        builder.assert_called_once_with(config_path='/custom/config.yaml')

    @pytest.mark.parametrize(
        ('handler', 'back_to'),
        [('_on_record_template', '_open_record'), ('_on_transcribe_template', '_open_transcribe')],
    )
    def test_edit_with_ai_opens_the_builder_on_that_template(self, handler, back_to):
        from lazy_take_notes.l4_frameworks_and_drivers.pickers.template_picker import EditTemplateRequest

        shell = ShellApp(kiosk=True, config_path='/custom/config.yaml')
        loader = MagicMock()
        with (
            patch(f'{_APPS}.template_builder.TemplateBuilderApp') as builder,
            patch.object(shell, 'push_screen') as push,
            patch.object(shell, '_record_context', return_value=(None, None, loader)),
            patch.object(shell, '_transcribe_context', return_value=(None, None, loader)),
        ):
            getattr(shell, handler)(EditTemplateRequest('standup'))
        loader.load.assert_called_once_with('standup')
        builder.assert_called_once_with(
            config_path='/custom/config.yaml', base_template=loader.load.return_value, base_key='standup'
        )
        assert push.call_args.args[1] == getattr(shell, back_to)  # back to the picker after

    def test_settings_gets_the_config_path(self):
        shell = ShellApp(kiosk=True, config_path='/custom/config.yaml')
        with (
            patch(f'{_APPS}.config.ConfigApp') as settings,
            patch.object(shell, 'push_screen'),
        ):
            shell._open_config()  # noqa: SLF001 -- routing unit test
        settings.assert_called_once_with(config_path='/custom/config.yaml')

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


def _validation_error() -> ValidationError:
    class _Strict(BaseModel):
        value: int

    try:
        _Strict.model_validate({'value': 'not a number'})
    except ValidationError as error:
        return error
    raise AssertionError('expected a validation error')


_CONFIG_FAULTS = [
    SystemExit(1),
    _validation_error(),
    yaml.YAMLError('bad indent'),
    PermissionError(13, 'Permission denied'),
]
_CONFIG_FAULT_IDS = ['missing-file-exit', 'invalid-value', 'bad-yaml', 'unreadable']


class TestShellFaults:
    @pytest.mark.parametrize('error', _CONFIG_FAULTS, ids=_CONFIG_FAULT_IDS)
    @pytest.mark.parametrize(
        'open_step',
        [
            lambda shell: shell._open_record(),  # noqa: SLF001 -- routing unit test
            lambda shell: shell._on_transcribe_file(True),  # noqa: SLF001 -- routing unit test
            lambda shell: shell._open_session_picker(),  # noqa: SLF001 -- routing unit test
        ],
        ids=['record', 'transcribe', 'view'],
    )
    def test_a_broken_config_returns_to_the_welcome_screen(self, error, open_step):
        shell = ShellApp(kiosk=True)
        with (
            patch(f'{_CLI}.load_config', side_effect=error),
            patch.object(shell, 'notify') as notify,
            patch.object(shell, '_show_welcome') as welcome,
        ):
            open_step(shell)

        welcome.assert_called_once()
        assert notify.call_args.kwargs['severity'] == 'error'

    @pytest.mark.parametrize(
        'error',
        [ValueError('bad template'), KeyError('system_prompt'), yaml.YAMLError('bad'), OSError(2, 'gone')],
        ids=['invalid', 'missing-key', 'bad-yaml', 'missing-file'],
    )
    @pytest.mark.parametrize('mode', ['record', 'transcribe'])
    def test_a_broken_template_returns_to_the_welcome_screen(self, error, mode):
        shell = ShellApp(kiosk=True)
        loader = MagicMock()
        loader.load.side_effect = error
        state = (MagicMock(), MagicMock(), loader)
        if mode == 'record':
            shell._record_state = state  # noqa: SLF001 -- routing unit test
            pick = shell._on_record_template  # noqa: SLF001 -- routing unit test
        else:
            shell._transcribe_state = state  # noqa: SLF001 -- routing unit test
            pick = shell._on_transcribe_template  # noqa: SLF001 -- routing unit test
        with patch.object(shell, 'notify') as notify, patch.object(shell, '_show_welcome') as welcome:
            pick('custom')

        welcome.assert_called_once()
        assert notify.call_args.kwargs['severity'] == 'error'

    def test_a_session_that_cannot_start_returns_to_the_welcome_screen(self):
        shell = ShellApp(kiosk=True)
        shell._record_state = (MagicMock(), MagicMock(), MagicMock())  # noqa: SLF001 -- routing unit test
        with (
            patch(f'{_CLI}.build_record_session', side_effect=PermissionError(13, 'output folder')),
            patch.object(shell, 'notify') as notify,
            patch.object(shell, '_show_welcome') as welcome,
        ):
            shell._after_preflight(MagicMock(), MagicMock(), MagicMock())  # noqa: SLF001 -- routing unit test

        welcome.assert_called_once()
        assert 'output folder' in notify.call_args.args[0]


class TestBackFromTheBuilder:
    """Closing the builder returns to the template picker it came from."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize('opener', ['_open_record', '_open_transcribe'])
    async def test_esc_returns_to_the_picker(self, opener, tmp_path, monkeypatch):
        from lazy_take_notes.l4_frameworks_and_drivers.apps.template_builder import TemplateBuilderApp
        from lazy_take_notes.l4_frameworks_and_drivers.pickers import template_picker
        from lazy_take_notes.l4_frameworks_and_drivers.pickers.template_picker import TemplatePicker

        # Without this flag file, the picker first shows the built-in templates notice.
        noticed = tmp_path / 'noticed'
        noticed.touch()
        monkeypatch.setattr(template_picker, 'BUILTIN_TEMPLATES_NOTICED_PATH', noticed)
        shell = ShellApp(kiosk=True, transcribe_options={'audio_path': str(tmp_path / 'a.wav')})
        async with shell.run_test(size=(170, 50)) as pilot:
            await pilot.pause()
            getattr(shell, opener)()
            await pilot.pause()
            assert isinstance(shell.screen, TemplatePicker)
            shell.screen.dismiss('__create_template__')
            await pilot.pause()
            assert isinstance(shell.screen, TemplateBuilderApp)
            shell.screen.dismiss()
            for _ in range(4):
                await pilot.pause()
            assert isinstance(shell.screen, TemplatePicker)
