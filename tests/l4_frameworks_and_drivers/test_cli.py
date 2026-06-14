"""Tests for CLI entry point — patches deferred imports at source module level."""

from __future__ import annotations

import json
import os
import re
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import MagicMock, patch

import click
import pytest
from click.testing import CliRunner

from lazy_take_notes import __version__
from lazy_take_notes.l4_frameworks_and_drivers.cli import (
    _load_plugins,  # noqa: PLC2701 -- testing private helper
    cli,
)
from lazy_take_notes.l4_frameworks_and_drivers.cli_helpers import (
    make_session_dir as _make_session_dir,
)
from lazy_take_notes.l4_frameworks_and_drivers.cli_helpers import (
    preflight_llm as _preflight_llm,
)
from lazy_take_notes.l4_frameworks_and_drivers.cli_helpers import (
    preflight_microphone as _preflight_microphone,
)
from lazy_take_notes.l4_frameworks_and_drivers.config import InfraConfig, build_app_config

# Patch targets at SOURCE module level (not cli module) because cli() uses
# deferred `from X import Y` which creates local bindings that bypass
# module-level attribute patches.
_YAML_CFG = 'lazy_take_notes.l3_interface_adapters.gateways.yaml_config_loader.YamlConfigLoader'
_YAML_TPL = 'lazy_take_notes.l3_interface_adapters.gateways.yaml_template_loader.YamlTemplateLoader'
_BUILD = 'lazy_take_notes.l4_frameworks_and_drivers.config.build_app_config'
_INFRA = 'lazy_take_notes.l4_frameworks_and_drivers.config.InfraConfig'
_PICKER = 'lazy_take_notes.l4_frameworks_and_drivers.pickers.template_picker.TemplatePicker'
_SESSION_PICKER = 'lazy_take_notes.l4_frameworks_and_drivers.pickers.session_picker.SessionPicker'
_FILE_PICKER = 'lazy_take_notes.l4_frameworks_and_drivers.pickers.file_picker.FilePicker'
_WELCOME_PICKER = 'lazy_take_notes.l4_frameworks_and_drivers.pickers.welcome_picker.WelcomePicker'

_CLI = 'lazy_take_notes.l4_frameworks_and_drivers.cli'
_CLI_HELPERS = 'lazy_take_notes.l4_frameworks_and_drivers.cli_helpers'


@pytest.fixture(autouse=True)
def _no_caffeinate(monkeypatch):
    """Never spawn the real `caffeinate` inhibitor during CLI tests."""
    monkeypatch.setattr(
        'lazy_take_notes.l4_frameworks_and_drivers.keep_awake.inhibit_sleep',
        lambda: None,
    )


class TestMakeSessionDir:
    def test_creates_dir_with_timestamp(self, tmp_path: Path):
        result = _make_session_dir(tmp_path, label=None)
        assert result.exists()
        assert re.match(r'\d{4}-\d{2}-\d{2}_\d{6}', result.name)

    def test_appends_sanitized_label(self, tmp_path: Path):
        result = _make_session_dir(tmp_path, label='sprint review!')
        assert result.exists()
        assert 'sprint_review_' in result.name

    def test_label_hyphens_preserved(self, tmp_path: Path):
        result = _make_session_dir(tmp_path, label='sprint-review')
        assert 'sprint-review' in result.name

    def test_creates_parents(self, tmp_path: Path):
        deep = tmp_path / 'a' / 'b'
        result = _make_session_dir(deep, label=None)
        assert result.exists()


class TestPreflightLLM:
    def test_ollama_unreachable_returns_empty_lists(self):
        mock_client = MagicMock()
        mock_client.check_connectivity.return_value = (False, 'Connection refused')
        mock_cls = MagicMock(return_value=mock_client)

        with patch(
            'lazy_take_notes.l3_interface_adapters.gateways.ollama_llm_client.OllamaLLMClient',
            mock_cls,
        ):
            infra = InfraConfig()
            config = build_app_config({})
            missing_d, missing_i = _preflight_llm(infra, config)

        assert missing_d == []
        assert missing_i == []

    def test_ollama_all_models_present_returns_empty(self):
        mock_client = MagicMock()
        mock_client.check_connectivity.return_value = (True, '')
        mock_client.check_models.return_value = []
        mock_cls = MagicMock(return_value=mock_client)

        with patch(
            'lazy_take_notes.l3_interface_adapters.gateways.ollama_llm_client.OllamaLLMClient',
            mock_cls,
        ):
            infra = InfraConfig()
            config = build_app_config({})
            missing_d, missing_i = _preflight_llm(infra, config)

        assert missing_d == []
        assert missing_i == []

    def test_ollama_missing_digest_model_returned(self):
        config = build_app_config({})

        mock_client = MagicMock()
        mock_client.check_connectivity.return_value = (True, '')
        mock_client.check_models.return_value = [config.digest.model]
        mock_cls = MagicMock(return_value=mock_client)

        with patch(
            'lazy_take_notes.l3_interface_adapters.gateways.ollama_llm_client.OllamaLLMClient',
            mock_cls,
        ):
            infra = InfraConfig()
            missing_d, _missing_i = _preflight_llm(infra, config)

        assert missing_d == [config.digest.model]

    def test_openai_provider_checks_connectivity(self):
        mock_client = MagicMock()
        mock_client.check_connectivity.return_value = (True, '')
        mock_client.check_models.return_value = []
        mock_cls = MagicMock(return_value=mock_client)

        with patch(
            'lazy_take_notes.l3_interface_adapters.gateways.openai_llm_client.OpenAICompatLLMClient',
            mock_cls,
        ):
            infra = InfraConfig(llm_provider='openai')
            config = build_app_config({})
            missing_d, missing_i = _preflight_llm(infra, config)

        assert missing_d == []
        assert missing_i == []

    def test_openai_provider_unreachable_returns_empty_lists(self):
        mock_client = MagicMock()
        mock_client.check_connectivity.return_value = (False, 'Auth failed')
        mock_cls = MagicMock(return_value=mock_client)

        with patch(
            'lazy_take_notes.l3_interface_adapters.gateways.openai_llm_client.OpenAICompatLLMClient',
            mock_cls,
        ):
            infra = InfraConfig(llm_provider='openai')
            config = build_app_config({})
            missing_d, missing_i = _preflight_llm(infra, config)

        assert missing_d == []
        assert missing_i == []


class TestPreflightMicrophone:
    def test_no_input_devices_warns(self):
        mock_sd = MagicMock()
        mock_sd.query_devices.return_value = [{'max_input_channels': 0}]
        with patch.dict('sys.modules', {'sounddevice': mock_sd}):
            _preflight_microphone()

    def test_query_devices_exception_warns(self):
        mock_sd = MagicMock()
        mock_sd.query_devices.side_effect = RuntimeError('No audio backend')
        with patch.dict('sys.modules', {'sounddevice': mock_sd}):
            _preflight_microphone()


class TestCliGroup:
    def test_version_flag(self):
        runner = CliRunner()
        result = runner.invoke(cli, ['--version'])
        assert result.exit_code == 0
        assert __version__ in result.output

    def test_no_subcommand_opens_welcome_picker_then_exits(self):
        """Running `lazy-take-notes` with no subcommand opens the WelcomePicker."""
        runner = CliRunner()
        mock_welcome = MagicMock()
        mock_welcome.run.return_value = None  # user cancels

        with patch(_WELCOME_PICKER, return_value=mock_welcome):
            result = runner.invoke(cli, [])

        assert result.exit_code == 0
        mock_welcome.run.assert_called_once()

    def test_welcome_picker_transcribe_cancel_exits_cleanly(self):
        """WelcomePicker selects transcribe; FilePicker cancelled → clean exit."""
        runner = CliRunner()
        mock_welcome = MagicMock()
        mock_welcome.run.return_value = 'transcribe'
        mock_file_picker = MagicMock()
        mock_file_picker.run.return_value = None  # user cancels

        with (
            patch(_WELCOME_PICKER, return_value=mock_welcome),
            patch(_FILE_PICKER, return_value=mock_file_picker),
        ):
            result = runner.invoke(cli, [])

        assert result.exit_code == 0
        mock_file_picker.run.assert_called_once()

    def test_welcome_picker_transcribe_proceeds_to_app(self, tmp_path: Path):
        """WelcomePicker selects transcribe; FilePicker picks file → TranscribeApp runs."""
        runner = CliRunner()
        audio_file = tmp_path / 'audio.wav'
        audio_file.touch()

        mock_welcome = MagicMock()
        mock_welcome.run.return_value = 'transcribe'
        mock_file_picker = MagicMock()
        mock_file_picker.run.return_value = audio_file

        mock_template_picker = MagicMock()
        mock_template_picker.run.return_value = ('default_en', MagicMock())

        mock_template_loader = MagicMock()
        mock_template_loader.load.return_value = MagicMock(
            metadata=MagicMock(locale='en-US'),
            quick_actions=[],
            recognition_hints=[],
        )

        with (
            patch(_WELCOME_PICKER, return_value=mock_welcome),
            patch(_FILE_PICKER, return_value=mock_file_picker),
            patch(_YAML_CFG) as mock_config_cls,
            patch(_YAML_TPL, return_value=mock_template_loader),
            patch(_BUILD) as mock_build,
            patch(_INFRA),
            patch(_PICKER, return_value=mock_template_picker),
            patch(f'{_CLI_HELPERS}.preflight_llm', return_value=([], [])),
            patch('lazy_take_notes.l4_frameworks_and_drivers.apps.transcribe.TranscribeApp') as mock_app_cls,
            patch('lazy_take_notes.l4_frameworks_and_drivers.container.DependencyContainer'),
        ):
            mock_config_cls.return_value.load.return_value = {}
            mock_build.return_value = MagicMock(output=MagicMock(directory=str(tmp_path)))
            result = runner.invoke(cli, [])

        assert result.exit_code == 0
        mock_app_cls.return_value.run.assert_called_once()

    def test_welcome_picker_record_proceeds_to_app(self, tmp_path: Path):
        """WelcomePicker selects record → RecordApp runs."""
        runner = CliRunner()
        mock_welcome = MagicMock()
        mock_welcome.run.return_value = 'record'

        mock_template_picker = MagicMock()
        mock_template_picker.run.return_value = ('default_en', MagicMock())

        mock_template_loader = MagicMock()
        mock_template_loader.load.return_value = MagicMock(
            metadata=MagicMock(locale='en-US'),
            quick_actions=[],
            recognition_hints=[],
        )

        with (
            patch(_WELCOME_PICKER, return_value=mock_welcome),
            patch(_YAML_CFG) as mock_config_cls,
            patch(_YAML_TPL, return_value=mock_template_loader),
            patch(_BUILD) as mock_build,
            patch(_INFRA),
            patch(_PICKER, return_value=mock_template_picker),
            patch(f'{_CLI_HELPERS}.preflight_llm', return_value=([], [])),
            patch(f'{_CLI_HELPERS}.preflight_microphone'),
            patch('lazy_take_notes.l4_frameworks_and_drivers.apps.record.RecordApp') as mock_app_cls,
            patch('lazy_take_notes.l4_frameworks_and_drivers.container.DependencyContainer'),
        ):
            mock_config_cls.return_value.load.return_value = {}
            mock_build.return_value = MagicMock(output=MagicMock(directory=str(tmp_path)))
            result = runner.invoke(cli, [])

        assert result.exit_code == 0
        mock_app_cls.return_value.run.assert_called_once()

    def test_welcome_picker_view_proceeds_to_app(self, tmp_path: Path):
        """WelcomePicker selects view → ViewApp runs → loops back → Esc exits."""
        runner = CliRunner()
        session_dir = tmp_path / '2026-02-22_120000'
        session_dir.mkdir()

        # First call returns 'view', second returns None (Esc) to exit loop
        mock_welcome_1 = MagicMock()
        mock_welcome_1.run.return_value = 'view'
        mock_welcome_2 = MagicMock()
        mock_welcome_2.run.return_value = None

        mock_session_picker = MagicMock()
        mock_session_picker.run.side_effect = [session_dir, None]

        with (
            patch(_WELCOME_PICKER, side_effect=[mock_welcome_1, mock_welcome_2]),
            patch(_YAML_CFG) as mock_config_cls,
            patch(_YAML_TPL),
            patch(_BUILD) as mock_build,
            patch(_INFRA),
            patch(_SESSION_PICKER, return_value=mock_session_picker),
            patch('lazy_take_notes.l4_frameworks_and_drivers.apps.view.ViewApp') as mock_app_cls,
        ):
            mock_config_cls.return_value.load.return_value = {}
            mock_build.return_value = MagicMock(output=MagicMock(directory=str(tmp_path)))
            result = runner.invoke(cli, [])

        assert result.exit_code == 0
        mock_app_cls.return_value.run.assert_called_once()

    def test_config_file_not_found_exits_1(self):
        runner = CliRunner()

        with (
            patch(_YAML_CFG) as mock_config_cls,
            patch(_YAML_TPL),
            patch(_BUILD),
            patch(_INFRA),
        ):
            mock_config_cls.return_value.load.side_effect = FileNotFoundError('not found')
            result = runner.invoke(cli, ['record'])

        assert result.exit_code == 1
        assert 'Error' in result.output


class TestRecordSubcommand:
    def test_picker_returns_none_exits_cleanly(self, tmp_path: Path):
        runner = CliRunner()
        mock_picker = MagicMock()
        mock_picker.run.return_value = None

        with (
            patch(_YAML_CFG) as mock_config_cls,
            patch(_YAML_TPL),
            patch(_BUILD) as mock_build,
            patch(_INFRA),
            patch(_PICKER, return_value=mock_picker),
        ):
            mock_config_cls.return_value.load.return_value = {}
            mock_build.return_value = MagicMock()
            result = runner.invoke(cli, ['record'])

        assert result.exit_code == 0

    def test_template_not_found_exits_1(self, tmp_path: Path):
        runner = CliRunner()
        mock_picker = MagicMock()
        mock_picker.run.return_value = ('nonexistent_template', MagicMock())

        with (
            patch(_YAML_CFG) as mock_config_cls,
            patch(_YAML_TPL) as mock_tpl_cls,
            patch(_BUILD) as mock_build,
            patch(_INFRA),
            patch(_PICKER, return_value=mock_picker),
        ):
            mock_config_cls.return_value.load.return_value = {}
            mock_build.return_value = MagicMock()
            mock_tpl_cls.return_value.load.side_effect = FileNotFoundError('not found')
            result = runner.invoke(cli, ['record'])

        assert result.exit_code == 1
        assert 'Error' in result.output

    def test_normal_run_calls_record_app(self, tmp_path: Path):
        runner = CliRunner()
        mock_picker = MagicMock()
        mock_picker.run.return_value = ('default_en', MagicMock())

        mock_template_loader = MagicMock()
        mock_template_loader.load.return_value = MagicMock(
            metadata=MagicMock(locale='en-US'),
            quick_actions=[],
            recognition_hints=[],
        )

        with (
            patch(_YAML_CFG) as mock_config_cls,
            patch(_YAML_TPL, return_value=mock_template_loader),
            patch(_BUILD) as mock_build,
            patch(_INFRA),
            patch(_PICKER, return_value=mock_picker),
            patch(f'{_CLI_HELPERS}.preflight_llm', return_value=([], [])),
            patch(f'{_CLI_HELPERS}.preflight_microphone'),
            patch('lazy_take_notes.l4_frameworks_and_drivers.apps.record.RecordApp') as mock_app_cls,
            patch('lazy_take_notes.l4_frameworks_and_drivers.container.DependencyContainer'),
        ):
            mock_config_cls.return_value.load.return_value = {}
            mock_build.return_value = MagicMock(output=MagicMock(directory=str(tmp_path)))
            runner.invoke(cli, ['record'])

        mock_app_cls.return_value.run.assert_called_once()

    def test_output_dir_override(self, tmp_path: Path):
        runner = CliRunner()
        custom_dir = tmp_path / 'custom_output'

        mock_picker = MagicMock()
        mock_picker.run.return_value = ('default_en', MagicMock())

        mock_template_loader = MagicMock()
        mock_template_loader.load.return_value = MagicMock(
            metadata=MagicMock(locale='en-US'),
            quick_actions=[],
            recognition_hints=[],
        )

        with (
            patch(_YAML_CFG) as mock_config_cls,
            patch(_YAML_TPL, return_value=mock_template_loader),
            patch(_BUILD) as mock_build,
            patch(_INFRA),
            patch(_PICKER, return_value=mock_picker),
            patch(f'{_CLI_HELPERS}.preflight_llm', return_value=([], [])),
            patch(f'{_CLI_HELPERS}.preflight_microphone'),
            patch('lazy_take_notes.l4_frameworks_and_drivers.apps.record.RecordApp'),
            patch('lazy_take_notes.l4_frameworks_and_drivers.container.DependencyContainer'),
        ):
            mock_config_cls.return_value.load.return_value = {}
            mock_build.return_value = MagicMock(output=MagicMock(directory=str(tmp_path)))
            result = runner.invoke(cli, ['-o', str(custom_dir), 'record'])

        assert result.exit_code == 0
        call_kwargs = mock_config_cls.return_value.load.call_args
        overrides = call_kwargs[1].get('overrides')
        assert overrides == {'output': {'directory': str(custom_dir)}}


class TestTranscribeSubcommand:
    def test_picker_returns_none_exits_cleanly(self, tmp_path: Path):
        runner = CliRunner()
        audio_file = tmp_path / 'audio.wav'
        audio_file.touch()

        mock_picker = MagicMock()
        mock_picker.run.return_value = None

        with (
            patch(_YAML_CFG) as mock_config_cls,
            patch(_YAML_TPL),
            patch(_BUILD) as mock_build,
            patch(_INFRA),
            patch(_PICKER, return_value=mock_picker),
        ):
            mock_config_cls.return_value.load.return_value = {}
            mock_build.return_value = MagicMock()
            result = runner.invoke(cli, ['transcribe', str(audio_file)])

        assert result.exit_code == 0

    def test_template_not_found_exits_1(self, tmp_path: Path):
        runner = CliRunner()
        audio_file = tmp_path / 'audio.wav'
        audio_file.touch()

        mock_picker = MagicMock()
        mock_picker.run.return_value = ('nonexistent_template', MagicMock())

        with (
            patch(_YAML_CFG) as mock_config_cls,
            patch(_YAML_TPL) as mock_tpl_cls,
            patch(_BUILD) as mock_build,
            patch(_INFRA),
            patch(_PICKER, return_value=mock_picker),
        ):
            mock_config_cls.return_value.load.return_value = {}
            mock_build.return_value = MagicMock()
            mock_tpl_cls.return_value.load.side_effect = FileNotFoundError('not found')
            result = runner.invoke(cli, ['transcribe', str(audio_file)])

        assert result.exit_code == 1
        assert 'Error' in result.output

    def test_transcribe_calls_transcribe_app(self, tmp_path: Path):
        runner = CliRunner()
        audio_file = tmp_path / 'audio.wav'
        audio_file.touch()

        mock_picker = MagicMock()
        mock_picker.run.return_value = ('default_en', MagicMock())

        mock_template_loader = MagicMock()
        mock_template_loader.load.return_value = MagicMock(
            metadata=MagicMock(locale='en-US'),
            quick_actions=[],
            recognition_hints=[],
        )

        with (
            patch(_YAML_CFG) as mock_config_cls,
            patch(_YAML_TPL, return_value=mock_template_loader),
            patch(_BUILD) as mock_build,
            patch(_INFRA),
            patch(_PICKER, return_value=mock_picker),
            patch(f'{_CLI_HELPERS}.preflight_llm', return_value=([], [])),
            patch('lazy_take_notes.l4_frameworks_and_drivers.apps.transcribe.TranscribeApp') as mock_app_cls,
            patch('lazy_take_notes.l4_frameworks_and_drivers.container.DependencyContainer'),
        ):
            mock_config_cls.return_value.load.return_value = {}
            mock_build.return_value = MagicMock(output=MagicMock(directory=str(tmp_path)))
            runner.invoke(cli, ['transcribe', str(audio_file)])

        mock_app_cls.return_value.run.assert_called_once()

    def test_no_audio_file_opens_file_picker_cancel_exits_cleanly(self):
        """transcribe with no args opens FilePicker; cancelling exits cleanly."""
        runner = CliRunner()
        mock_file_picker = MagicMock()
        mock_file_picker.run.return_value = None  # user cancels

        with (
            patch(_FILE_PICKER, return_value=mock_file_picker),
        ):
            result = runner.invoke(cli, ['transcribe'])

        assert result.exit_code == 0
        mock_file_picker.run.assert_called_once()

    def test_no_audio_file_opens_file_picker_proceeds_to_app(self, tmp_path: Path):
        """transcribe with no args opens FilePicker; selecting a file runs TranscribeApp."""
        runner = CliRunner()
        audio_file = tmp_path / 'audio.wav'
        audio_file.touch()

        mock_file_picker = MagicMock()
        mock_file_picker.run.return_value = audio_file

        mock_template_picker = MagicMock()
        mock_template_picker.run.return_value = ('default_en', MagicMock())

        mock_template_loader = MagicMock()
        mock_template_loader.load.return_value = MagicMock(
            metadata=MagicMock(locale='en-US'),
            quick_actions=[],
            recognition_hints=[],
        )

        with (
            patch(_FILE_PICKER, return_value=mock_file_picker),
            patch(_YAML_CFG) as mock_config_cls,
            patch(_YAML_TPL, return_value=mock_template_loader),
            patch(_BUILD) as mock_build,
            patch(_INFRA),
            patch(_PICKER, return_value=mock_template_picker),
            patch(f'{_CLI_HELPERS}.preflight_llm', return_value=([], [])),
            patch('lazy_take_notes.l4_frameworks_and_drivers.apps.transcribe.TranscribeApp') as mock_app_cls,
            patch('lazy_take_notes.l4_frameworks_and_drivers.container.DependencyContainer'),
        ):
            mock_config_cls.return_value.load.return_value = {}
            mock_build.return_value = MagicMock(output=MagicMock(directory=str(tmp_path)))
            runner.invoke(cli, ['transcribe'])

        mock_file_picker.run.assert_called_once()
        mock_app_cls.return_value.run.assert_called_once()


class TestViewSubcommand:
    def test_picker_returns_none_exits_cleanly(self, tmp_path: Path):
        runner = CliRunner()

        mock_session_picker = MagicMock()
        mock_session_picker.run.return_value = None

        with (
            patch(_YAML_CFG) as mock_config_cls,
            patch(_YAML_TPL),
            patch(_BUILD) as mock_build,
            patch(_INFRA),
            patch(_SESSION_PICKER, return_value=mock_session_picker),
        ):
            mock_config_cls.return_value.load.return_value = {}
            mock_build.return_value = MagicMock(output=MagicMock(directory=str(tmp_path)))
            result = runner.invoke(cli, ['view'])

        assert result.exit_code == 0

    def test_view_calls_view_app(self, tmp_path: Path):
        runner = CliRunner()
        session_dir = tmp_path / '2026-02-22_120000'
        session_dir.mkdir()

        mock_session_picker = MagicMock()
        mock_session_picker.run.side_effect = [session_dir, None]

        with (
            patch(_YAML_CFG) as mock_config_cls,
            patch(_YAML_TPL),
            patch(_BUILD) as mock_build,
            patch(_INFRA),
            patch(_SESSION_PICKER, return_value=mock_session_picker),
            patch('lazy_take_notes.l4_frameworks_and_drivers.apps.view.ViewApp') as mock_app_cls,
        ):
            mock_config_cls.return_value.load.return_value = {}
            mock_build.return_value = MagicMock(output=MagicMock(directory=str(tmp_path)))
            runner.invoke(cli, ['view'])

        mock_app_cls.return_value.run.assert_called_once()


def _make_session(base_dir: Path, name: str, *, notes: str | None = None, transcript: str = 'hello\n') -> Path:
    session_dir = base_dir / name
    session_dir.mkdir(parents=True)
    (session_dir / 'transcript.txt').write_text(transcript, encoding='utf-8')
    if notes is not None:
        (session_dir / 'notes.md').write_text(notes, encoding='utf-8')
    return session_dir


def _invoke_read(args: list[str], base_dir: Path):
    runner = CliRunner()
    with (
        patch(_YAML_CFG) as mock_config_cls,
        patch(_YAML_TPL),
        patch(_BUILD) as mock_build,
        patch(_INFRA),
    ):
        mock_config_cls.return_value.load.return_value = {}
        mock_build.return_value = MagicMock(output=MagicMock(directory=str(base_dir)))
        return runner.invoke(cli, args)


class TestReadCommands:
    def test_ls_empty(self, tmp_path: Path):
        result = _invoke_read(['ls'], tmp_path)
        assert result.exit_code == 0
        assert 'No sessions found.' in result.output

    def test_ls_lists_newest_first(self, tmp_path: Path):
        _make_session(tmp_path, '2026-02-20_120000')
        _make_session(tmp_path, '2026-02-21_120000', notes='# Notes')

        result = _invoke_read(['ls'], tmp_path)

        assert result.exit_code == 0
        lines = result.output.strip().splitlines()
        assert '2026-02-21_120000' in lines[0]
        assert 'notes' in lines[0]
        assert '2026-02-20_120000' in lines[1]

    def test_ls_json(self, tmp_path: Path):
        _make_session(tmp_path, '2026-02-21_120000', notes='# Notes')

        result = _invoke_read(['ls', '--json'], tmp_path)

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload[0]['name'] == '2026-02-21_120000'
        assert payload[0]['has_notes'] is True

    def test_ls_caps_at_twenty_by_default(self, tmp_path: Path):
        for day in range(1, 23):  # 22 sessions
            _make_session(tmp_path, f'2026-06-{day:02d}_120000')

        result = _invoke_read(['ls'], tmp_path)

        assert result.exit_code == 0
        # Newest 20 shown (days 22..03); the two oldest hidden.
        assert '2026-06-22_120000' in result.output
        assert '2026-06-01_120000' not in result.output
        assert '2026-06-02_120000' not in result.output
        assert '... 2 more' in result.stderr

    def test_ls_all_shows_everything(self, tmp_path: Path):
        for day in range(1, 23):
            _make_session(tmp_path, f'2026-06-{day:02d}_120000')

        result = _invoke_read(['ls', '--all'], tmp_path)

        assert result.exit_code == 0
        assert '2026-06-01_120000' in result.output
        assert 'more' not in result.stderr

    def test_ls_json_is_unlimited(self, tmp_path: Path):
        for day in range(1, 23):
            _make_session(tmp_path, f'2026-06-{day:02d}_120000')

        result = _invoke_read(['ls', '--json'], tmp_path)

        assert result.exit_code == 0
        assert len(json.loads(result.output)) == 22

    def test_transcript_defaults_to_newest(self, tmp_path: Path):
        _make_session(tmp_path, '2026-02-20_120000', transcript='old\n')
        _make_session(tmp_path, '2026-02-21_120000', transcript='newest transcript\n')

        result = _invoke_read(['transcript'], tmp_path)

        assert result.exit_code == 0
        assert result.output == 'newest transcript\n'

    def test_transcript_by_exact_name(self, tmp_path: Path):
        _make_session(tmp_path, '2026-02-20_120000_a', transcript='AAA\n')
        _make_session(tmp_path, '2026-02-21_120000_b', transcript='BBB\n')

        result = _invoke_read(['transcript', '2026-02-20_120000_a'], tmp_path)

        assert result.exit_code == 0
        assert result.output == 'AAA\n'

    def test_transcript_by_substring(self, tmp_path: Path):
        _make_session(tmp_path, '2026-02-20_120000_standup', transcript='standup notes\n')
        _make_session(tmp_path, '2026-02-21_120000_retro', transcript='retro notes\n')

        result = _invoke_read(['transcript', 'standup'], tmp_path)

        assert result.exit_code == 0
        assert result.output == 'standup notes\n'

    def test_transcript_no_sessions_errors(self, tmp_path: Path):
        result = _invoke_read(['transcript'], tmp_path)
        assert result.exit_code != 0
        assert 'No sessions found.' in result.output

    def test_transcript_ambiguous_errors(self, tmp_path: Path):
        _make_session(tmp_path, '2026-02-20_120000_sync')
        _make_session(tmp_path, '2026-02-21_120000_sync')

        result = _invoke_read(['transcript', 'sync'], tmp_path)

        assert result.exit_code != 0
        assert 'multiple sessions' in result.output

    def test_transcript_no_match_errors(self, tmp_path: Path):
        _make_session(tmp_path, '2026-02-20_120000')

        result = _invoke_read(['transcript', 'nope'], tmp_path)

        assert result.exit_code != 0
        assert 'No session matching' in result.output

    def test_notes_defaults_to_newest(self, tmp_path: Path):
        _make_session(tmp_path, '2026-02-21_120000', notes='# Summary\nDone.')

        result = _invoke_read(['notes'], tmp_path)

        assert result.exit_code == 0
        assert result.output == '# Summary\nDone.'

    def test_notes_missing_errors(self, tmp_path: Path):
        _make_session(tmp_path, '2026-02-21_120000')

        result = _invoke_read(['notes'], tmp_path)

        assert result.exit_code != 0
        assert 'has no notes' in result.output


_HEADLESS = 'lazy_take_notes.l4_frameworks_and_drivers.headless'


class TestHeadlessFlag:
    def test_record_headless_requires_language(self, tmp_path: Path):
        runner = CliRunner()
        result = runner.invoke(cli, ['record', '--headless'])
        assert result.exit_code != 0
        assert 'requires --language' in result.output

    def test_record_headless_custom_template(self, tmp_path: Path):
        runner = CliRunner()
        with patch(f'{_HEADLESS}.run_record_headless') as mock_run:
            result = runner.invoke(cli, ['record', '--headless', '--template', 'sprint_retro_en'])
        assert result.exit_code == 0
        assert mock_run.call_args.kwargs['template_name'] == 'sprint_retro_en'

    def test_record_headless_language(self, tmp_path: Path):
        runner = CliRunner()
        with patch(f'{_HEADLESS}.run_record_headless') as mock_run:
            result = runner.invoke(cli, ['record', '--headless', '--language', 'zh-TW'])
        assert result.exit_code == 0
        assert mock_run.call_args.kwargs['language'] == 'zh-TW'

    def test_record_headless_mute_mic(self, tmp_path: Path):
        runner = CliRunner()
        with patch(f'{_HEADLESS}.run_record_headless') as mock_run:
            result = runner.invoke(cli, ['record', '--headless', '--language', 'en', '--mute-mic'])
        assert result.exit_code == 0
        assert mock_run.call_args.kwargs['mute_mic'] is True

    def test_template_and_language_are_mutually_exclusive(self):
        runner = CliRunner()
        result = runner.invoke(cli, ['record', '--template', 'default_en', '--language', 'en'])
        assert result.exit_code != 0
        assert 'not both' in result.output

    def test_record_mute_mic_tui(self, tmp_path: Path):
        runner = CliRunner()
        mock_picker = MagicMock()
        mock_picker.run.return_value = ('default_en', MagicMock())
        mock_template_loader = MagicMock()
        mock_template_loader.load.return_value = MagicMock(
            metadata=MagicMock(locale='en-US'), quick_actions=[], recognition_hints=[]
        )
        with (
            patch(_YAML_CFG) as mock_config_cls,
            patch(_YAML_TPL, return_value=mock_template_loader),
            patch(_BUILD) as mock_build,
            patch(_INFRA),
            patch(_PICKER, return_value=mock_picker),
            patch(f'{_CLI_HELPERS}.preflight_llm', return_value=([], [])),
            patch(f'{_CLI_HELPERS}.preflight_microphone'),
            patch('lazy_take_notes.l4_frameworks_and_drivers.apps.record.RecordApp'),
            patch('lazy_take_notes.l4_frameworks_and_drivers.container.DependencyContainer') as mock_container_cls,
        ):
            mock_config_cls.return_value.load.return_value = {}
            mock_build.return_value = MagicMock(output=MagicMock(directory=str(tmp_path)))
            result = runner.invoke(cli, ['record', '--mute-mic'])

        assert result.exit_code == 0
        assert mock_container_cls.return_value.audio_source.mic_muted is True

    def test_transcribe_headless_routes_to_runner(self, tmp_path: Path):
        runner = CliRunner()
        audio = tmp_path / 'a.wav'
        audio.touch()
        with patch(f'{_HEADLESS}.run_transcribe_headless') as mock_run:
            result = runner.invoke(cli, ['transcribe', '--headless', str(audio), '--language', 'en'])
        assert result.exit_code == 0
        assert mock_run.call_args.kwargs['audio_path'] == audio
        assert mock_run.call_args.kwargs['language'] == 'en'

    def test_transcribe_headless_without_file_errors(self):
        runner = CliRunner()
        result = runner.invoke(cli, ['transcribe', '--headless'])
        assert result.exit_code != 0
        assert 'requires an audio file' in result.output

    def test_transcribe_headless_requires_language(self, tmp_path: Path):
        runner = CliRunner()
        audio = tmp_path / 'a.wav'
        audio.touch()
        result = runner.invoke(cli, ['transcribe', '--headless', str(audio)])
        assert result.exit_code != 0
        assert 'requires --language' in result.output


def _write_status(
    base_dir: Path,
    name: str,
    *,
    state: str = 'recording',
    pid: int | None = None,
    segment_count: int = 3,
    digest_count: int = 1,
    error: str = '',
) -> Path:
    from lazy_take_notes.l1_entities.session_status import SessionStatus
    from lazy_take_notes.l3_interface_adapters.gateways.session_status import write_status

    session_dir = base_dir / name
    session_dir.mkdir(parents=True)
    write_status(
        session_dir,
        SessionStatus(
            state=state,
            pid=os.getpid() if pid is None else pid,
            started_at='2026-06-06T09:00:00',
            updated_at='2026-06-06T09:01:00',
            segment_count=segment_count,
            digest_count=digest_count,
            error=error,
        ),
    )
    return session_dir


class TestStatusCommand:
    def test_shows_newest_session(self, tmp_path: Path):
        _write_status(tmp_path, '2026-06-01_090000', segment_count=1)
        _write_status(tmp_path, '2026-06-05_140000', segment_count=9)

        result = _invoke_read(['status'], tmp_path)

        assert result.exit_code == 0
        assert '2026-06-05_140000' in result.output
        assert 'segments: 9' in result.output
        assert f'pid:      {os.getpid()}' in result.output

    def test_crashed_session_flagged(self, tmp_path: Path):
        _write_status(tmp_path, '2026-06-05_140000', state='recording', pid=2_000_000_000)

        result = _invoke_read(['status'], tmp_path)

        assert result.exit_code == 0
        assert 'crashed' in result.output

    def test_no_session_errors(self, tmp_path: Path):
        result = _invoke_read(['status'], tmp_path)
        assert result.exit_code != 0
        assert 'No headless session found.' in result.output

    def test_status_by_name(self, tmp_path: Path):
        _write_status(tmp_path, '2026-06-01_090000', segment_count=1)
        _write_status(tmp_path, '2026-06-05_140000', segment_count=9)

        result = _invoke_read(['status', '2026-06-01_090000'], tmp_path)

        assert result.exit_code == 0
        assert 'segments: 1' in result.output

    def test_status_corrupt_file_errors(self, tmp_path: Path):
        from lazy_take_notes.l3_interface_adapters.gateways.session_status import STATUS_FILE

        session_dir = tmp_path / '2026-06-05_140000'
        session_dir.mkdir()
        (session_dir / STATUS_FILE).write_text('{not json', encoding='utf-8')

        result = _invoke_read(['status'], tmp_path)

        assert result.exit_code != 0
        assert 'missing or unreadable' in result.output

    def test_error_state_shown(self, tmp_path: Path):
        _write_status(tmp_path, '2026-06-05_140000', state='error', error='boom')

        result = _invoke_read(['status'], tmp_path)

        assert result.exit_code == 0
        assert 'error:    boom' in result.output


class TestTemplateSelection:
    def _loader(self):
        from lazy_take_notes.l3_interface_adapters.gateways.yaml_template_loader import YamlTemplateLoader

        return YamlTemplateLoader()

    def test_resolve_language_match(self):
        from lazy_take_notes.l4_frameworks_and_drivers.cli_helpers import resolve_language

        template = resolve_language(self._loader(), 'zh-TW')
        assert template.metadata.key == 'default_zh_tw'
        assert template.metadata.locale == 'zh-TW'

    def test_resolve_language_is_case_insensitive(self):
        from lazy_take_notes.l4_frameworks_and_drivers.cli_helpers import resolve_language

        assert resolve_language(self._loader(), 'EN').metadata.key == 'default_en'

    def test_resolve_language_unsupported_stops(self):
        from lazy_take_notes.l4_frameworks_and_drivers.cli_helpers import resolve_language

        with pytest.raises(click.ClickException, match='Unsupported language'):
            resolve_language(self._loader(), 'xx')

    def test_resolve_language_primary_subtag_fallback(self):
        from lazy_take_notes.l4_frameworks_and_drivers.cli_helpers import resolve_language

        # en-US has no exact template; falls back to the unambiguous primary subtag 'en'.
        assert resolve_language(self._loader(), 'en-US').metadata.key == 'default_en'

    def test_resolve_language_ambiguous_subtag_stops(self):
        from lazy_take_notes.l4_frameworks_and_drivers.cli_helpers import resolve_language

        # 'zh' matches both zh-TW and zh-min-nan -> ambiguous -> fail loud.
        with pytest.raises(click.ClickException, match='Unsupported language'):
            resolve_language(self._loader(), 'zh')

    def test_load_template_key_valid(self):
        from lazy_take_notes.l4_frameworks_and_drivers.cli_helpers import load_template_key

        assert load_template_key(self._loader(), 'sprint_retro_en').metadata.key == 'sprint_retro_en'

    def test_load_template_key_unknown_stops(self):
        from lazy_take_notes.l4_frameworks_and_drivers.cli_helpers import load_template_key

        with pytest.raises(click.ClickException, match='Unknown template'):
            load_template_key(self._loader(), 'no_such_template')

    def test_select_template_prefers_language(self):
        from lazy_take_notes.l4_frameworks_and_drivers.cli_helpers import select_template

        template = select_template(self._loader(), template_name=None, language='en', show_builtins=True)
        assert template.metadata.key == 'default_en'

    def test_select_template_by_key(self):
        from lazy_take_notes.l4_frameworks_and_drivers.cli_helpers import select_template

        template = select_template(self._loader(), template_name='lecture_notes_en', language=None, show_builtins=True)
        assert template.metadata.key == 'lecture_notes_en'

    def test_select_template_falls_back_to_picker(self, monkeypatch):
        from lazy_take_notes.l4_frameworks_and_drivers import cli_helpers as ch

        sentinel = object()
        monkeypatch.setattr(ch, 'pick_template', lambda loader, show_builtins: sentinel)
        result = ch.select_template(self._loader(), template_name=None, language=None, show_builtins=True)
        assert result is sentinel

    def test_set_mic_muted_on_supporting_source(self):
        from lazy_take_notes.l4_frameworks_and_drivers.cli_helpers import set_mic_muted

        class _Src:
            mic_muted = False

        src = _Src()
        set_mic_muted(src, True)
        assert src.mic_muted is True

    def test_set_mic_muted_noop_without_attr(self):
        from lazy_take_notes.l4_frameworks_and_drivers.cli_helpers import set_mic_muted

        set_mic_muted(object(), True)  # must not raise


class TestLoadPlugins:
    def test_valid_plugin_registered_as_subcommand(self):
        """A valid click.Command entry point gets added to the group."""
        fake_command = click.Command('greet', callback=lambda: None)
        mock_ep = MagicMock()
        mock_ep.name = 'greet'
        mock_ep.load.return_value = fake_command

        group = click.Group('test')
        with patch('lazy_take_notes.l4_frameworks_and_drivers.cli.entry_points', return_value=[mock_ep]):
            _load_plugins(group)

        assert 'greet' in group.commands
        assert group.commands['greet'] is fake_command

    def test_non_command_entry_point_skipped(self):
        """An entry point that resolves to a non-Command object is skipped with a warning."""
        mock_ep = MagicMock()
        mock_ep.name = 'bad'
        mock_ep.load.return_value = 'not a command'

        group = click.Group('test')
        with patch('lazy_take_notes.l4_frameworks_and_drivers.cli.entry_points', return_value=[mock_ep]):
            _load_plugins(group)

        assert 'bad' not in group.commands

    def test_broken_plugin_does_not_crash_cli(self):
        """A plugin that raises on load prints a warning, doesn't crash."""
        mock_ep = MagicMock()
        mock_ep.name = 'broken'
        mock_ep.load.side_effect = ImportError('no such module')

        group = click.Group('test')
        with patch('lazy_take_notes.l4_frameworks_and_drivers.cli.entry_points', return_value=[mock_ep]):
            _load_plugins(group)

        assert 'broken' not in group.commands

    def test_no_plugins_is_noop(self):
        """When no plugins are installed, the group is unchanged."""
        group = click.Group('test')
        original_commands = dict(group.commands)

        with patch('lazy_take_notes.l4_frameworks_and_drivers.cli.entry_points', return_value=[]):
            _load_plugins(group)

        assert group.commands == original_commands

    def test_plugin_subcommand_visible_in_help(self):
        """A registered plugin shows up in --help output."""
        fake_command = click.Command('my-source', callback=lambda: None, help='Import from an external source.')
        mock_ep = MagicMock()
        mock_ep.name = 'my-source'
        mock_ep.load.return_value = fake_command

        with patch('lazy_take_notes.l4_frameworks_and_drivers.cli.entry_points', return_value=[mock_ep]):
            _load_plugins(cli)

        runner = CliRunner()
        result = runner.invoke(cli, ['--help'])
        assert 'my-source' in result.output

        # Cleanup: remove the test command so it doesn't leak into other tests
        cli.commands.pop('my-source', None)


_MANIFEST = 'lazy_take_notes.l4_frameworks_and_drivers.plugin_manifest'


@pytest.fixture()
def plugin_dir(tmp_path: Path):
    """Patch plugin manifest paths to use tmp_path, yield the dir."""
    with (
        patch(f'{_MANIFEST}.PLUGINS_YAML', tmp_path / 'plugins.yaml'),
        patch(f'{_MANIFEST}.PLUGINS_TXT', tmp_path / 'plugins.txt'),
    ):
        yield tmp_path


class TestPluginCommands:
    """Tests for `take-note plugin add/remove/list` CLI commands."""

    def test_plugin_add_success(self, plugin_dir: Path):
        runner = CliRunner()
        with patch(f'{_MANIFEST}.validate_spec', return_value=(True, '')):
            result = runner.invoke(cli, ['plugin', 'add', 'my-plugin @ git+https://example.com'])
        assert result.exit_code == 0
        assert 'my-plugin added' in result.output

    def test_plugin_add_validation_failure(self, plugin_dir: Path):
        runner = CliRunner()
        with patch(f'{_MANIFEST}.validate_spec', return_value=(False, 'No matching distribution')):
            result = runner.invoke(cli, ['plugin', 'add', 'nonexistent'])
        assert result.exit_code == 1

    def test_plugin_remove_success(self, plugin_dir: Path):
        runner = CliRunner()
        (plugin_dir / 'plugins.yaml').write_text('plugins:\n  - my-plugin\n')
        result = runner.invoke(cli, ['plugin', 'remove', 'my-plugin'])
        assert result.exit_code == 0
        assert 'removed' in result.output

    def test_plugin_remove_not_found(self, plugin_dir: Path):
        runner = CliRunner()
        result = runner.invoke(cli, ['plugin', 'remove', 'nonexistent'])
        assert result.exit_code == 0
        assert 'not found' in result.output

    def test_plugin_list_empty(self, plugin_dir: Path):
        runner = CliRunner()
        result = runner.invoke(cli, ['plugin', 'list'])
        assert result.exit_code == 0
        assert 'No plugins installed' in result.output

    def test_plugin_list_shows_installed(self, plugin_dir: Path):
        runner = CliRunner()
        (plugin_dir / 'plugins.yaml').write_text('plugins:\n  - plugin-a\n  - plugin-b\n')
        result = runner.invoke(cli, ['plugin', 'list'])
        assert result.exit_code == 0
        assert 'plugin-a' in result.output
        assert 'plugin-b' in result.output


class TestSessionEntryPointsReturnArtifacts:
    """run_transcribe / run_record hand SessionArtifacts back to callers."""

    def _ctx(self, output_dir: Path) -> click.Context:
        ctx = click.Context(click.Command('x'))
        ctx.obj = {'config_path': None, 'output_dir': str(output_dir)}
        return ctx

    def _enter_common(self, stack: ExitStack, out_dir: Path) -> None:
        infra = MagicMock(show_builtin_templates=True)
        for cm in (
            patch(f'{_CLI_HELPERS}.load_config', return_value=(MagicMock(), infra, MagicMock())),
            patch(f'{_CLI_HELPERS}.select_template', return_value=MagicMock()),
            patch(f'{_CLI_HELPERS}.make_session_dir', return_value=out_dir),
            patch(f'{_CLI_HELPERS}.preflight_llm', return_value=([], [])),
            patch('lazy_take_notes.l4_frameworks_and_drivers.container.DependencyContainer'),
        ):
            stack.enter_context(cm)

    def test_run_transcribe_returns_artifacts_with_notes_path(self, tmp_path: Path):
        from lazy_take_notes.l1_entities.session_files import NOTES, SessionArtifacts
        from lazy_take_notes.l4_frameworks_and_drivers.cli_helpers import run_transcribe

        out_dir = tmp_path / 'session'
        out_dir.mkdir()
        notes = out_dir / NOTES.name
        mock_app = MagicMock()
        mock_app.run.side_effect = lambda: notes.write_text('# notes', encoding='utf-8')

        with ExitStack() as stack:
            self._enter_common(stack, out_dir)
            stack.enter_context(
                patch('lazy_take_notes.l4_frameworks_and_drivers.apps.transcribe.TranscribeApp', return_value=mock_app)
            )
            result = run_transcribe(self._ctx(tmp_path))

        assert isinstance(result, SessionArtifacts)
        assert result.session_dir == out_dir
        assert result.notes_path == notes

    def test_run_transcribe_artifacts_notes_path_none_when_not_written(self, tmp_path: Path):
        from lazy_take_notes.l1_entities.session_files import SessionArtifacts
        from lazy_take_notes.l4_frameworks_and_drivers.cli_helpers import run_transcribe

        out_dir = tmp_path / 'session'
        out_dir.mkdir()

        with ExitStack() as stack:
            self._enter_common(stack, out_dir)
            stack.enter_context(
                patch(
                    'lazy_take_notes.l4_frameworks_and_drivers.apps.transcribe.TranscribeApp',
                    return_value=MagicMock(),
                )
            )
            result = run_transcribe(self._ctx(tmp_path))

        assert isinstance(result, SessionArtifacts)
        assert result.session_dir == out_dir
        assert result.notes_path is None

    def test_run_transcribe_returns_none_on_template_cancel(self, tmp_path: Path):
        from lazy_take_notes.l4_frameworks_and_drivers.cli_helpers import run_transcribe

        with (
            patch(
                f'{_CLI_HELPERS}.load_config',
                return_value=(MagicMock(), MagicMock(show_builtin_templates=True), MagicMock()),
            ),
            patch(f'{_CLI_HELPERS}.select_template', return_value=None),
        ):
            result = run_transcribe(self._ctx(tmp_path))

        assert result is None

    def test_run_record_returns_artifacts_with_notes_path(self, tmp_path: Path):
        from lazy_take_notes.l1_entities.session_files import NOTES, SessionArtifacts
        from lazy_take_notes.l4_frameworks_and_drivers.cli_helpers import run_record

        out_dir = tmp_path / 'session'
        out_dir.mkdir()
        notes = out_dir / NOTES.name
        mock_app = MagicMock()
        mock_app.run.side_effect = lambda: notes.write_text('# notes', encoding='utf-8')

        with ExitStack() as stack:
            self._enter_common(stack, out_dir)
            stack.enter_context(patch(f'{_CLI_HELPERS}.preflight_microphone'))
            stack.enter_context(patch('lazy_take_notes.l4_frameworks_and_drivers.keep_awake.keep_awake'))
            stack.enter_context(
                patch('lazy_take_notes.l4_frameworks_and_drivers.apps.record.RecordApp', return_value=mock_app)
            )
            result = run_record(self._ctx(tmp_path))

        assert isinstance(result, SessionArtifacts)
        assert result.session_dir == out_dir
        assert result.notes_path == notes
