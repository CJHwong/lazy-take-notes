"""Tests for CLI entry point — patches deferred imports at source module level."""

from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from lazy_take_notes import __version__
from lazy_take_notes.l4_frameworks_and_drivers.cli import (
    _make_session_dir,  # noqa: PLC2701 -- testing private helper
    _preflight_microphone,  # noqa: PLC2701 -- testing private helper
    _preflight_ollama,  # noqa: PLC2701 -- testing private helper
    cli,
)
from lazy_take_notes.l4_frameworks_and_drivers.infra_config import InfraConfig, build_app_config

# Patch targets at SOURCE module level (not cli module) because cli() uses
# deferred `from X import Y` which creates local bindings that bypass
# module-level attribute patches.
_YAML_CFG = 'lazy_take_notes.l3_interface_adapters.gateways.yaml_config_loader.YamlConfigLoader'
_YAML_TPL = 'lazy_take_notes.l3_interface_adapters.gateways.yaml_template_loader.YamlTemplateLoader'
_BUILD = 'lazy_take_notes.l4_frameworks_and_drivers.infra_config.build_app_config'
_INFRA = 'lazy_take_notes.l4_frameworks_and_drivers.infra_config.InfraConfig'
_PICKER = 'lazy_take_notes.l4_frameworks_and_drivers.template_picker.TemplatePicker'
_CLI = 'lazy_take_notes.l4_frameworks_and_drivers.cli'


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


class TestPreflightOllama:
    def test_unreachable_returns_empty_lists(self):
        mock_client = MagicMock()
        mock_client.check_connectivity.return_value = (False, 'Connection refused')
        mock_cls = MagicMock(return_value=mock_client)

        with patch(
            'lazy_take_notes.l3_interface_adapters.gateways.ollama_llm_client.OllamaLLMClient',
            mock_cls,
        ):
            infra = InfraConfig()
            config = build_app_config({})
            missing_d, missing_i = _preflight_ollama(infra, config)

        assert missing_d == []
        assert missing_i == []

    def test_all_models_present_returns_empty(self):
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
            missing_d, missing_i = _preflight_ollama(infra, config)

        assert missing_d == []
        assert missing_i == []

    def test_missing_digest_model_returned(self):
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
            missing_d, _missing_i = _preflight_ollama(infra, config)

        assert missing_d == [config.digest.model]


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


class TestCliCommand:
    def test_version_flag(self):
        runner = CliRunner()
        result = runner.invoke(cli, ['--version'])
        assert result.exit_code == 0
        assert __version__ in result.output

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
            result = runner.invoke(cli, [])

        assert result.exit_code == 0

    def test_config_file_not_found_exits_1(self, tmp_path: Path):
        runner = CliRunner()

        with (
            patch(_YAML_CFG) as mock_config_cls,
            patch(_YAML_TPL),
            patch(_BUILD),
            patch(_INFRA),
        ):
            mock_config_cls.return_value.load.side_effect = FileNotFoundError('not found')
            result = runner.invoke(cli, [])

        assert result.exit_code == 1
        assert 'Error' in result.output

    def test_audio_file_mode_calls_run_batch(self, tmp_path: Path):
        runner = CliRunner()
        audio_file = tmp_path / 'audio.wav'
        audio_file.touch()

        mock_picker = MagicMock()
        mock_picker.run.return_value = ('default_en', MagicMock())

        mock_template_loader = MagicMock()
        mock_template_loader.load.return_value = MagicMock(metadata=MagicMock(locale='en-US'))

        with (
            patch(_YAML_CFG) as mock_config_cls,
            patch(_YAML_TPL, return_value=mock_template_loader),
            patch(_BUILD) as mock_build,
            patch(_INFRA),
            patch(_PICKER, return_value=mock_picker),
            patch('lazy_take_notes.l4_frameworks_and_drivers.batch_runner.run_batch') as mock_batch,
        ):
            mock_config_cls.return_value.load.return_value = {}
            mock_build.return_value = MagicMock(output=MagicMock(directory=str(tmp_path)))
            runner.invoke(cli, ['-f', str(audio_file)])

        assert mock_batch.called

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
            patch(f'{_CLI}._preflight_ollama', return_value=([], [])),
            patch(f'{_CLI}._preflight_microphone'),
            patch('lazy_take_notes.l4_frameworks_and_drivers.app.App'),
            patch('lazy_take_notes.l4_frameworks_and_drivers.container.DependencyContainer'),
        ):
            mock_config_cls.return_value.load.return_value = {}
            mock_build.return_value = MagicMock(output=MagicMock(directory=str(tmp_path)))
            result = runner.invoke(cli, ['-o', str(custom_dir)])

        assert result.exit_code == 0
        call_kwargs = mock_config_cls.return_value.load.call_args
        overrides = call_kwargs[1].get('overrides')
        assert overrides == {'output': {'directory': str(custom_dir)}}

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
            result = runner.invoke(cli, [])

        assert result.exit_code == 1
        assert 'Error' in result.output

    def test_normal_run_calls_app_run(self, tmp_path: Path):
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
            patch(f'{_CLI}._preflight_ollama', return_value=([], [])),
            patch(f'{_CLI}._preflight_microphone'),
            patch('lazy_take_notes.l4_frameworks_and_drivers.app.App') as mock_app_cls,
            patch('lazy_take_notes.l4_frameworks_and_drivers.container.DependencyContainer'),
        ):
            mock_config_cls.return_value.load.return_value = {}
            mock_build.return_value = MagicMock(output=MagicMock(directory=str(tmp_path)))
            runner.invoke(cli, [])

        mock_app_cls.return_value.run.assert_called_once()
