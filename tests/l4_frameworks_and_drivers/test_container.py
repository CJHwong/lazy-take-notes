"""Tests for the dependency container."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from lazy_take_notes.l3_interface_adapters.gateways.yaml_template_loader import YamlTemplateLoader
from lazy_take_notes.l4_frameworks_and_drivers.config import InfraConfig, build_app_config
from lazy_take_notes.l4_frameworks_and_drivers.container import DependencyContainer


class TestDependencyContainer:
    @patch('lazy_take_notes.l4_frameworks_and_drivers.container.SubprocessWhisperTranscriber')
    @patch('lazy_take_notes.l4_frameworks_and_drivers.container.DependencyContainer._build_mixed_source')
    def test_creates_all_components(self, mock_audio, mock_whisper, tmp_path: Path):
        config = build_app_config({})
        template = YamlTemplateLoader().load('default_zh_tw')
        output_dir = tmp_path / 'output'

        container = DependencyContainer(config, template, output_dir)

        assert container.config is config
        assert container.template is template
        assert container.persistence is not None
        assert container.llm_client is not None
        assert container.controller is not None

    @patch('lazy_take_notes.l4_frameworks_and_drivers.container.SubprocessWhisperTranscriber')
    @patch('lazy_take_notes.l4_frameworks_and_drivers.container.DependencyContainer._build_mixed_source')
    def test_openai_provider_creates_openai_client(self, mock_audio, mock_whisper, tmp_path: Path):
        from lazy_take_notes.l3_interface_adapters.gateways.openai_llm_client import OpenAICompatLLMClient

        config = build_app_config({})
        template = YamlTemplateLoader().load('default_zh_tw')
        infra = InfraConfig(llm_provider='openai')

        container = DependencyContainer(config, template, tmp_path, infra=infra)

        assert isinstance(container.llm_client, OpenAICompatLLMClient)

    def test_config_loader_factory(self):
        loader = DependencyContainer.config_loader()
        assert hasattr(loader, 'load')

    def test_template_loader_factory(self):
        loader = DependencyContainer.template_loader()
        assert hasattr(loader, 'load')
        assert hasattr(loader, 'list_templates')


class TestDependencyContainerOverrides:
    """Plugin-supplied overrides bypass the default factories."""

    def test_llm_client_override_used(self, tmp_path: Path):
        config = build_app_config({})
        template = YamlTemplateLoader().load('default_zh_tw')
        fake_llm = MagicMock()

        with patch('lazy_take_notes.l4_frameworks_and_drivers.container.SubprocessWhisperTranscriber'):
            container = DependencyContainer(
                config,
                template,
                tmp_path,
                build_audio=False,
                llm_client=fake_llm,
            )

        assert container.llm_client is fake_llm

    def test_transcriber_override_used(self, tmp_path: Path):
        config = build_app_config({})
        template = YamlTemplateLoader().load('default_zh_tw')
        fake_transcriber = MagicMock()

        container = DependencyContainer(
            config,
            template,
            tmp_path,
            build_audio=False,
            transcriber=fake_transcriber,
        )

        assert container.transcriber is fake_transcriber

    def test_audio_source_override_used(self, tmp_path: Path):
        config = build_app_config({})
        template = YamlTemplateLoader().load('default_zh_tw')
        fake_audio = MagicMock()

        with patch('lazy_take_notes.l4_frameworks_and_drivers.container.SubprocessWhisperTranscriber'):
            container = DependencyContainer(
                config,
                template,
                tmp_path,
                build_audio=True,
                audio_source=fake_audio,
            )

        assert container.audio_source is fake_audio

    def test_audio_source_override_skips_build_mixed_source(self, tmp_path: Path):
        config = build_app_config({})
        template = YamlTemplateLoader().load('default_zh_tw')
        fake_audio = MagicMock()

        with (
            patch('lazy_take_notes.l4_frameworks_and_drivers.container.SubprocessWhisperTranscriber'),
            patch.object(DependencyContainer, '_build_mixed_source') as mock_build,
        ):
            DependencyContainer(
                config,
                template,
                tmp_path,
                build_audio=True,
                audio_source=fake_audio,
            )

        mock_build.assert_not_called()
