"""Tests for the dependency container."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lazy_take_notes.l3_interface_adapters.gateways.yaml_template_loader import YamlTemplateLoader
from lazy_take_notes.l4_frameworks_and_drivers.config import InfraConfig, build_app_config
from lazy_take_notes.l4_frameworks_and_drivers.container import (
    DependencyContainer,
    TranscriptionBackend,
    _load_plugin_provider,  # noqa: PLC2701 -- testing private helper
)


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


class TestPluginProviderDiscovery:
    """Plugin LLM providers discovered via entry_points."""

    def test_unknown_provider_raises_value_error(self):
        from lazy_take_notes.l4_frameworks_and_drivers.container import BUILTIN_LLM_PROVIDERS, LLM_PROVIDERS_GROUP

        infra = InfraConfig(llm_provider='nonexistent')
        with pytest.raises(ValueError, match='Unknown provider'):
            _load_plugin_provider('nonexistent', LLM_PROVIDERS_GROUP, BUILTIN_LLM_PROVIDERS, infra)

    def test_plugin_provider_factory_called(self):
        from lazy_take_notes.l4_frameworks_and_drivers.container import BUILTIN_LLM_PROVIDERS, LLM_PROVIDERS_GROUP

        fake_client = MagicMock()
        fake_factory = MagicMock(return_value=fake_client)
        fake_ep = MagicMock()
        fake_ep.name = 'test-provider'
        fake_ep.load.return_value = fake_factory

        infra = InfraConfig(llm_provider='test-provider')
        with patch(
            'lazy_take_notes.l4_frameworks_and_drivers.container.entry_points',
            return_value=[fake_ep],
        ):
            result = _load_plugin_provider('test-provider', LLM_PROVIDERS_GROUP, BUILTIN_LLM_PROVIDERS, infra)

        fake_factory.assert_called_once_with(infra)
        assert result is fake_client

    def test_resolve_falls_through_to_plugin(self, tmp_path: Path):
        fake_client = MagicMock()
        fake_factory = MagicMock(return_value=fake_client)
        fake_ep = MagicMock()
        fake_ep.name = 'my-plugin'
        fake_ep.load.return_value = fake_factory

        infra = InfraConfig(llm_provider='my-plugin')
        with patch(
            'lazy_take_notes.l4_frameworks_and_drivers.container.entry_points',
            return_value=[fake_ep],
        ):
            result = DependencyContainer.resolve_llm_client(infra)

        assert result is fake_client


class TestTranscriptionBackendResolution:
    """Transcription provider dispatch — built-in + plugin discovery."""

    def test_whisper_cpp_returns_correct_backend(self):
        infra = InfraConfig(transcription_provider='whisper-cpp')
        backend = DependencyContainer.resolve_transcription_backend(infra)

        assert isinstance(backend, TranscriptionBackend)
        # Factory produces SubprocessWhisperTranscriber
        transcriber = backend.create_transcriber()
        from lazy_take_notes.l3_interface_adapters.gateways.subprocess_whisper_transcriber import (
            SubprocessWhisperTranscriber,
        )

        assert isinstance(transcriber, SubprocessWhisperTranscriber)

        # Factory produces HfModelResolver
        from lazy_take_notes.l3_interface_adapters.gateways.hf_model_resolver import HfModelResolver

        resolver = backend.create_model_resolver(None)
        assert isinstance(resolver, HfModelResolver)

    def test_whisper_cpp_resolver_passes_on_progress(self):
        infra = InfraConfig(transcription_provider='whisper-cpp')
        backend = DependencyContainer.resolve_transcription_backend(infra)

        progress_calls = []
        resolver = backend.create_model_resolver(lambda p: progress_calls.append(p))
        from lazy_take_notes.l3_interface_adapters.gateways.hf_model_resolver import HfModelResolver

        assert isinstance(resolver, HfModelResolver)
        assert resolver._on_progress is not None

    def test_unknown_provider_raises_value_error(self):
        from lazy_take_notes.l4_frameworks_and_drivers.container import (
            BUILTIN_TRANSCRIPTION_PROVIDERS,
            TRANSCRIPTION_PROVIDERS_GROUP,
        )

        infra = InfraConfig(transcription_provider='nonexistent')
        with pytest.raises(ValueError, match='Unknown provider'):
            _load_plugin_provider(
                'nonexistent',
                TRANSCRIPTION_PROVIDERS_GROUP,
                BUILTIN_TRANSCRIPTION_PROVIDERS,
                infra,
            )

    def test_plugin_provider_factory_called(self):
        from lazy_take_notes.l4_frameworks_and_drivers.container import (
            BUILTIN_TRANSCRIPTION_PROVIDERS,
            TRANSCRIPTION_PROVIDERS_GROUP,
        )

        fake_backend = MagicMock(spec=TranscriptionBackend)
        fake_factory = MagicMock(return_value=fake_backend)
        fake_ep = MagicMock()
        fake_ep.name = 'test-whisper'
        fake_ep.load.return_value = fake_factory

        infra = InfraConfig(transcription_provider='test-whisper')
        with patch(
            'lazy_take_notes.l4_frameworks_and_drivers.container.entry_points',
            return_value=[fake_ep],
        ):
            result = _load_plugin_provider(
                'test-whisper',
                TRANSCRIPTION_PROVIDERS_GROUP,
                BUILTIN_TRANSCRIPTION_PROVIDERS,
                infra,
            )

        fake_factory.assert_called_once_with(infra)
        assert result is fake_backend

    def test_resolve_falls_through_to_plugin(self):
        fake_backend = MagicMock(spec=TranscriptionBackend)
        fake_factory = MagicMock(return_value=fake_backend)
        fake_ep = MagicMock()
        fake_ep.name = 'my-whisper-plugin'
        fake_ep.load.return_value = fake_factory

        infra = InfraConfig(transcription_provider='my-whisper-plugin')
        with patch(
            'lazy_take_notes.l4_frameworks_and_drivers.container.entry_points',
            return_value=[fake_ep],
        ):
            result = DependencyContainer.resolve_transcription_backend(infra)

        assert result is fake_backend

    def test_container_uses_backend_factories(self, tmp_path: Path):
        """Container wires transcriber and model_resolver from the backend."""
        config = build_app_config({})
        template = YamlTemplateLoader().load('default_zh_tw')

        container = DependencyContainer(
            config,
            template,
            tmp_path,
            build_audio=False,
        )

        from lazy_take_notes.l3_interface_adapters.gateways.hf_model_resolver import HfModelResolver
        from lazy_take_notes.l3_interface_adapters.gateways.subprocess_whisper_transcriber import (
            SubprocessWhisperTranscriber,
        )

        assert isinstance(container.transcriber, SubprocessWhisperTranscriber)
        assert isinstance(container.model_resolver, HfModelResolver)
        assert isinstance(container._transcription_backend, TranscriptionBackend)
