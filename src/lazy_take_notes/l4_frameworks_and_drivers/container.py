"""Dependency container — composition root for wiring all layers together."""

from __future__ import annotations

import sys
from pathlib import Path

from lazy_take_notes.l1_entities.config import AppConfig
from lazy_take_notes.l1_entities.template import SessionTemplate
from lazy_take_notes.l2_use_cases.ports.audio_source import AudioSource
from lazy_take_notes.l2_use_cases.ports.config_loader import ConfigLoader
from lazy_take_notes.l2_use_cases.ports.llm_client import LLMClient
from lazy_take_notes.l2_use_cases.ports.model_resolver import ModelResolver
from lazy_take_notes.l2_use_cases.ports.persistence import PersistenceGateway
from lazy_take_notes.l2_use_cases.ports.template_loader import TemplateLoader
from lazy_take_notes.l2_use_cases.ports.transcriber import Transcriber
from lazy_take_notes.l3_interface_adapters.controllers.session_controller import SessionController
from lazy_take_notes.l3_interface_adapters.gateways.file_persistence import FilePersistenceGateway
from lazy_take_notes.l3_interface_adapters.gateways.hf_model_resolver import HfModelResolver
from lazy_take_notes.l3_interface_adapters.gateways.subprocess_whisper_transcriber import SubprocessWhisperTranscriber
from lazy_take_notes.l3_interface_adapters.gateways.yaml_config_loader import YamlConfigLoader
from lazy_take_notes.l3_interface_adapters.gateways.yaml_template_loader import YamlTemplateLoader
from lazy_take_notes.l4_frameworks_and_drivers.config import InfraConfig


class DependencyContainer:
    """Creates and wires all concrete instances. Easy to override for testing."""

    def __init__(
        self,
        config: AppConfig,
        template: SessionTemplate,
        output_dir: Path,
        infra: InfraConfig | None = None,
        build_audio: bool = True,
        *,
        llm_client: LLMClient | None = None,
        transcriber: Transcriber | None = None,
        audio_source: AudioSource | None = None,
    ) -> None:
        self.config = config
        self.template = template
        self.output_dir = output_dir

        _infra = infra or InfraConfig()
        self.persistence: PersistenceGateway = FilePersistenceGateway(output_dir)
        self.llm_client: LLMClient = llm_client or self._build_llm_client(_infra)
        self.transcriber: Transcriber = transcriber or SubprocessWhisperTranscriber()
        self.audio_source: AudioSource | None = audio_source or (self._build_mixed_source() if build_audio else None)
        self.model_resolver: ModelResolver = HfModelResolver()

        self.controller = SessionController(
            config=config,
            template=template,
            llm_client=self.llm_client,
            persistence=self.persistence,
        )

    @staticmethod
    def _build_mixed_source() -> AudioSource:
        from lazy_take_notes.l3_interface_adapters.gateways.mixed_audio_source import (  # noqa: PLC0415 -- deferred: audio stack loaded only when needed
            MixedAudioSource,
        )
        from lazy_take_notes.l3_interface_adapters.gateways.sounddevice_audio_source import (  # noqa: PLC0415 -- deferred: sounddevice loaded only when audio is needed
            SounddeviceAudioSource,
        )

        if sys.platform == 'darwin':
            from lazy_take_notes.l3_interface_adapters.gateways.coreaudio_tap_source import (  # noqa: PLC0415 -- deferred: macOS only
                CoreAudioTapSource,
            )

            return MixedAudioSource(SounddeviceAudioSource(), CoreAudioTapSource())

        # Linux / Windows — use soundcard loopback
        from lazy_take_notes.l3_interface_adapters.gateways.soundcard_loopback_source import (  # noqa: PLC0415 -- deferred: non-macOS only
            SoundCardLoopbackSource,
        )

        return MixedAudioSource(SounddeviceAudioSource(), SoundCardLoopbackSource())

    @staticmethod
    def _build_llm_client(infra: InfraConfig) -> LLMClient:
        if infra.llm_provider == 'openai':
            from lazy_take_notes.l3_interface_adapters.gateways.openai_llm_client import (  # noqa: PLC0415 -- deferred: only loaded when provider is openai
                OpenAICompatLLMClient,
            )

            return OpenAICompatLLMClient(api_key=infra.openai.api_key, base_url=infra.openai.base_url)

        from lazy_take_notes.l3_interface_adapters.gateways.ollama_llm_client import (  # noqa: PLC0415 -- deferred: only loaded when provider is ollama
            OllamaLLMClient,
        )

        return OllamaLLMClient(host=infra.ollama.host)

    @staticmethod
    def config_loader() -> ConfigLoader:
        return YamlConfigLoader()

    @staticmethod
    def template_loader() -> TemplateLoader:
        return YamlTemplateLoader()
