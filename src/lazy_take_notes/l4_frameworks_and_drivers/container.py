"""Dependency container — composition root for wiring all layers together."""

from __future__ import annotations

from pathlib import Path

from lazy_take_notes.l1_entities.config import AppConfig
from lazy_take_notes.l1_entities.template import SessionTemplate
from lazy_take_notes.l2_use_cases.ports.audio_source import AudioSource
from lazy_take_notes.l2_use_cases.ports.llm_client import LLMClient
from lazy_take_notes.l2_use_cases.ports.model_resolver import ModelResolver
from lazy_take_notes.l2_use_cases.ports.persistence import PersistenceGateway
from lazy_take_notes.l2_use_cases.ports.transcriber import Transcriber
from lazy_take_notes.l3_interface_adapters.controllers.session_controller import SessionController
from lazy_take_notes.l3_interface_adapters.gateways.file_persistence import FilePersistenceGateway
from lazy_take_notes.l3_interface_adapters.gateways.hf_model_resolver import HfModelResolver
from lazy_take_notes.l3_interface_adapters.gateways.ollama_llm_client import OllamaLLMClient
from lazy_take_notes.l3_interface_adapters.gateways.sounddevice_audio_source import SounddeviceAudioSource
from lazy_take_notes.l3_interface_adapters.gateways.whisper_transcriber import WhisperTranscriber
from lazy_take_notes.l3_interface_adapters.gateways.yaml_config_loader import YamlConfigLoader
from lazy_take_notes.l3_interface_adapters.gateways.yaml_template_loader import YamlTemplateLoader
from lazy_take_notes.l4_frameworks_and_drivers.infra_config import InfraConfig


class DependencyContainer:
    """Creates and wires all concrete instances. Easy to override for testing."""

    def __init__(
        self,
        config: AppConfig,
        template: SessionTemplate,
        output_dir: Path,
        infra: InfraConfig | None = None,
    ) -> None:
        self.config = config
        self.template = template
        self.output_dir = output_dir

        _infra = infra or InfraConfig()
        self.persistence: PersistenceGateway = FilePersistenceGateway(output_dir)
        self.llm_client: LLMClient = OllamaLLMClient(host=_infra.ollama.host)
        self.transcriber: Transcriber = WhisperTranscriber()
        self.audio_source: AudioSource = SounddeviceAudioSource()
        self.model_resolver: ModelResolver = HfModelResolver()

        self.controller = SessionController(
            config=config,
            template=template,
            llm_client=self.llm_client,
            persistence=self.persistence,
        )

    @staticmethod
    def config_loader() -> YamlConfigLoader:
        return YamlConfigLoader()

    @staticmethod
    def template_loader() -> YamlTemplateLoader:
        return YamlTemplateLoader()
