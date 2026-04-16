"""Dependency container — composition root for wiring all layers together."""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass
from importlib.metadata import entry_points
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

LLM_PROVIDERS_GROUP = 'lazy_take_notes.llm_providers'
BUILTIN_LLM_PROVIDERS = ('ollama', 'openai')

TRANSCRIPTION_PROVIDERS_GROUP = 'lazy_take_notes.transcription_providers'
BUILTIN_TRANSCRIPTION_PROVIDERS = ('whisper-cpp',)


@dataclass
class TranscriptionBackend:
    """Holds factories for creating a transcriber and model resolver.

    Plugin transcription providers return an instance of this from their
    factory callable so the container can wire up the correct backends.
    """

    create_transcriber: Callable[[], Transcriber]
    create_model_resolver: Callable[[Callable[[int], None] | None], ModelResolver]


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
        self.llm_client: LLMClient = llm_client or self.resolve_llm_client(_infra)

        backend = self.resolve_transcription_backend(_infra)
        self._transcription_backend = backend
        self.transcriber: Transcriber = transcriber or backend.create_transcriber()
        self.audio_source: AudioSource | None = audio_source or (
            self._build_mixed_source(config.transcription.silence_threshold) if build_audio else None
        )
        self.model_resolver: ModelResolver = backend.create_model_resolver(None)

        self.controller = SessionController(
            config=config,
            template=template,
            llm_client=self.llm_client,
            persistence=self.persistence,
        )

    @staticmethod
    def _build_mixed_source(silence_threshold: float = 0.01) -> AudioSource:
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

            return MixedAudioSource(
                SounddeviceAudioSource(),
                CoreAudioTapSource(),
                silence_threshold=silence_threshold,
            )

        # Linux / Windows — use soundcard loopback
        from lazy_take_notes.l3_interface_adapters.gateways.soundcard_loopback_source import (  # noqa: PLC0415 -- deferred: non-macOS only
            SoundCardLoopbackSource,
        )

        return MixedAudioSource(
            SounddeviceAudioSource(),
            SoundCardLoopbackSource(),
            silence_threshold=silence_threshold,
        )

    @staticmethod
    def resolve_llm_client(infra: InfraConfig) -> LLMClient:
        """Build an LLM client from built-in providers or plugin discovery."""
        if infra.llm_provider == 'openai':
            from lazy_take_notes.l3_interface_adapters.gateways.openai_llm_client import (  # noqa: PLC0415 -- deferred: only loaded when provider is openai
                OpenAICompatLLMClient,
            )

            return OpenAICompatLLMClient(api_key=infra.openai.api_key, base_url=infra.openai.base_url)

        if infra.llm_provider == 'ollama':
            from lazy_take_notes.l3_interface_adapters.gateways.ollama_llm_client import (  # noqa: PLC0415 -- deferred: only loaded when provider is ollama
                OllamaLLMClient,
            )

            return OllamaLLMClient(host=infra.ollama.host)

        return _load_plugin_provider(infra.llm_provider, LLM_PROVIDERS_GROUP, BUILTIN_LLM_PROVIDERS, infra)

    @staticmethod
    def resolve_transcription_backend(infra: InfraConfig) -> TranscriptionBackend:
        """Build a transcription backend from built-in providers or plugin discovery."""
        if infra.transcription_provider == 'whisper-cpp':
            return TranscriptionBackend(
                create_transcriber=SubprocessWhisperTranscriber,
                create_model_resolver=lambda on_progress: HfModelResolver(on_progress=on_progress),
            )
        return _load_plugin_provider(
            infra.transcription_provider,
            TRANSCRIPTION_PROVIDERS_GROUP,
            BUILTIN_TRANSCRIPTION_PROVIDERS,
            infra,
        )

    @property
    def model_resolver_factory(self) -> Callable[[Callable[[int], None] | None], ModelResolver]:
        """Public access to the transcription backend's model resolver factory."""
        return self._transcription_backend.create_model_resolver

    @staticmethod
    def config_loader() -> ConfigLoader:
        return YamlConfigLoader()

    @staticmethod
    def template_loader() -> TemplateLoader:
        return YamlTemplateLoader()


def _load_plugin_provider(
    provider_name: str,
    group: str,
    builtins: tuple[str, ...],
    infra: InfraConfig,
):
    """Discover and load a provider from a plugin entry point.

    Scans *group* for an entry point matching *provider_name*. The entry
    point must resolve to a callable ``(InfraConfig) -> T``.
    """
    eps = list(entry_points(group=group))
    for ep in eps:
        if ep.name == provider_name:
            factory = ep.load()
            return factory(infra)
    available = [*builtins, *(ep.name for ep in eps)]
    msg = f'Unknown provider {provider_name!r}. Available: {", ".join(available)}'
    raise ValueError(msg)
