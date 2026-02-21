"""Shared test fixtures and protocol-conforming fakes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lazy_take_notes.l1_entities.chat_message import ChatMessage
from lazy_take_notes.l1_entities.config import AppConfig
from lazy_take_notes.l1_entities.template import SessionTemplate
from lazy_take_notes.l1_entities.transcript import TranscriptSegment
from lazy_take_notes.l2_use_cases.ports.llm_client import ChatResponse
from lazy_take_notes.l3_interface_adapters.gateways.yaml_template_loader import (
    YamlTemplateLoader,
    builtin_names,
)
from lazy_take_notes.l4_frameworks_and_drivers.infra_config import build_app_config

# --- Protocol-conforming Fakes ---


class FakeLLMClient:
    """Fake LLM client for L2 use case tests."""

    def __init__(self, response: str = 'Fake LLM response', prompt_tokens: int = 100):
        self._response = response
        self._prompt_tokens = prompt_tokens
        self.chat_calls: list[tuple[str, list[ChatMessage]]] = []
        self.chat_single_calls: list[tuple[str, str]] = []
        self._connectivity = (True, '')
        self._missing_models: list[str] = []

    async def chat(self, model: str, messages: list[ChatMessage]) -> ChatResponse:
        self.chat_calls.append((model, list(messages)))
        return ChatResponse(content=self._response, prompt_tokens=self._prompt_tokens)

    async def chat_single(self, model: str, prompt: str) -> str:
        self.chat_single_calls.append((model, prompt))
        return self._response

    def check_connectivity(self) -> tuple[bool, str]:
        return self._connectivity

    def set_response(self, response: str, prompt_tokens: int = 100) -> None:
        self._response = response
        self._prompt_tokens = prompt_tokens

    def check_models(self, models: list[str]) -> list[str]:
        return [m for m in models if m in self._missing_models]

    def set_connectivity(self, ok: bool, msg: str = '') -> None:
        self._connectivity = (ok, msg)

    def set_missing_models(self, models: list[str]) -> None:
        self._missing_models = list(models)


class FakeTranscriber:
    """Fake transcriber for L2 use case tests."""

    def __init__(self, segments: list[TranscriptSegment] | None = None):
        self._segments = segments or []
        self.load_model_calls: list[str] = []
        self.transcribe_calls: list[tuple[np.ndarray, str, list[str] | None]] = []

    def load_model(self, model_path: str) -> None:
        self.load_model_calls.append(model_path)

    def transcribe(
        self,
        audio: np.ndarray,
        language: str,
        hints: list[str] | None = None,
    ) -> list[TranscriptSegment]:
        self.transcribe_calls.append((audio, language, hints))
        return self._segments

    def close(self) -> None:
        pass

    def set_segments(self, segments: list[TranscriptSegment]) -> None:
        self._segments = segments


class FakeAudioSource:
    """Fake audio source for L4 worker tests — implements AudioSource protocol."""

    def __init__(self, chunks: list[np.ndarray] | None = None) -> None:
        self._chunks = list(chunks or [])
        self.open_calls: list[tuple[int, int]] = []
        self.close_calls: int = 0
        self._idx = 0

    def open(self, sample_rate: int, channels: int) -> None:
        self.open_calls.append((sample_rate, channels))

    def read(self, timeout: float = 0.1) -> np.ndarray | None:
        if self._idx >= len(self._chunks):
            return None
        chunk = self._chunks[self._idx]
        self._idx += 1
        return chunk

    def close(self) -> None:
        self.close_calls += 1

    def drain(self) -> np.ndarray | None:
        return None


class FakePersistence:
    """Fake persistence gateway for L2/L3 tests."""

    def __init__(self, output_dir: Path | None = None):
        self._output_dir = output_dir or Path('/fake/output')
        self.transcript_calls: list[tuple[list[TranscriptSegment], bool]] = []
        self.digest_calls: list[tuple[str, int]] = []
        self.history_calls: list[tuple[str, int, bool]] = []
        self.context_calls: list[str] = []

    def save_transcript_lines(self, segments: list[TranscriptSegment], *, append: bool = True) -> Path:
        self.transcript_calls.append((segments, append))
        return self._output_dir / 'transcript_raw.txt'

    def save_digest_md(self, markdown: str, digest_number: int) -> Path:
        self.digest_calls.append((markdown, digest_number))
        return self._output_dir / 'digest.md'

    def save_history(self, markdown: str, digest_number: int, *, is_final: bool = False) -> Path:
        self.history_calls.append((markdown, digest_number, is_final))
        return self._output_dir / 'history' / f'digest_{digest_number:03d}.md'

    def save_session_context(self, context: str) -> Path:
        self.context_calls.append(context)
        return self._output_dir / 'session_context.txt'


# --- Standard Fixtures ---


@pytest.fixture
def tmp_output_dir(tmp_path: Path) -> Path:
    d = tmp_path / 'output'
    d.mkdir()
    return d


@pytest.fixture
def default_config() -> AppConfig:
    return build_app_config({})


@pytest.fixture
def default_template() -> SessionTemplate:
    return YamlTemplateLoader().load('default_zh_tw')


@pytest.fixture(params=sorted(builtin_names()))
def any_builtin_template(request: pytest.FixtureRequest) -> SessionTemplate:
    return YamlTemplateLoader().load(request.param)


@pytest.fixture
def sample_config_yaml(tmp_path: Path) -> Path:
    content = """\
transcription:
  model: "breeze-q5"
  chunk_duration: 8.0
  overlap: 1.0
  silence_threshold: 0.01
  pause_duration: 1.5
digest:
  model: "llama3:8b"
  min_lines: 10
  min_interval: 30
  compact_token_threshold: 100000
interactive:
  model: "llama3:8b"
template: "default_zh_tw"
output:
  directory: "./test_output"
  save_audio: true
"""
    p = tmp_path / 'config.yaml'
    p.write_text(content, encoding='utf-8')
    return p


@pytest.fixture
def fake_llm() -> FakeLLMClient:
    return FakeLLMClient()


@pytest.fixture
def fake_transcriber() -> FakeTranscriber:
    return FakeTranscriber()


@pytest.fixture
def fake_persistence(tmp_output_dir: Path) -> FakePersistence:
    return FakePersistence(tmp_output_dir)
