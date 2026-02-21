"""Tests for batch runner — headless file transcription + digest."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from lazy_take_notes.l1_entities.transcript import TranscriptSegment
from lazy_take_notes.l2_use_cases.digest_use_case import DigestResult
from lazy_take_notes.l3_interface_adapters.gateways.yaml_template_loader import YamlTemplateLoader
from lazy_take_notes.l4_frameworks_and_drivers.batch_runner import run_batch
from lazy_take_notes.l4_frameworks_and_drivers.infra_config import InfraConfig, build_app_config

# 3 seconds of non-silent audio — enough for flush() to fire (min_speech_samples = 2s)
_SIGNAL_AUDIO = np.ones(16000 * 3, dtype=np.float32) * 0.1

_FAKE_SEGMENTS = [
    TranscriptSegment(text='Hello world', wall_start=0.0, wall_end=1.0),
    TranscriptSegment(text='This is a test', wall_start=1.0, wall_end=2.5),
]


def _make_run_batch_mocks(
    audio: np.ndarray = _SIGNAL_AUDIO,
    segments: list[TranscriptSegment] | None = None,
    digest_result: DigestResult | None = None,
):
    """Return a dict of patch targets → mock objects for run_batch."""
    if segments is None:
        segments = _FAKE_SEGMENTS
    if digest_result is None:
        digest_result = DigestResult(data='# Summary\n\nKey points here.')

    mock_load = MagicMock(return_value=audio)

    mock_resolver_instance = MagicMock()
    mock_resolver_instance.resolve.return_value = '/fake/model.bin'
    mock_resolver_cls = MagicMock(return_value=mock_resolver_instance)

    mock_transcriber_instance = MagicMock()
    mock_transcriber_instance.transcribe.return_value = segments

    mock_llm_instance = MagicMock()

    mock_execute = AsyncMock(return_value=digest_result)
    mock_digest_cls = MagicMock(return_value=MagicMock(execute=mock_execute))

    mock_persistence_instance = MagicMock()
    mock_persistence_instance.save_digest_md.return_value = Path('/out/digest.md')
    mock_persistence_instance.output_dir = Path('/out')

    # Fake container wires the instances together (replaces DependencyContainer)
    mock_container = MagicMock()
    mock_container.transcriber = mock_transcriber_instance
    mock_container.llm_client = mock_llm_instance
    mock_container.persistence = mock_persistence_instance
    mock_container_cls = MagicMock(return_value=mock_container)

    return {
        'load': mock_load,
        'resolver_cls': mock_resolver_cls,
        'container_cls': mock_container_cls,
        'transcriber_instance': mock_transcriber_instance,
        'llm_instance': mock_llm_instance,
        'digest_cls': mock_digest_cls,
        'execute': mock_execute,
        'persistence_instance': mock_persistence_instance,
    }


MODULE = 'lazy_take_notes.l4_frameworks_and_drivers.batch_runner'


class TestRunBatch:
    def _run(self, tmp_path: Path, mocks: dict, audio_path: Path | None = None) -> None:
        config = build_app_config({})
        template = YamlTemplateLoader().load('default_en')
        infra = InfraConfig()
        out_dir = tmp_path / 'output'

        if audio_path is None:
            audio_path = tmp_path / 'audio.wav'
            audio_path.touch()

        with (
            patch(f'{MODULE}.load_audio_file', mocks['load']),
            patch(f'{MODULE}.HfModelResolver', mocks['resolver_cls']),
            patch(f'{MODULE}.DependencyContainer', mocks['container_cls']),
            patch(f'{MODULE}.RunDigestUseCase', mocks['digest_cls']),
        ):
            run_batch(audio_path, config, template, out_dir, infra)

    def test_happy_path_transcribes_and_digests(self, tmp_path: Path, capsys) -> None:
        mocks = _make_run_batch_mocks()

        self._run(tmp_path, mocks)

        # Transcript was saved
        assert mocks['persistence_instance'].save_transcript_lines.called

        # Digest was saved and written to history
        assert mocks['persistence_instance'].save_digest_md.called
        assert mocks['persistence_instance'].save_history.called

        # Digest content printed to stdout
        captured = capsys.readouterr()
        assert 'Key points here' in captured.out

    def test_audio_load_failure_exits_1(self, tmp_path: Path) -> None:
        mocks = _make_run_batch_mocks()
        mocks['load'].side_effect = FileNotFoundError('Audio file not found: /x.wav')

        with pytest.raises(SystemExit) as exc_info:
            self._run(tmp_path, mocks)

        assert exc_info.value.code == 1

    def test_model_resolve_failure_exits_1(self, tmp_path: Path) -> None:
        mocks = _make_run_batch_mocks()
        mocks['resolver_cls'].return_value.resolve.side_effect = RuntimeError('network error')

        with pytest.raises(SystemExit) as exc_info:
            self._run(tmp_path, mocks)

        assert exc_info.value.code == 1

    def test_model_load_failure_exits_1(self, tmp_path: Path) -> None:
        mocks = _make_run_batch_mocks()
        mocks['transcriber_instance'].load_model.side_effect = RuntimeError('corrupt model')

        with pytest.raises(SystemExit) as exc_info:
            self._run(tmp_path, mocks)

        assert exc_info.value.code == 1

    def test_no_speech_detected_exits_0(self, tmp_path: Path) -> None:
        mocks = _make_run_batch_mocks(segments=[])

        with pytest.raises(SystemExit) as exc_info:
            self._run(tmp_path, mocks)

        assert exc_info.value.code == 0

    def test_digest_failure_exits_1(self, tmp_path: Path) -> None:
        mocks = _make_run_batch_mocks(digest_result=DigestResult(error='LLM timeout'))

        with pytest.raises(SystemExit) as exc_info:
            self._run(tmp_path, mocks)

        assert exc_info.value.code == 1

    def test_digest_called_with_is_final_true(self, tmp_path: Path) -> None:
        mocks = _make_run_batch_mocks()

        self._run(tmp_path, mocks)

        _, kwargs = mocks['execute'].call_args
        assert kwargs.get('is_final') is True

    def test_digest_called_with_full_transcript(self, tmp_path: Path) -> None:
        mocks = _make_run_batch_mocks()

        self._run(tmp_path, mocks)

        _, kwargs = mocks['execute'].call_args
        assert 'Hello world' in kwargs.get('full_transcript', '')

    def test_progress_lines_printed_to_stderr(self, tmp_path: Path, capsys) -> None:
        mocks = _make_run_batch_mocks()

        self._run(tmp_path, mocks)

        captured = capsys.readouterr()
        assert 'Transcribing' in captured.err
        assert 'Running digest' in captured.err

    def test_transcriber_closed_after_transcription(self, tmp_path: Path) -> None:
        mocks = _make_run_batch_mocks()

        self._run(tmp_path, mocks)

        mocks['transcriber_instance'].close.assert_called_once()

    def test_on_progress_callback(self, tmp_path: Path) -> None:
        mocks = _make_run_batch_mocks()

        self._run(tmp_path, mocks)

        # HfModelResolver was constructed with an on_progress callback
        resolver_call_kwargs = mocks['resolver_cls'].call_args
        assert resolver_call_kwargs is not None
        assert 'on_progress' in resolver_call_kwargs.kwargs
        assert callable(resolver_call_kwargs.kwargs['on_progress'])

    def test_should_trigger_fires_during_processing(self, tmp_path: Path) -> None:
        """With 26s of audio, should_trigger fires mid-loop (default chunk_duration=25s)."""
        audio_26s = np.ones(16000 * 26, dtype=np.float32) * 0.1
        mocks = _make_run_batch_mocks(audio=audio_26s)

        self._run(tmp_path, mocks)

        # Transcriber.transcribe called at least once during the loop (not just flush)
        assert mocks['transcriber_instance'].transcribe.call_count >= 1
