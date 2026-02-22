"""Tests for file transcription worker — thread body for audio file processing."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from lazy_take_notes.l1_entities.transcript import TranscriptSegment
from lazy_take_notes.l4_frameworks_and_drivers.messages import (
    AudioWorkerStatus,
    ModelDownloadProgress,
    TranscriptChunk,
)
from lazy_take_notes.l4_frameworks_and_drivers.workers.file_transcription_worker import run_file_transcription

# Patch at SOURCE module level — deferred imports inside run_file_transcription
# create local bindings, so patches must target the original modules.
_LOAD_AUDIO = 'lazy_take_notes.l3_interface_adapters.gateways.audio_file_loader.load_audio_file'
_HF_RESOLVER = 'lazy_take_notes.l3_interface_adapters.gateways.hf_model_resolver.HfModelResolver'

# 3 seconds of non-silent audio — enough for flush() to fire
_SIGNAL_AUDIO = np.ones(16000 * 3, dtype=np.float32) * 0.1

_FAKE_SEGMENTS = [
    TranscriptSegment(text='Hello world', wall_start=0.0, wall_end=1.0),
    TranscriptSegment(text='This is a test', wall_start=1.0, wall_end=2.5),
]


def _make_mocks(
    audio: np.ndarray = _SIGNAL_AUDIO,
    segments: list[TranscriptSegment] | None = None,
):
    if segments is None:
        segments = _FAKE_SEGMENTS

    mock_load = MagicMock(return_value=audio)
    mock_resolver_instance = MagicMock()
    mock_resolver_instance.resolve.return_value = '/fake/model.bin'
    mock_resolver_cls = MagicMock(return_value=mock_resolver_instance)

    mock_transcriber = MagicMock()
    mock_transcriber.transcribe.return_value = segments

    return mock_load, mock_resolver_cls, mock_transcriber


class TestRunFileTranscription:
    def _run(
        self,
        mock_load,
        mock_resolver_cls,
        mock_transcriber,
        is_cancelled=lambda: False,
    ):
        messages: list = []

        with (
            patch(_LOAD_AUDIO, mock_load),
            patch(_HF_RESOLVER, mock_resolver_cls),
        ):
            result = run_file_transcription(
                post_message=messages.append,
                is_cancelled=is_cancelled,
                audio_path=Path('/fake/audio.wav'),
                model_name='breeze-q5',
                language='zh',
                chunk_duration=25.0,
                overlap=1.0,
                silence_threshold=0.01,
                pause_duration=1.5,
                transcriber=mock_transcriber,
            )

        return result, messages

    def test_happy_path_posts_segments_and_stopped(self):
        mock_load, mock_resolver_cls, mock_transcriber = _make_mocks()
        result, messages = self._run(mock_load, mock_resolver_cls, mock_transcriber)

        # Transcriber was loaded
        mock_transcriber.load_model.assert_called_once_with('/fake/model.bin')

        # Status messages in order
        statuses = [m.status for m in messages if isinstance(m, AudioWorkerStatus)]
        assert 'loading_model' in statuses
        assert 'model_ready' in statuses
        assert 'recording' in statuses
        assert 'stopped' in statuses

        # Transcript chunks posted
        chunks = [m for m in messages if isinstance(m, TranscriptChunk)]
        assert len(chunks) > 0

        # Transcriber closed
        mock_transcriber.close.assert_called_once()

    def test_audio_load_failure_posts_error(self):
        mock_load, mock_resolver_cls, mock_transcriber = _make_mocks()
        mock_load.side_effect = FileNotFoundError('not found')

        result, messages = self._run(mock_load, mock_resolver_cls, mock_transcriber)

        assert result == []
        errors = [m for m in messages if isinstance(m, AudioWorkerStatus) and m.status == 'error']
        assert len(errors) == 1
        assert 'not found' in errors[0].error

    def test_model_resolve_failure_posts_error(self):
        mock_load, mock_resolver_cls, mock_transcriber = _make_mocks()
        mock_resolver_cls.return_value.resolve.side_effect = RuntimeError('network error')

        result, messages = self._run(mock_load, mock_resolver_cls, mock_transcriber)

        assert result == []
        errors = [m for m in messages if isinstance(m, AudioWorkerStatus) and m.status == 'error']
        assert len(errors) == 1

    def test_model_load_failure_posts_error(self):
        mock_load, mock_resolver_cls, mock_transcriber = _make_mocks()
        mock_transcriber.load_model.side_effect = RuntimeError('corrupt model')

        result, messages = self._run(mock_load, mock_resolver_cls, mock_transcriber)

        assert result == []
        errors = [m for m in messages if isinstance(m, AudioWorkerStatus) and m.status == 'error']
        assert len(errors) == 1
        mock_transcriber.close.assert_called_once()

    def test_cancellation_stops_early(self):
        mock_load, mock_resolver_cls, mock_transcriber = _make_mocks()

        call_count = 0

        def cancel_after_first():
            nonlocal call_count
            call_count += 1
            return call_count > 1  # cancel after first chunk

        result, messages = self._run(mock_load, mock_resolver_cls, mock_transcriber, is_cancelled=cancel_after_first)

        statuses = [m.status for m in messages if isinstance(m, AudioWorkerStatus)]
        assert 'stopped' in statuses
        mock_transcriber.close.assert_called_once()

    def test_no_speech_returns_empty_segments(self):
        mock_load, mock_resolver_cls, mock_transcriber = _make_mocks(segments=[])
        result, messages = self._run(mock_load, mock_resolver_cls, mock_transcriber)

        # No transcript chunks posted (empty segments from transcriber)
        chunks = [m for m in messages if isinstance(m, TranscriptChunk)]
        assert len(chunks) == 0
        assert result == []

    def test_download_progress_posted(self):
        mock_load, mock_resolver_cls, mock_transcriber = _make_mocks()

        # Make resolver call on_progress during resolve
        def _resolve_with_progress(name):
            mock_resolver_cls.call_args[1]['on_progress'](50)
            return '/fake/model.bin'

        mock_resolver_cls.return_value.resolve.side_effect = _resolve_with_progress

        result, messages = self._run(mock_load, mock_resolver_cls, mock_transcriber)

        progress = [m for m in messages if isinstance(m, ModelDownloadProgress)]
        assert len(progress) == 1
        assert progress[0].percent == 50

    def test_long_audio_triggers_mid_loop(self):
        """With 26s of audio, should_trigger fires mid-loop (default chunk_duration=25s)."""
        audio_26s = np.ones(16000 * 26, dtype=np.float32) * 0.1
        mock_load, mock_resolver_cls, mock_transcriber = _make_mocks(audio=audio_26s)

        result, messages = self._run(mock_load, mock_resolver_cls, mock_transcriber)

        # Transcriber.transcribe called at least once
        assert mock_transcriber.transcribe.call_count >= 1
