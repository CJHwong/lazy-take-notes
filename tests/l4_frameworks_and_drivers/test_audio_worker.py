"""Tests for audio worker with mocked sounddevice and transcriber."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

from lazy_take_notes.l3_interface_adapters.presenters.messages import AudioWorkerStatus, TranscriptChunk
from lazy_take_notes.l4_frameworks_and_drivers.audio_worker import run_audio_worker
from tests.conftest import FakeTranscriber


class TestAudioWorker:
    @patch('lazy_take_notes.l4_frameworks_and_drivers.audio_worker.sd')
    def test_model_load_failure(self, mock_sd):
        """If model fails to load, worker should post error and return empty."""
        fake_transcriber = FakeTranscriber()
        fake_transcriber.load_model = MagicMock(  # type: ignore[invalid-assignment]
            side_effect=RuntimeError('model not found')
        )

        messages = []
        result = run_audio_worker(
            post_message=messages.append,
            is_cancelled=lambda: True,
            model_path='bad-model',
            language='zh',
            transcriber=fake_transcriber,
        )

        assert result == []
        statuses = [m for m in messages if isinstance(m, AudioWorkerStatus)]
        assert any(s.status == 'loading_model' for s in statuses)
        assert any(s.status == 'error' for s in statuses)

    @patch('lazy_take_notes.l4_frameworks_and_drivers.audio_worker.sd')
    def test_immediate_cancel(self, mock_sd):
        """If cancelled immediately, worker should load model and stop cleanly."""
        fake_transcriber = FakeTranscriber()

        mock_sd.InputStream = MagicMock()
        mock_sd.InputStream.return_value.__enter__ = MagicMock()
        mock_sd.InputStream.return_value.__exit__ = MagicMock(return_value=False)

        messages = []
        call_count = 0

        def is_cancelled():
            nonlocal call_count
            call_count += 1
            return call_count > 1

        run_audio_worker(
            post_message=messages.append,
            is_cancelled=is_cancelled,
            model_path='test-model',
            language='zh',
            transcriber=fake_transcriber,
        )

        statuses = [m for m in messages if isinstance(m, AudioWorkerStatus)]
        assert any(s.status == 'model_ready' for s in statuses)
        assert any(s.status == 'stopped' for s in statuses)

    @patch('lazy_take_notes.l4_frameworks_and_drivers.audio_worker.sd')
    def test_pause_discards_audio(self, mock_sd):
        """When pause_event is set, worker should drain queue but not transcribe."""
        fake_transcriber = FakeTranscriber()

        mock_sd.InputStream = MagicMock()
        mock_sd.InputStream.return_value.__enter__ = MagicMock()
        mock_sd.InputStream.return_value.__exit__ = MagicMock(return_value=False)

        messages = []
        pause_event = threading.Event()
        pause_event.set()

        call_count = 0

        def is_cancelled():
            nonlocal call_count
            call_count += 1
            return call_count > 3

        run_audio_worker(
            post_message=messages.append,
            is_cancelled=is_cancelled,
            model_path='test-model',
            language='zh',
            pause_event=pause_event,
            transcriber=fake_transcriber,
        )

        assert len(fake_transcriber.transcribe_calls) == 0
        chunks = [m for m in messages if isinstance(m, TranscriptChunk)]
        assert len(chunks) == 0
        statuses = [m for m in messages if isinstance(m, AudioWorkerStatus)]
        assert any(s.status == 'stopped' for s in statuses)
