"""Tests for audio worker — uses FakeAudioSource instead of patching sounddevice."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock

import numpy as np

from lazy_take_notes.l3_interface_adapters.presenters.messages import AudioWorkerStatus, TranscriptChunk
from lazy_take_notes.l4_frameworks_and_drivers.audio_worker import run_audio_worker
from tests.conftest import FakeAudioSource, FakeTranscriber


class TestAudioWorker:
    def test_model_load_failure(self):
        """If model fails to load, worker should post error and return empty."""
        fake_transcriber = FakeTranscriber()
        fake_transcriber.load_model = MagicMock(  # type: ignore[invalid-assignment]
            side_effect=RuntimeError('model not found')
        )
        fake_source = FakeAudioSource()

        messages = []
        result = run_audio_worker(
            post_message=messages.append,
            is_cancelled=lambda: True,
            model_path='bad-model',
            language='zh',
            transcriber=fake_transcriber,
            audio_source=fake_source,
        )

        assert result == []
        statuses = [m for m in messages if isinstance(m, AudioWorkerStatus)]
        assert any(s.status == 'loading_model' for s in statuses)
        assert any(s.status == 'error' for s in statuses)

    def test_immediate_cancel(self):
        """If cancelled immediately, worker should load model and stop cleanly."""
        fake_transcriber = FakeTranscriber()
        fake_source = FakeAudioSource()

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
            audio_source=fake_source,
        )

        statuses = [m for m in messages if isinstance(m, AudioWorkerStatus)]
        assert any(s.status == 'model_ready' for s in statuses)
        assert any(s.status == 'stopped' for s in statuses)
        assert fake_source.open_calls == [(16000, 1)]
        assert fake_source.close_calls == 1

    def test_pause_discards_audio(self):
        """When pause_event is set, worker should not feed audio to transcriber."""
        fake_transcriber = FakeTranscriber()
        fake_source = FakeAudioSource(chunks=[np.zeros(1600, dtype=np.float32)] * 3)

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
            audio_source=fake_source,
        )

        assert len(fake_transcriber.transcribe_calls) == 0
        chunks = [m for m in messages if isinstance(m, TranscriptChunk)]
        assert len(chunks) == 0
        statuses = [m for m in messages if isinstance(m, AudioWorkerStatus)]
        assert any(s.status == 'stopped' for s in statuses)

    def test_audio_chunks_fed_to_use_case(self):
        """Chunks from audio_source should be processed; audio_source opened and closed."""
        fake_transcriber = FakeTranscriber()
        chunks = [np.zeros(1600, dtype=np.float32)]
        fake_source = FakeAudioSource(chunks=chunks)

        messages = []
        cancelled_after = 2
        call_count = 0

        def is_cancelled():
            nonlocal call_count
            call_count += 1
            return call_count > cancelled_after

        run_audio_worker(
            post_message=messages.append,
            is_cancelled=is_cancelled,
            model_path='test-model',
            language='zh',
            transcriber=fake_transcriber,
            audio_source=fake_source,
        )

        assert fake_source.open_calls == [(16000, 1)]
        assert fake_source.close_calls == 1
        statuses = [m for m in messages if isinstance(m, AudioWorkerStatus)]
        assert any(s.status == 'stopped' for s in statuses)
