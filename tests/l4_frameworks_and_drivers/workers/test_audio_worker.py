"""Tests for audio worker — uses FakeAudioSource instead of patching sounddevice."""

from __future__ import annotations

import threading
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from lazy_take_notes.l1_entities.audio_constants import SAMPLE_RATE
from lazy_take_notes.l1_entities.transcript import TranscriptSegment
from lazy_take_notes.l4_frameworks_and_drivers.messages import (
    AudioLevel,
    AudioWorkerStatus,
    TranscriptChunk,
)
from lazy_take_notes.l4_frameworks_and_drivers.workers.audio_worker import (
    _start_processed_recorder,  # noqa: PLC2701 -- testing private helper
    run_audio_worker,
)
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


# ---------------------------------------------------------------------------
# Step 2: _start_processed_recorder
# ---------------------------------------------------------------------------


class TestProcessedRecorder:
    def test_writes_wav_file(self, tmp_path: Path):
        """Push a float32 chunk + None sentinel, verify WAV exists and is valid."""
        rec_q, writer = _start_processed_recorder(tmp_path, SAMPLE_RATE)
        chunk = np.sin(np.linspace(0, 2 * np.pi, 1600, dtype=np.float32)) * 0.5
        rec_q.put(chunk)
        rec_q.put(None)
        writer.join(timeout=5)

        wav_path = tmp_path / 'recording.wav'
        assert wav_path.exists()
        with wave.open(str(wav_path), 'rb') as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == SAMPLE_RATE
            assert wf.getnframes() == 1600

    def test_clipping(self, tmp_path: Path):
        """Push data > 1.0, verify output clipped to [-1.0, 1.0]."""
        rec_q, writer = _start_processed_recorder(tmp_path, SAMPLE_RATE)
        loud = np.full(100, 2.0, dtype=np.float32)  # way above 1.0
        rec_q.put(loud)
        rec_q.put(None)
        writer.join(timeout=5)

        wav_path = tmp_path / 'recording.wav'
        with wave.open(str(wav_path), 'rb') as wf:
            raw = np.frombuffer(wf.readframes(100), dtype=np.int16)
        # int16 max for 1.0 clipped data is 32767
        assert np.all(raw == 32767)

    def test_sentinel_stops_writer(self, tmp_path: Path):
        """Push None immediately, verify WAV created with 0 frames."""
        rec_q, writer = _start_processed_recorder(tmp_path, SAMPLE_RATE)
        rec_q.put(None)
        writer.join(timeout=5)

        wav_path = tmp_path / 'recording.wav'
        assert wav_path.exists()
        with wave.open(str(wav_path), 'rb') as wf:
            assert wf.getnframes() == 0


# ---------------------------------------------------------------------------
# Step 3: run_audio_worker advanced paths
# ---------------------------------------------------------------------------


def _make_nonsilent_chunk(n_samples: int = 1600) -> np.ndarray:
    """Return a float32 chunk that passes the silence threshold."""
    return np.ones(n_samples, dtype=np.float32) * 0.5


class TestAudioWorkerTranscription:
    def test_transcription_triggered_and_segments_posted(self):
        """Feed enough non-silent audio to trigger transcription, verify segments posted."""
        segments = [TranscriptSegment(text='hello', wall_start=0.0, wall_end=1.0)]
        fake_transcriber = FakeTranscriber(segments=segments)

        # chunk_duration=0.1 → 1600 samples triggers at just 1 chunk
        chunk = _make_nonsilent_chunk(1600)
        fake_source = FakeAudioSource(chunks=[chunk, chunk])

        messages: list = []
        call_count = 0

        def is_cancelled():
            nonlocal call_count
            call_count += 1
            # Let it run a few iterations to process + drain
            return call_count > 4

        result = run_audio_worker(
            post_message=messages.append,
            is_cancelled=is_cancelled,
            model_path='test-model',
            language='zh',
            chunk_duration=0.1,
            overlap=0.0,
            silence_threshold=0.001,
            transcriber=fake_transcriber,
            audio_source=fake_source,
        )

        # Transcription was called at least once
        assert len(fake_transcriber.transcribe_calls) >= 1
        # TranscriptChunk posted with real segments
        transcript_msgs = [m for m in messages if isinstance(m, TranscriptChunk)]
        assert len(transcript_msgs) >= 1
        all_seg_texts = [s.text for tc in transcript_msgs for s in tc.segments]
        assert 'hello' in all_seg_texts
        # AudioLevel posted
        level_msgs = [m for m in messages if isinstance(m, AudioLevel)]
        assert len(level_msgs) >= 1
        assert level_msgs[0].rms > 0
        # Return value includes segments
        assert len(result) >= 1

    def test_flush_posts_remaining_segments(self):
        """Feed audio that triggers transcription, then cancel — flush should produce segments."""
        segments = [TranscriptSegment(text='flushed', wall_start=0.0, wall_end=0.5)]
        fake_transcriber = FakeTranscriber(segments=segments)

        # Feed enough to accumulate, but cancel before second trigger
        # Use larger chunk_duration so trigger fires via flush only
        chunk = _make_nonsilent_chunk(SAMPLE_RATE * 3)  # 3 seconds of audio
        fake_source = FakeAudioSource(chunks=[chunk])

        messages: list = []
        call_count = 0

        def is_cancelled():
            nonlocal call_count
            call_count += 1
            return call_count > 2

        run_audio_worker(
            post_message=messages.append,
            is_cancelled=is_cancelled,
            model_path='test-model',
            language='zh',
            chunk_duration=10.0,  # large: won't trigger during loop
            overlap=0.0,
            silence_threshold=0.001,
            transcriber=fake_transcriber,
            audio_source=fake_source,
        )

        # flush() calls process_buffer which calls transcribe
        assert len(fake_transcriber.transcribe_calls) >= 1
        transcript_msgs = [m for m in messages if isinstance(m, TranscriptChunk)]
        assert len(transcript_msgs) >= 1

    def test_processed_recorder_receives_chunks(self, tmp_path: Path):
        """save_audio=True with non-SounddeviceAudioSource writes recording.wav."""
        fake_transcriber = FakeTranscriber()
        chunk = _make_nonsilent_chunk(1600)
        fake_source = FakeAudioSource(chunks=[chunk])

        messages: list = []
        call_count = 0

        def is_cancelled():
            nonlocal call_count
            call_count += 1
            return call_count > 2

        run_audio_worker(
            post_message=messages.append,
            is_cancelled=is_cancelled,
            model_path='test-model',
            language='zh',
            save_audio=True,
            output_dir=tmp_path,
            transcriber=fake_transcriber,
            audio_source=fake_source,
        )

        wav_path = tmp_path / 'recording.wav'
        assert wav_path.exists()
        with wave.open(str(wav_path), 'rb') as wf:
            assert wf.getnframes() > 0

    def test_audio_source_open_error(self):
        """If audio_source.open raises, worker should post error status."""
        fake_transcriber = FakeTranscriber()
        fake_source = FakeAudioSource()
        fake_source.open = MagicMock(side_effect=RuntimeError('device not found'))  # type: ignore[method-assign]

        messages: list = []
        run_audio_worker(
            post_message=messages.append,
            is_cancelled=lambda: True,
            model_path='test-model',
            language='zh',
            transcriber=fake_transcriber,
            audio_source=fake_source,
        )

        statuses = [m for m in messages if isinstance(m, AudioWorkerStatus)]
        assert any(s.status == 'error' and 'device not found' in s.error for s in statuses)

    def test_in_flight_transcription_completes_at_shutdown(self):
        """If transcription is in-flight at shutdown, it completes and posts segments."""
        segments = [TranscriptSegment(text='in-flight', wall_start=0.0, wall_end=1.0)]
        fake_transcriber = FakeTranscriber(segments=segments)

        # Feed enough to trigger, then cancel on next iteration
        chunk = _make_nonsilent_chunk(1600)
        fake_source = FakeAudioSource(chunks=[chunk, chunk, chunk])

        messages: list = []
        call_count = 0

        def is_cancelled():
            nonlocal call_count
            call_count += 1
            # Let it trigger once then cancel
            return call_count > 3

        run_audio_worker(
            post_message=messages.append,
            is_cancelled=is_cancelled,
            model_path='test-model',
            language='zh',
            chunk_duration=0.1,
            overlap=0.0,
            silence_threshold=0.001,
            transcriber=fake_transcriber,
            audio_source=fake_source,
        )

        # Worker should have waited for in-flight transcription
        assert len(fake_transcriber.transcribe_calls) >= 1
        statuses = [m for m in messages if isinstance(m, AudioWorkerStatus)]
        assert any(s.status == 'stopped' for s in statuses)

    def test_transcription_error_posts_error_status(self):
        """If transcriber.transcribe raises, worker should post error status and continue."""
        fake_transcriber = FakeTranscriber()
        # After model load succeeds, make transcribe raise
        fake_transcriber.transcribe = MagicMock(  # type: ignore[method-assign]
            side_effect=RuntimeError('decode failed')
        )

        chunk = _make_nonsilent_chunk(1600)
        fake_source = FakeAudioSource(chunks=[chunk, chunk])

        messages: list = []
        call_count = 0

        def is_cancelled():
            nonlocal call_count
            call_count += 1
            return call_count > 4

        run_audio_worker(
            post_message=messages.append,
            is_cancelled=is_cancelled,
            model_path='test-model',
            language='zh',
            chunk_duration=0.1,
            overlap=0.0,
            silence_threshold=0.001,
            transcriber=fake_transcriber,
            audio_source=fake_source,
        )

        statuses = [m for m in messages if isinstance(m, AudioWorkerStatus)]
        # Should have error status from transcription failure
        assert any(s.status == 'error' and 'decode failed' in s.error for s in statuses)
        # But worker still completes normally
        assert any(s.status == 'stopped' for s in statuses)

    def test_drain_reads_remaining_and_records(self, tmp_path: Path):
        """Drain phase reads remaining chunks from source and feeds to recorder."""

        class DelayedAudioSource:
            """Source that yields extra chunks during drain phase."""

            def __init__(self):
                self._calls = 0
                self.open_calls = []
                self.close_calls = 0

            def open(self, sr, ch):
                self.open_calls.append((sr, ch))

            def read(self, timeout=0.1):
                self._calls += 1
                if self._calls <= 3:
                    return _make_nonsilent_chunk(800)
                return None

            def close(self):
                self.close_calls += 1

            def drain(self):
                return None

        fake_transcriber = FakeTranscriber()
        source = DelayedAudioSource()

        messages: list = []
        call_count = 0

        def is_cancelled():
            nonlocal call_count
            call_count += 1
            return call_count > 1  # Cancel after first loop iteration

        run_audio_worker(
            post_message=messages.append,
            is_cancelled=is_cancelled,
            model_path='test-model',
            language='zh',
            save_audio=True,
            output_dir=tmp_path,
            transcriber=fake_transcriber,
            audio_source=source,  # type: ignore[arg-type]  # DelayedAudioSource is a test-only duck type
        )

        # Recorder should have received at least the drain chunks
        wav_path = tmp_path / 'recording.wav'
        assert wav_path.exists()
        statuses = [m for m in messages if isinstance(m, AudioWorkerStatus)]
        assert any(s.status == 'stopped' for s in statuses)

    def test_periodic_stats_logged(self):
        """After 30s elapses, periodic stats line should execute."""
        fake_transcriber = FakeTranscriber()
        chunk = _make_nonsilent_chunk(1600)
        fake_source = FakeAudioSource(chunks=[chunk])

        messages: list = []
        call_count = 0

        def is_cancelled():
            nonlocal call_count
            call_count += 1
            return call_count > 1  # one loop iteration

        # First monotonic call → _worker_start=100.  All later calls → 131 (31s gap > 30s interval).
        _mono_calls = 0

        def fake_monotonic():
            nonlocal _mono_calls
            _mono_calls += 1
            return 100.0 if _mono_calls == 1 else 131.0

        with patch('time.monotonic', fake_monotonic):
            run_audio_worker(
                post_message=messages.append,
                is_cancelled=is_cancelled,
                model_path='test-model',
                language='zh',
                transcriber=fake_transcriber,
                audio_source=fake_source,
            )

        statuses = [m for m in messages if isinstance(m, AudioWorkerStatus)]
        assert any(s.status == 'stopped' for s in statuses)
