"""Tests for audio worker — uses FakeAudioSource instead of patching sounddevice."""

from __future__ import annotations

import threading
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from lazy_take_notes.l1_entities.audio_constants import SAMPLE_RATE
from lazy_take_notes.l1_entities.transcript import TranscriptSegment
from lazy_take_notes.l4_frameworks_and_drivers.messages import (
    AudioLevel,
    AudioWorkerStatus,
    TranscriptChunk,
    TranscriptionStatus,
)
from lazy_take_notes.l4_frameworks_and_drivers.workers.audio_worker import (
    _start_processed_recorder,  # noqa: PLC2701 -- testing private helper
    run_audio_worker,
)
from tests.conftest import FakeAudioSource, FakeTranscriber


class TestAudioWorker:
    def test_a_model_that_fails_to_load_still_records_the_audio(self, monkeypatch, tmp_path):
        """The audio is the record of the meeting; a missing transcript must not lose it."""
        fake_transcriber = FakeTranscriber()
        monkeypatch.setattr(fake_transcriber, 'load_model', MagicMock(side_effect=RuntimeError('model not found')))
        fake_source = FakeAudioSource(chunks=[_make_nonsilent_chunk(1600)])
        calls = 0

        def is_cancelled():
            nonlocal calls
            calls += 1
            return calls > 3

        messages = []
        result = run_audio_worker(
            post_message=messages.append,
            is_cancelled=is_cancelled,
            model_path='bad-model',
            language='zh',
            save_audio=True,
            output_dir=tmp_path,
            transcriber=fake_transcriber,
            audio_source=fake_source,
        )

        assert result == []
        statuses = [m for m in messages if isinstance(m, AudioWorkerStatus)]
        assert not any(s.status == 'error' for s in statuses)
        warning = next(s for s in statuses if s.status == 'warning')
        assert 'model not found' in warning.error
        assert 'Recording audio only' in warning.error
        assert any(s.status == 'recording' for s in statuses)
        assert fake_source.open_calls == [(16000, 1)]
        with wave.open(str(tmp_path / 'recording.wav'), 'rb') as wf:
            assert wf.getnframes() == 1600

    def test_the_stand_in_for_a_failed_model_transcribes_nothing(self):
        from lazy_take_notes.l4_frameworks_and_drivers.workers.audio_worker import (
            _NoTranscription,  # noqa: PLC2701 -- testing private helper
        )

        stand_in = _NoTranscription()
        stand_in.load_model('any')
        assert stand_in.transcribe(np.zeros(1600, dtype=np.float32), 'en') == []
        stand_in.close()

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

    def test_pause_before_any_audio_is_noop(self):
        """Pause set before any audio captured: flush finds empty buffer, no transcription."""
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

        # Empty buffer → nothing to flush → transcriber untouched.
        assert len(fake_transcriber.transcribe_calls) == 0
        chunks = [m for m in messages if isinstance(m, TranscriptChunk)]
        assert len(chunks) == 0
        statuses = [m for m in messages if isinstance(m, AudioWorkerStatus)]
        assert any(s.status == 'stopped' for s in statuses)
        # drain() should be called at least on pause entry.
        assert fake_source.drain_calls >= 1

    def test_pause_flushes_buffered_audio_then_discards(self):
        """Pause entry must commit buffered speech before discarding subsequent audio.

        Without this, anything buffered but not yet transcribed (up to chunk_duration)
        is lost whenever the user pauses mid-sentence.
        """
        fake_transcriber = FakeTranscriber()
        fake_transcriber.set_segments(
            [
                TranscriptSegment(text='hello world', wall_start=0.0, wall_end=0.2),
            ]
        )
        # Non-zero samples so RMS passes the silence_threshold gate in prepare_buffer.
        speech = np.full(1600, 0.3, dtype=np.float32)
        post_pause = np.full(1600, 0.5, dtype=np.float32)
        fake_source = FakeAudioSource(chunks=[speech, speech, post_pause, post_pause])

        messages = []
        pause_event = threading.Event()

        call_count = 0

        def is_cancelled():
            nonlocal call_count
            call_count += 1
            # Iter 1-2: unpaused — 2 speech chunks enter the buffer
            # Iter 3: flip pause → flush transcribes buffered speech; drain discards rest
            # Iter 4-5: still paused — no additional transcriber calls
            # Iter 6: cancel
            if call_count == 3:
                pause_event.set()
            return call_count > 5

        run_audio_worker(
            post_message=messages.append,
            is_cancelled=is_cancelled,
            model_path='test-model',
            language='zh',
            pause_event=pause_event,
            transcriber=fake_transcriber,
            audio_source=fake_source,
        )

        # Exactly one transcriber call — the flush of pre-pause speech.
        # Post-pause chunks were drained (or never read), not fed.
        assert len(fake_transcriber.transcribe_calls) == 1
        flush_audio = fake_transcriber.transcribe_calls[0][0]
        # Flushed buffer should span the 2 speech chunks (3200 samples).
        assert len(flush_audio) == 3200
        # All samples are from the pre-pause speech chunks (value 0.3).
        np.testing.assert_allclose(flush_audio, 0.3, atol=1e-6)
        # Transcript from the flush was posted.
        chunks = [m for m in messages if isinstance(m, TranscriptChunk)]
        assert len(chunks) == 1
        # drain() called on pause entry (and during pause).
        assert fake_source.drain_calls >= 1

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

    def test_transcription_posts_status_active_and_inactive(self):
        """TranscriptionStatus(active=True) posted before transcription, False after."""
        segments = [TranscriptSegment(text='hello', wall_start=0.0, wall_end=1.0)]
        fake_transcriber = FakeTranscriber(segments=segments)

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

        status_msgs = [m for m in messages if isinstance(m, TranscriptionStatus)]
        # At least one active=True and one active=False
        assert any(s.active for s in status_msgs), 'expected TranscriptionStatus(active=True)'
        assert any(not s.active for s in status_msgs), 'expected TranscriptionStatus(active=False)'
        # Every True must be followed by a False (paired)
        active_count = sum(1 for s in status_msgs if s.active)
        inactive_count = sum(1 for s in status_msgs if not s.active)
        assert inactive_count >= active_count

    def test_flush_posts_transcription_status(self):
        """Flush at shutdown brackets transcription with TranscriptionStatus messages."""
        segments = [TranscriptSegment(text='flushed', wall_start=0.0, wall_end=0.5)]
        fake_transcriber = FakeTranscriber(segments=segments)

        chunk = _make_nonsilent_chunk(SAMPLE_RATE * 3)
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
            chunk_duration=10.0,
            overlap=0.0,
            silence_threshold=0.001,
            transcriber=fake_transcriber,
            audio_source=fake_source,
        )

        status_msgs = [m for m in messages if isinstance(m, TranscriptionStatus)]
        # Flush path should produce at least one True/False pair
        assert any(s.active for s in status_msgs)
        assert any(not s.active for s in status_msgs)

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

    def test_audio_source_open_error(self, monkeypatch):
        """If audio_source.open raises, worker should post error status."""
        fake_transcriber = FakeTranscriber()
        fake_source = FakeAudioSource()
        monkeypatch.setattr(fake_source, 'open', MagicMock(side_effect=RuntimeError('device not found')))

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

    def test_transcription_error_posts_error_status(self, monkeypatch):
        """If transcriber.transcribe raises, worker should post error status and continue."""
        fake_transcriber = FakeTranscriber()
        # After model load succeeds, make transcribe raise
        monkeypatch.setattr(fake_transcriber, 'transcribe', MagicMock(side_effect=RuntimeError('decode failed')))

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

            mic_muted = False

            def __init__(self):
                self._calls = 0
                self.open_calls = []
                self.close_calls = 0

            def open(self, sample_rate, channels):
                self.open_calls.append((sample_rate, channels))

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
            audio_source=source,
        )

        # Recorder should have received at least the drain chunks
        wav_path = tmp_path / 'recording.wav'
        assert wav_path.exists()
        statuses = [m for m in messages if isinstance(m, AudioWorkerStatus)]
        assert any(s.status == 'stopped' for s in statuses)

    def test_exhausted_source_breaks_loop_and_posts_error(self):
        """When audio source signals exhaustion, worker should break, post error, and stop."""

        class ExhaustibleSource:
            """Audio source that signals exhaustion after producing all chunks."""

            mic_muted = False

            def __init__(self, chunks):
                self._chunks = list(chunks)
                self._idx = 0
                self._exhausted = False
                self.open_calls = []
                self.close_calls = 0

            @property
            def exhausted(self):
                return self._exhausted

            def open(self, sample_rate, channels):
                self.open_calls.append((sample_rate, channels))

            def read(self, timeout=0.1):
                if self._idx >= len(self._chunks):
                    self._exhausted = True
                    return None
                chunk = self._chunks[self._idx]
                self._idx += 1
                return chunk

            def drain(self):
                return None

            def close(self):
                self.close_calls += 1

        fake_transcriber = FakeTranscriber()
        source = ExhaustibleSource(chunks=[_make_nonsilent_chunk(1600)])

        messages: list = []
        run_audio_worker(
            post_message=messages.append,
            is_cancelled=lambda: False,  # never cancels — exhaustion should break the loop
            model_path='test-model',
            language='zh',
            transcriber=fake_transcriber,
            audio_source=source,
        )

        statuses = [m for m in messages if isinstance(m, AudioWorkerStatus)]
        assert any(s.status == 'error' and 'unexpectedly' in s.error for s in statuses)
        assert any(s.status == 'stopped' for s in statuses)
        assert source.close_calls == 1

    def test_nan_audio_posts_zero_level(self):
        """NaN audio data should result in AudioLevel(rms=0.0), not crash or propagate NaN."""
        fake_transcriber = FakeTranscriber()
        nan_chunk = np.full(1600, float('nan'), dtype=np.float32)
        fake_source = FakeAudioSource(chunks=[nan_chunk])

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
            transcriber=fake_transcriber,
            audio_source=fake_source,
        )

        level_msgs = [m for m in messages if isinstance(m, AudioLevel)]
        assert len(level_msgs) >= 1
        assert level_msgs[0].rms == 0.0

    def test_prolonged_no_data_logs_warning(self):
        """When read() returns None for 5+ seconds, worker should log a warning."""
        fake_transcriber = FakeTranscriber()
        # One real chunk, then source returns None forever
        fake_source = FakeAudioSource(chunks=[_make_nonsilent_chunk(1600)])

        messages: list = []
        _mono_calls = 0

        def fake_monotonic():
            nonlocal _mono_calls
            _mono_calls += 1
            # First 3 calls: _worker_start, _last_data_time (chunk), now_abs (chunk)
            # After that: jump to 106s so gap = 6.0 triggers the warning branch
            return 100.0 if _mono_calls <= 3 else 106.0

        call_count = 0

        def is_cancelled():
            nonlocal call_count
            call_count += 1
            return call_count > 5

        with patch('time.monotonic', fake_monotonic):
            run_audio_worker(
                post_message=messages.append,
                is_cancelled=is_cancelled,
                model_path='test-model',
                language='zh',
                transcriber=fake_transcriber,
                audio_source=fake_source,
            )

        # Worker should complete cleanly (the warning is logged, not posted)
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


class _SourceWithoutSignal(FakeAudioSource):
    system_signal_seen = False


class _SourceWithSignal(FakeAudioSource):
    system_signal_seen = True


class _SourceAwaitingPermission(FakeAudioSource):
    system_signal_seen = False
    awaiting_permission = True


class TestNoSystemSignalWarning:
    """A missing capture grant makes the tap deliver zeros with no error."""

    @staticmethod
    def _run_with_elapsed(fake_source, messages: list, iterations: int = 3) -> None:
        loop_calls = 0

        def is_cancelled():
            nonlocal loop_calls
            loop_calls += 1
            return loop_calls > iterations

        mono_calls = 0

        def fake_monotonic():
            nonlocal mono_calls
            mono_calls += 1
            return 100.0 if mono_calls == 1 else 120.0

        with patch('time.monotonic', fake_monotonic):
            run_audio_worker(
                post_message=messages.append,
                is_cancelled=is_cancelled,
                model_path='test-model',
                language='zh',
                transcriber=FakeTranscriber(),
                audio_source=fake_source,
            )

    def test_warns_once_when_system_source_never_had_signal(self):
        silent = np.zeros(1600, dtype=np.float32)
        fake_source = _SourceWithoutSignal(chunks=[silent, silent, silent])
        messages: list = []

        self._run_with_elapsed(fake_source, messages)

        warnings = [m for m in messages if isinstance(m, AudioWorkerStatus) and m.status == 'warning']
        assert len(warnings) == 1
        assert 'System Audio Recording Only' in warnings[0].error

    def test_no_warning_when_system_source_has_signal(self):
        chunk = _make_nonsilent_chunk(1600)
        fake_source = _SourceWithSignal(chunks=[chunk, chunk, chunk])
        messages: list = []

        self._run_with_elapsed(fake_source, messages)

        warnings = [m for m in messages if isinstance(m, AudioWorkerStatus) and m.status == 'warning']
        assert warnings == []

    def test_no_warning_while_the_permission_dialog_is_open(self):
        # A first-time user spends more than 15 s in System Settings. Telling
        # them to grant what the open dialog already asks is noise.
        silent = np.zeros(1600, dtype=np.float32)
        fake_source = _SourceAwaitingPermission(chunks=[silent, silent, silent])
        messages: list = []

        self._run_with_elapsed(fake_source, messages)

        warnings = [m for m in messages if isinstance(m, AudioWorkerStatus) and m.status == 'warning']
        assert warnings == []

    def test_no_warning_for_source_that_cannot_report_signal(self):
        # Linux loopback and the test fake have no system_signal_seen attribute.
        chunk = _make_nonsilent_chunk(1600)
        fake_source = FakeAudioSource(chunks=[chunk, chunk, chunk])
        messages: list = []

        self._run_with_elapsed(fake_source, messages)

        warnings = [m for m in messages if isinstance(m, AudioWorkerStatus) and m.status == 'warning']
        assert warnings == []


class TestStreamingWavWriter:
    def test_header_stays_valid_while_writing(self, tmp_path):
        import struct
        import wave as wave_module

        from lazy_take_notes.l4_frameworks_and_drivers.workers.audio_worker import (
            _StreamingWavWriter,  # noqa: PLC2701 -- testing private helper
        )

        path = tmp_path / 'recording.wav'
        writer = _StreamingWavWriter(path, 16000)
        writer.write(b'\x01\x00' * 160)

        # A reader must see a valid header mid-session, not zeroed sizes.
        with wave_module.open(str(path)) as reader:
            assert reader.getnframes() == 160
            assert reader.getframerate() == 16000

        writer.write(b'\x02\x00' * 160)
        with wave_module.open(str(path)) as reader:
            assert reader.getnframes() == 320

        raw = path.read_bytes()
        assert struct.unpack('<I', raw[40:44])[0] == 640
        writer.close()


class TestSourceFailureWarnings:
    def test_reports_lost_source_while_healthy_audio_continues(self):
        class DegradedSource(FakeAudioSource):
            source_errors = ('System audio stopped: test helper exited',)

        source = DegradedSource(chunks=[np.zeros(1600, dtype=np.float32)] * 3)
        messages = []
        remaining_reads = iter([False, False, False, True])

        run_audio_worker(
            post_message=messages.append,
            is_cancelled=lambda: next(remaining_reads, True),
            model_path='test-model',
            language='en',
            transcriber=FakeTranscriber(),
            audio_source=source,
        )

        warnings = [
            message.error
            for message in messages
            if isinstance(message, AudioWorkerStatus) and message.status == 'warning'
        ]
        assert warnings == ['System audio stopped: test helper exited']
        assert any(isinstance(message, AudioLevel) for message in messages)
        assert source.close_calls == 1

    def test_reports_source_failure_without_audio_frames(self):
        class FailedSource(FakeAudioSource):
            source_errors = ('Microphone stopped: test input disconnected',)

        messages = []
        remaining_reads = iter([False, False, True])
        run_audio_worker(
            post_message=messages.append,
            is_cancelled=lambda: next(remaining_reads, True),
            model_path='test-model',
            language='en',
            transcriber=FakeTranscriber(),
            audio_source=FailedSource(),
        )

        warnings = [
            message.error
            for message in messages
            if isinstance(message, AudioWorkerStatus) and message.status == 'warning'
        ]
        assert warnings == ['Microphone stopped: test input disconnected']

    def test_reports_failure_again_after_source_recovers(self):
        class IntermittentSource(FakeAudioSource):
            source_errors: tuple[str, ...] = ()

            def read(self, timeout=0.1):
                chunk = super().read(timeout)
                self.source_errors = () if self._idx == 2 else ('System audio stopped',)
                return chunk

        messages = []
        remaining_reads = iter([False, False, False, True])
        run_audio_worker(
            post_message=messages.append,
            is_cancelled=lambda: next(remaining_reads, True),
            model_path='test-model',
            language='en',
            transcriber=FakeTranscriber(),
            audio_source=IntermittentSource(chunks=[np.zeros(1600, dtype=np.float32)] * 3),
        )
        warnings = [
            message.error
            for message in messages
            if isinstance(message, AudioWorkerStatus) and message.status == 'warning'
        ]
        assert warnings == ['System audio stopped', 'System audio stopped']


class TestProcessedRecorderFaults:
    """A failing disk pauses the WAV; the audio waits in memory and the recording resumes."""

    def _run(self, output_dir, chunks, errors, between=None):
        from lazy_take_notes.l4_frameworks_and_drivers.workers import audio_worker as module

        rec_q, writer = module._start_processed_recorder(output_dir, 16000, on_error=errors.append)  # noqa: SLF001 -- private helper under test
        for index, chunk in enumerate(chunks):
            if between is not None:
                between(index)
            rec_q.put(chunk)
        rec_q.put(None)
        writer.join(timeout=5)
        assert not writer.is_alive()
        assert rec_q.empty()

    @staticmethod
    def _frames(path):
        with wave.open(str(path), 'rb') as wf:
            return wf.readframes(wf.getnframes())

    def test_a_full_disk_keeps_the_audio_and_writes_it_to_the_next_file(self, tmp_path, monkeypatch):
        from lazy_take_notes.l4_frameworks_and_drivers.workers import audio_worker as module

        real_write = module._StreamingWavWriter.write  # noqa: SLF001 -- fault injection
        calls = {'count': 0}

        def write(self, pcm):
            calls['count'] += 1
            if calls['count'] in (3, 4):  # the third chunk fails, then its first retry fails too
                raise OSError(28, 'No space left on device')
            real_write(self, pcm)

        monkeypatch.setattr(module._StreamingWavWriter, 'write', write)  # noqa: SLF001 -- fault injection
        monkeypatch.setattr(module, 'WAV_RETRY_S', 0.0)
        chunks = [np.full(160, value / 10, dtype=np.float32) for value in range(6)]

        errors: list[str] = []
        self._run(tmp_path, chunks, errors)

        assert sorted(path.name for path in tmp_path.glob('*.wav')) == ['recording.wav', 'recording_2.wav']
        first, second = self._frames(tmp_path / 'recording.wav'), self._frames(tmp_path / 'recording_2.wav')
        expected = b''.join((np.clip(c, -1, 1) * 32767).astype(np.int16).tobytes() for c in chunks)
        assert first + second == expected  # every chunk once, in order
        assert len(errors) == 2
        assert 'No space left' in errors[0]
        assert 'recording_2.wav' in errors[1]

    @pytest.mark.parametrize('remove', ['delete', 'move_to_trash'])
    def test_a_removed_session_folder_is_created_again(self, tmp_path, monkeypatch, remove):
        import shutil

        from lazy_take_notes.l4_frameworks_and_drivers.workers import audio_worker as module

        monkeypatch.setattr(module, 'WAV_RETRY_S', 0.0)
        session = tmp_path / 'session'
        session.mkdir()
        errors: list[str] = []
        recording = module._RecoveringWav(session, 16000, errors.append)  # noqa: SLF001 -- private helper under test
        recording.write(np.full(160, 0.1, dtype=np.float32))
        if remove == 'delete':
            shutil.rmtree(session)
        else:
            session.rename(tmp_path / 'Trash')
        recording.write(np.full(160, 0.2, dtype=np.float32))  # lands in the removed file, so it is kept
        recording.write(np.full(160, 0.3, dtype=np.float32))
        recording.close()

        frames = np.frombuffer(self._frames(session / 'recording.wav'), dtype=np.int16)
        assert len(frames) == 320
        assert frames[0] == int(0.2 * 32767)
        assert 'removed' in errors[0]
        assert 'recording.wav' in errors[1]

    def test_a_folder_renamed_by_the_app_keeps_one_recording(self, tmp_path):
        from lazy_take_notes.l4_frameworks_and_drivers.workers import audio_worker as module

        folder = module.SessionFolder(tmp_path / '2026-02-21_143052')
        folder.path.mkdir()
        errors: list[str] = []
        recording = module._RecoveringWav(folder, 16000, errors.append)  # noqa: SLF001 -- private helper under test
        recording.write(np.full(160, 0.1, dtype=np.float32))
        folder.rename(tmp_path / '2026-02-21_143052_standup')
        recording.write(np.full(160, 0.2, dtype=np.float32))
        recording.close()

        assert not (tmp_path / '2026-02-21_143052').exists()
        assert list(folder.path.iterdir()) == [folder.path / 'recording.wav']
        frames = np.frombuffer(self._frames(folder.path / 'recording.wav'), dtype=np.int16)
        assert len(frames) == 320
        assert errors == []

    def test_a_rename_during_a_stalled_write_gives_up_instead_of_freezing(self, tmp_path, monkeypatch):
        from lazy_take_notes.l4_frameworks_and_drivers.workers import session_folder

        monkeypatch.setattr(session_folder, 'RENAME_WAIT_S', 0.05)
        folder = session_folder.SessionFolder(tmp_path / 'session')
        folder.path.mkdir()
        with folder.lock, pytest.raises(OSError, match='busy'):  # the writer is inside a slow write
            folder.rename(tmp_path / 'session_label')
        assert folder.path == tmp_path / 'session'

    def test_the_writer_takes_the_lock_once_per_chunk(self, tmp_path):
        # Holding it for the whole backlog (up to 10 min of audio) would block a rename for that long.
        import threading

        from lazy_take_notes.l4_frameworks_and_drivers.workers import audio_worker as module

        class CountingLock:
            def __init__(self) -> None:
                self.inner = threading.Lock()
                self.entries = 0

            def __enter__(self):
                self.entries += 1
                return self.inner.__enter__()

            def __exit__(self, *exc):
                return self.inner.__exit__(*exc)

        folder = module.SessionFolder(tmp_path / 'session')
        folder.path.mkdir()
        lock = CountingLock()
        folder.lock = lock  # ty: ignore[invalid-assignment] -- a counting stand-in for threading.Lock
        recording = module._RecoveringWav(folder, 16000, None)  # noqa: SLF001 -- private helper under test
        recording._hold(b'\0' * 320)  # noqa: SLF001 -- a backlog of three chunks
        recording._hold(b'\0' * 320)  # noqa: SLF001
        recording.write(np.zeros(160, dtype=np.float32))

        assert lock.entries == 4  # three chunks, then one pass that finds the backlog empty

    def test_save_audio_off_deletes_the_file_in_the_renamed_folder(self, tmp_path):
        from lazy_take_notes.l4_frameworks_and_drivers.workers import audio_worker as module

        folder = module.SessionFolder(tmp_path / 'session')
        folder.path.mkdir()
        recording = module._RecoveringWav(folder, 16000, None, keep_files=lambda: False)  # noqa: SLF001 -- private helper under test
        recording.write(np.full(160, 0.1, dtype=np.float32))
        folder.rename(tmp_path / 'session_label')
        recording.close()

        assert list(folder.path.iterdir()) == []

    def test_a_disk_that_never_recovers_reports_once_and_logs_the_lost_audio(self, tmp_path, monkeypatch, caplog):
        from lazy_take_notes.l4_frameworks_and_drivers.workers import audio_worker as module

        def _full_disk(self, pcm):
            raise OSError(28, 'No space left on device')

        monkeypatch.setattr(module._StreamingWavWriter, 'write', _full_disk)  # noqa: SLF001 -- fault injection
        monkeypatch.setattr(module, 'WAV_RETRY_S', 0.0)
        errors: list[str] = []
        with caplog.at_level('ERROR', logger='ltn.audio'):
            self._run(tmp_path, [np.zeros(16000, dtype=np.float32)] * 3, errors)

        assert len(errors) == 1
        assert 'No space left' in errors[0]
        assert '3 s of audio was not saved' in caplog.text

    def test_the_audio_held_in_memory_is_bounded(self, tmp_path, monkeypatch, caplog):
        from lazy_take_notes.l4_frameworks_and_drivers.workers import audio_worker as module

        def _full_disk(self, pcm):
            raise OSError(28, 'No space left on device')

        monkeypatch.setattr(module._StreamingWavWriter, 'write', _full_disk)  # noqa: SLF001 -- fault injection
        monkeypatch.setattr(module, 'WAV_BACKLOG_S', 2.0)
        recording = module._RecoveringWav(tmp_path, 16000, None)  # noqa: SLF001 -- private helper under test
        with caplog.at_level('ERROR', logger='ltn.audio'):
            for _ in range(5):
                recording.write(np.zeros(16000, dtype=np.float32))

        assert recording._backlog_bytes <= 2 * 16000 * 2  # noqa: SLF001 -- the bound under test
        assert caplog.text.count('dropping the oldest audio') == 1

    def test_a_wav_that_fails_to_close_after_a_failed_write_keeps_draining(self, tmp_path, monkeypatch):
        from lazy_take_notes.l4_frameworks_and_drivers.workers import audio_worker as module

        def _full_disk(self, pcm):
            raise OSError(28, 'No space left on device')

        def _bad_close(self):
            raise OSError(5, 'Input/output error')

        monkeypatch.setattr(module._StreamingWavWriter, 'write', _full_disk)  # noqa: SLF001 -- fault injection
        monkeypatch.setattr(module._StreamingWavWriter, 'close', _bad_close)  # noqa: SLF001 -- fault injection
        errors: list[str] = []
        self._run(tmp_path, [np.zeros(160, dtype=np.float32)] * 5, errors)
        assert len(errors) == 1


class TestAudioHealth:
    """A source that reports health gets one message per change, not one toast per retry."""

    def _run(self, source, reads):
        from lazy_take_notes.l4_frameworks_and_drivers.messages import AudioHealth

        messages = []
        remaining = iter([False] * reads + [True])
        run_audio_worker(
            post_message=messages.append,
            is_cancelled=lambda: next(remaining, True),
            model_path='test-model',
            language='en',
            transcriber=FakeTranscriber(),
            audio_source=source,
        )
        return [m for m in messages if isinstance(m, AudioHealth)], messages

    def test_posts_each_health_change_once(self):
        states = iter(
            [
                {'mic': 'ok', 'system': 'ok'},
                {'mic': 'lost', 'system': 'ok'},
                {'mic': 'lost', 'system': 'ok'},
                {'mic': 'lost', 'system': 'ok'},
                {'mic': 'ok', 'system': 'ok'},
            ]
        )

        class HealthSource(FakeAudioSource):
            health = {'mic': 'ok', 'system': 'ok'}
            source_errors: tuple[str, ...] = ()

            def read(self, timeout=0.1):
                self.health = next(states, self.health)
                self.source_errors = ('Microphone: unplugged',) if self.health['mic'] == 'lost' else ()
                return super().read(timeout)

        health, messages = self._run(HealthSource(chunks=[np.zeros(1600, dtype=np.float32)] * 5), reads=5)

        assert [(h.mic, h.system) for h in health] == [('ok', 'ok'), ('lost', 'ok'), ('ok', 'ok')]
        assert 'unplugged' in health[1].detail
        warnings = [m for m in messages if isinstance(m, AudioWorkerStatus) and m.status == 'warning']
        assert warnings == []  # the health message replaces the per-error toast

    def test_a_device_fallback_notice_is_posted(self):
        class FallbackSource(FakeAudioSource):
            health = {'mic': 'ok', 'system': 'ok'}
            mic_notice = "Microphone 'USB' is unavailable; using Built-in"

        health, _ = self._run(FallbackSource(chunks=[np.zeros(1600, dtype=np.float32)]), reads=1)
        assert health[0].notice == "Microphone 'USB' is unavailable; using Built-in"


class TestLoopErrorsDoNotEndRecording:
    def test_an_unexpected_error_in_one_iteration_keeps_the_recording_going(self):
        class GlitchSource(FakeAudioSource):
            def read(self, timeout=0.1):
                if self._idx == 1 and not getattr(self, '_glitched', False):
                    self._glitched = True
                    raise ValueError('bad chunk shape')
                return super().read(timeout)

        source = GlitchSource(chunks=[np.full(1600, 0.1, dtype=np.float32)] * 4)
        messages = []
        remaining = iter([False] * 7 + [True])
        run_audio_worker(
            post_message=messages.append,
            is_cancelled=lambda: next(remaining, True),
            model_path='test-model',
            language='en',
            transcriber=FakeTranscriber(),
            audio_source=source,
        )

        statuses = [m for m in messages if isinstance(m, AudioWorkerStatus)]
        assert not any(m.status == 'error' for m in statuses)
        assert any(m.status == 'warning' and 'bad chunk shape' in m.error for m in statuses)
        assert source._idx == 4  # every chunk after the glitch was still read
        assert statuses[-1].status == 'stopped'


class TestRepeatedLoopErrors:
    def test_a_loop_that_keeps_failing_raises_a_persistent_alert_and_clears_it(self):
        from lazy_take_notes.l4_frameworks_and_drivers.messages import AudioHealth

        class BrokenThenFine(FakeAudioSource):
            failures = 30

            def read(self, timeout=0.1):
                if self.failures:
                    self.failures -= 1
                    raise ValueError('bad chunk')
                return super().read(timeout)

        source = BrokenThenFine(chunks=[np.full(1600, 0.1, dtype=np.float32)] * 2)
        messages = []
        remaining = iter([False] * 33 + [True])
        with patch('lazy_take_notes.l4_frameworks_and_drivers.workers.audio_worker._LOOP_ALERT_AFTER_S', 0.2):
            run_audio_worker(
                post_message=messages.append,
                is_cancelled=lambda: next(remaining, True),
                model_path='test-model',
                language='en',
                transcriber=FakeTranscriber(),
                audio_source=source,
            )

        processing = [m.processing for m in messages if isinstance(m, AudioHealth)]
        assert processing == ['error', 'ok']


class TestAudioWithSaveAudioOff:
    """save_audio: false still writes the audio; a clean session deletes it, a failure keeps it."""

    def _record(self, tmp_path, *, save_audio=False, transcriber=None, seen=None):
        chunks = [_make_nonsilent_chunk(16000) for _ in range(3)]
        calls = 0

        def is_cancelled():
            nonlocal calls
            calls += 1
            if seen is not None and calls == 4:
                seen.append((tmp_path / 'recording.wav').exists())
            return calls > 5

        messages: list = []
        run_audio_worker(
            post_message=messages.append,
            is_cancelled=is_cancelled,
            model_path='test-model',
            language='en',
            chunk_duration=1.0,
            save_audio=save_audio,
            output_dir=tmp_path,
            transcriber=transcriber or FakeTranscriber(),
            audio_source=FakeAudioSource(chunks=chunks),
        )
        return [m.error for m in messages if isinstance(m, AudioWorkerStatus) and m.status == 'warning']

    def test_a_clean_session_writes_the_audio_then_deletes_it(self, tmp_path):
        seen: list[bool] = []
        warnings = self._record(tmp_path, seen=seen)
        assert seen == [True]  # written while the meeting runs
        assert not list(tmp_path.glob('*.wav'))
        assert warnings == []

    def test_a_model_that_fails_to_load_keeps_the_audio_and_says_why(self, tmp_path, monkeypatch):
        transcriber = FakeTranscriber()
        monkeypatch.setattr(transcriber, 'load_model', MagicMock(side_effect=RuntimeError('model not found')))
        warnings = self._record(tmp_path, transcriber=transcriber)
        assert (tmp_path / 'recording.wav').exists()
        notices = [w for w in warnings if 'save_audio is off' in w]
        assert len(notices) == 1
        assert 'speech model failed to load' in notices[0]

    def test_a_failed_transcription_keeps_the_audio(self, tmp_path):
        class FailingTranscriber(FakeTranscriber):
            def transcribe(self, audio, language, hints=None):
                raise RuntimeError('whisper crashed')

        warnings = self._record(tmp_path, transcriber=FailingTranscriber())
        assert (tmp_path / 'recording.wav').exists()
        assert any('a transcription failed' in w for w in warnings)

    def test_save_audio_on_keeps_the_audio_without_a_notice(self, tmp_path):
        warnings = self._record(tmp_path, save_audio=True)
        assert (tmp_path / 'recording.wav').exists()
        assert not any('save_audio is off' in w for w in warnings)

    def test_disk_notices_stay_in_the_log_until_the_audio_is_kept(self):
        from lazy_take_notes.l4_frameworks_and_drivers.workers.audio_worker import (
            _UnsavedAudio,  # noqa: PLC2701 -- testing private helper
        )

        posted: list = []
        unsaved = _UnsavedAudio(save_audio=False, post_message=posted.append)
        unsaved.notice('Audio recording paused: disk full')
        assert posted == []
        unsaved.keep_because('a transcription failed')
        unsaved.keep_because('a transcription failed')
        unsaved.notice('Audio recording continues in recording_2.wav.')
        assert [m.error for m in posted][1:] == ['Audio recording continues in recording_2.wav.']
        assert len(posted) == 2

    def test_a_file_that_cannot_be_deleted_is_logged(self, tmp_path, monkeypatch, caplog):
        from lazy_take_notes.l4_frameworks_and_drivers.workers import audio_worker as module

        def _locked(self, missing_ok=False):
            raise PermissionError(13, 'Permission denied')

        recording = module._RecoveringWav(tmp_path, 16000, None, keep_files=lambda: False)  # noqa: SLF001 -- private helper under test
        recording.write(np.zeros(160, dtype=np.float32))
        monkeypatch.setattr(module.Path, 'unlink', _locked)
        with caplog.at_level('WARNING', logger='ltn.audio'):
            recording.close()
        assert 'Could not delete' in caplog.text

    def test_a_transcription_that_fails_at_pause_keeps_the_audio(self, tmp_path, caplog):
        class FailingTranscriber(FakeTranscriber):
            def transcribe(self, audio, language, hints=None):
                raise RuntimeError('whisper crashed')

        speech = np.full(16000, 0.3, dtype=np.float32)
        pause_event = threading.Event()
        calls = 0

        def is_cancelled():
            nonlocal calls
            calls += 1
            if calls == 2:  # one chunk in flight, more speech in the buffer
                pause_event.set()
            return calls > 4

        messages: list = []
        with caplog.at_level('ERROR', logger='ltn.audio'):
            run_audio_worker(
                post_message=messages.append,
                is_cancelled=is_cancelled,
                model_path='test-model',
                language='en',
                chunk_duration=0.5,
                pause_event=pause_event,
                output_dir=tmp_path,
                transcriber=FailingTranscriber(),
                audio_source=FakeAudioSource(chunks=[speech, speech]),
            )

        assert (tmp_path / 'recording.wav').exists()
        assert 'Flushing in-flight transcription on pause failed' in caplog.text
        assert 'Pause flush transcription failed' in caplog.text
