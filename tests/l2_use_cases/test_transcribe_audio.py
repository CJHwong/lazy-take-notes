"""Tests for TranscribeAudioUseCase — uses FakeTranscriber, NOT @patch("pywhispercpp...")."""

import numpy as np
import pytest

from lazy_take_notes.l1_entities.transcript import TranscriptSegment
from lazy_take_notes.l2_use_cases.transcribe_audio_use_case import SAMPLE_RATE, TranscribeAudioUseCase
from tests.conftest import FakeTranscriber


class TestTranscribeAudioUseCase:
    def test_feed_and_trigger_on_chunk_size(self):
        fake = FakeTranscriber(
            segments=[
                TranscriptSegment(text='Hello', wall_start=0.0, wall_end=2.5),
            ]
        )
        uc = TranscribeAudioUseCase(
            transcriber=fake,
            language='zh',
            chunk_duration=1.0,  # 16000 samples
        )

        # Feed enough audio to trigger
        audio = np.random.randn(SAMPLE_RATE).astype(np.float32) * 0.1
        uc.set_session_offset(1.0)
        uc.feed_audio(audio)

        assert uc.should_trigger()

    def test_no_trigger_below_chunk_size(self):
        fake = FakeTranscriber()
        uc = TranscribeAudioUseCase(transcriber=fake, language='zh', chunk_duration=1.0)

        audio = np.random.randn(SAMPLE_RATE // 2).astype(np.float32) * 0.1
        uc.feed_audio(audio)

        assert not uc.should_trigger()

    def test_process_buffer_returns_segments(self):
        fake = FakeTranscriber(
            segments=[
                TranscriptSegment(text='Hello', wall_start=0.0, wall_end=2.5),
                TranscriptSegment(text='World', wall_start=2.5, wall_end=5.0),
            ]
        )
        uc = TranscribeAudioUseCase(
            transcriber=fake,
            language='zh',
            chunk_duration=1.0,
            overlap=0.0,
        )

        audio = np.random.randn(SAMPLE_RATE).astype(np.float32) * 0.1
        uc.set_session_offset(1.0)
        uc.feed_audio(audio)

        segments = uc.process_buffer()
        assert len(segments) == 2
        assert segments[0].text == 'Hello'
        assert segments[1].text == 'World'

    def test_silence_skip(self):
        fake = FakeTranscriber()
        uc = TranscribeAudioUseCase(
            transcriber=fake,
            language='zh',
            chunk_duration=1.0,
            silence_threshold=0.01,
        )

        # Feed silence (all zeros)
        audio = np.zeros(SAMPLE_RATE, dtype=np.float32)
        uc.feed_audio(audio)

        segments = uc.process_buffer()
        assert segments == []
        # Transcriber should NOT have been called
        assert len(fake.transcribe_calls) == 0

    def test_reset_buffer(self):
        fake = FakeTranscriber()
        uc = TranscribeAudioUseCase(transcriber=fake, language='zh', chunk_duration=1.0)

        audio = np.random.randn(SAMPLE_RATE).astype(np.float32) * 0.1
        uc.feed_audio(audio)
        assert uc.should_trigger()

        uc.reset_buffer()
        assert not uc.should_trigger()

    def test_flush_with_enough_data(self):
        fake = FakeTranscriber(
            segments=[
                TranscriptSegment(text='Flush', wall_start=0.0, wall_end=2.0),
            ]
        )
        uc = TranscribeAudioUseCase(
            transcriber=fake,
            language='zh',
            chunk_duration=10.0,  # Very large, won't trigger normally
            overlap=0.0,
        )

        # Feed enough for min_speech_samples (2 seconds)
        audio = np.random.randn(SAMPLE_RATE * 3).astype(np.float32) * 0.1
        uc.set_session_offset(3.0)
        uc.feed_audio(audio)

        segments = uc.flush()
        assert len(segments) == 1
        assert segments[0].text == 'Flush'

    def test_second_chunk_not_dropped_when_segment_starts_before_overlap_threshold(self):
        """Regression: VAD leaves 1s silence in overlap buffer; next speech starts at ~0.95s.

        Old filter (wall_start >= overlap) would drop a segment whose start falls just
        below the 1.0s overlap threshold even though it contains entirely new content.
        Fix: filter on wall_end > overlap instead.
        """
        # First chunk: whisper returns one segment at 0.0
        chunk1_seg = TranscriptSegment(text='First speech', wall_start=0.0, wall_end=3.0)
        # Second chunk: speech starts slightly before 1.0s (centisecond rounding)
        chunk2_seg = TranscriptSegment(text='Second speech', wall_start=0.95, wall_end=4.5)

        call_count = 0

        class _TwoPassFake:
            def load_model(self, model_path: str) -> None:
                pass

            def transcribe(self, audio, language, initial_prompt=''):
                nonlocal call_count
                call_count += 1
                return [chunk1_seg] if call_count == 1 else [chunk2_seg]

            def close(self) -> None:
                pass

        uc = TranscribeAudioUseCase(
            transcriber=_TwoPassFake(),
            language='en',
            chunk_duration=1.0,
            overlap=1.0,
        )

        audio = np.random.randn(SAMPLE_RATE).astype(np.float32) * 0.1
        uc.set_session_offset(1.0)
        uc.feed_audio(audio)
        first = uc.process_buffer()
        assert [s.text for s in first] == ['First speech']

        # Feed second chunk; overlap buffer already holds 1s from previous call
        uc.feed_audio(audio)
        uc.set_session_offset(2.0)
        second = uc.process_buffer()
        assert [s.text for s in second] == ['Second speech'], (
            'Second speech was silently dropped — wall_start filter bug'
        )

    # --- prepare_buffer / apply_result ---

    def test_prepare_buffer_returns_snapshot_and_resets(self):
        """prepare_buffer() should snapshot the buffer, reset to overlap tail, mark not-first."""
        fake = FakeTranscriber()
        overlap_s = 1.0
        uc = TranscribeAudioUseCase(transcriber=fake, language='zh', chunk_duration=1.0, overlap=overlap_s)

        audio = np.random.randn(SAMPLE_RATE).astype(np.float32) * 0.1
        uc.set_session_offset(1.0)
        uc.feed_audio(audio)

        result = uc.prepare_buffer()

        assert result is not None
        snapshot, prompt, buf_wall_start, is_first = result
        assert len(snapshot) == SAMPLE_RATE
        assert is_first is True
        assert buf_wall_start == pytest.approx(0.0)
        # Buffer reset to overlap tail
        assert len(uc._buffer) == int(SAMPLE_RATE * overlap_s)
        assert uc._is_first_chunk is False

    def test_prepare_buffer_silent_returns_none(self):
        """Silent buffer should return None (nothing to transcribe)."""
        fake = FakeTranscriber()
        uc = TranscribeAudioUseCase(transcriber=fake, language='zh', chunk_duration=1.0)

        uc.feed_audio(np.zeros(SAMPLE_RATE, dtype=np.float32))

        result = uc.prepare_buffer()
        assert result is None
        assert len(fake.transcribe_calls) == 0

    def test_apply_result_filters_overlap_and_adjusts_timestamps(self):
        """apply_result() should drop overlap region and shift wall times."""
        fake = FakeTranscriber()
        uc = TranscribeAudioUseCase(transcriber=fake, language='zh', chunk_duration=1.0, overlap=1.0)

        segments = [
            TranscriptSegment(text='in overlap', wall_start=0.0, wall_end=0.5),  # dropped
            TranscriptSegment(text='kept', wall_start=0.9, wall_end=2.0),  # wall_end > 1.0
        ]
        result = uc.apply_result(segments, buffer_wall_start=5.0, is_first_chunk=False)

        assert len(result) == 1
        assert result[0].text == 'kept'
        assert result[0].wall_start == pytest.approx(5.0 + 0.9)
        assert result[0].wall_end == pytest.approx(5.0 + 2.0)

    def test_apply_result_first_chunk_keeps_all_segments(self):
        """First chunk has no overlap to filter."""
        fake = FakeTranscriber()
        uc = TranscribeAudioUseCase(transcriber=fake, language='zh', overlap=1.0)

        segments = [TranscriptSegment(text='early', wall_start=0.0, wall_end=0.5)]
        result = uc.apply_result(segments, buffer_wall_start=0.0, is_first_chunk=True)

        assert len(result) == 1
        assert result[0].text == 'early'

    def test_apply_result_updates_prompt_chain(self):
        """apply_result() should update _current_prompt with the last segment text."""
        fake = FakeTranscriber()
        uc = TranscribeAudioUseCase(transcriber=fake, language='zh', overlap=0.0)

        segments = [
            TranscriptSegment(text='first', wall_start=0.0, wall_end=1.0),
            TranscriptSegment(text='last', wall_start=1.0, wall_end=2.0),
        ]
        uc.apply_result(segments, buffer_wall_start=0.0, is_first_chunk=True)

        assert uc._current_prompt == 'last'

    def test_prepare_then_apply_equivalent_to_process_buffer(self):
        """prepare_buffer + apply_result should produce same output as process_buffer."""
        segs = [TranscriptSegment(text='speech', wall_start=0.0, wall_end=2.0)]
        fake_a = FakeTranscriber(segments=segs)
        fake_b = FakeTranscriber(segments=segs)

        uc_sync = TranscribeAudioUseCase(transcriber=fake_a, language='zh', chunk_duration=1.0, overlap=0.0)
        uc_async = TranscribeAudioUseCase(transcriber=fake_b, language='zh', chunk_duration=1.0, overlap=0.0)

        audio = np.random.randn(SAMPLE_RATE).astype(np.float32) * 0.1
        uc_sync.set_session_offset(1.0)
        uc_sync.feed_audio(audio)
        uc_async.set_session_offset(1.0)
        uc_async.feed_audio(audio)

        sync_result = uc_sync.process_buffer()

        prepared = uc_async.prepare_buffer()
        assert prepared is not None
        buf, prompt, buf_wall_start, is_first = prepared
        raw = fake_b.transcribe(buf, 'zh', prompt)
        async_result = uc_async.apply_result(raw, buf_wall_start, is_first)

        assert len(sync_result) == len(async_result)
        assert sync_result[0].text == async_result[0].text
        assert sync_result[0].wall_start == pytest.approx(async_result[0].wall_start)

    def test_flush_with_too_little_data(self):
        fake = FakeTranscriber()
        uc = TranscribeAudioUseCase(transcriber=fake, language='zh', chunk_duration=10.0)

        # Feed less than min_speech_samples
        audio = np.random.randn(SAMPLE_RATE // 2).astype(np.float32) * 0.1
        uc.feed_audio(audio)

        segments = uc.flush()
        assert segments == []
