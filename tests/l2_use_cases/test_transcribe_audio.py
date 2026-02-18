"""Tests for TranscribeAudioUseCase — uses FakeTranscriber, NOT @patch("pywhispercpp...")."""

import numpy as np

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

    def test_flush_with_too_little_data(self):
        fake = FakeTranscriber()
        uc = TranscribeAudioUseCase(transcriber=fake, language='zh', chunk_duration=10.0)

        # Feed less than min_speech_samples
        audio = np.random.randn(SAMPLE_RATE // 2).astype(np.float32) * 0.1
        uc.feed_audio(audio)

        segments = uc.flush()
        assert segments == []
