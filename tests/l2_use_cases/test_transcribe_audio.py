"""Tests for TranscribeAudioUseCase — uses FakeTranscriber, NOT @patch("pywhispercpp...")."""

import numpy as np
import pytest

from lazy_take_notes.l1_entities.transcript import TranscriptSegment
from lazy_take_notes.l2_use_cases import transcribe_audio_use_case
from lazy_take_notes.l2_use_cases.transcribe_audio_use_case import SAMPLE_RATE, TranscribeAudioUseCase, voiced_seconds
from tests.conftest import FakeTranscriber, level_only_voiced_seconds, speech_like


@pytest.fixture(autouse=True)
def _gate(request, monkeypatch):
    """These tests feed 1 s chunks to test buffering; TestSpeechGate sets the real gate."""
    if not getattr(request.cls, 'real_gate', False):
        monkeypatch.setattr(transcribe_audio_use_case, 'MIN_VOICED_S', 0.0)
        monkeypatch.setattr(transcribe_audio_use_case, 'voiced_seconds', level_only_voiced_seconds)


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
        audio = speech_like(SAMPLE_RATE)
        uc.set_session_offset(1.0)
        uc.feed_audio(audio)

        assert uc.should_trigger()

    def test_no_trigger_below_chunk_size(self):
        fake = FakeTranscriber()
        uc = TranscribeAudioUseCase(transcriber=fake, language='zh', chunk_duration=1.0)

        audio = speech_like(SAMPLE_RATE // 2)
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
            chunk_duration=2.0,  # the buffer ends at a pause, not at the size limit
            overlap=0.0,
        )

        audio = speech_like(SAMPLE_RATE)
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

    def test_feed_audio_grows_storage_beyond_initial_capacity(self):
        """Internal storage starts at 10s and must grow on demand without losing
        samples. Regression guard for the pre-allocated buffer that replaced
        np.concatenate-per-feed (which was O(n) per call and caused live level
        meter lag under whisper backpressure on long meetings)."""
        fake = FakeTranscriber()
        uc = TranscribeAudioUseCase(transcriber=fake, language='en', chunk_duration=1000.0)

        # Feed 25 seconds in 32-ms chunks — exceeds the 10s pre-allocation,
        # triggering at least one storage grow.
        chunk = np.full(512, 0.05, dtype=np.float32)
        n_chunks = 25 * SAMPLE_RATE // 512
        for _ in range(n_chunks):
            uc.feed_audio(chunk)

        assert len(uc._buffer) == n_chunks * 512  # noqa: SLF001
        # All samples preserved at the expected value.
        np.testing.assert_allclose(uc._buffer, 0.05, atol=1e-6)  # noqa: SLF001

    def test_feed_audio_after_reset_to_tail_continues_correctly(self):
        """After prepare_buffer resets the buffer to the overlap tail, the next
        feed_audio must extend that tail correctly — exercises the _replace_buffer
        path that takes a copy when the new contents alias internal storage."""
        fake = FakeTranscriber(segments=[TranscriptSegment(text='x', wall_start=0, wall_end=1)])
        uc = TranscribeAudioUseCase(transcriber=fake, language='en', chunk_duration=2.0, overlap=0.5)

        # Feed 2 seconds at amplitude 0.5 so the buffer triggers and prepare_buffer
        # returns a non-silent snapshot.
        loud = np.full(SAMPLE_RATE * 2, 0.5, dtype=np.float32)
        uc.feed_audio(loud)
        prepared = uc.prepare_buffer()
        assert prepared is not None

        # Buffer is now reset to the last 0.5s = 8000 samples of value 0.5.
        assert len(uc._buffer) == int(SAMPLE_RATE * 0.5)  # noqa: SLF001
        np.testing.assert_allclose(uc._buffer, 0.5, atol=1e-6)  # noqa: SLF001

        # Feed a fresh chunk; the tail must remain intact and the new samples appended.
        fresh = np.full(SAMPLE_RATE, 0.7, dtype=np.float32)
        uc.feed_audio(fresh)
        assert len(uc._buffer) == int(SAMPLE_RATE * 0.5) + SAMPLE_RATE  # noqa: SLF001
        np.testing.assert_allclose(uc._buffer[: int(SAMPLE_RATE * 0.5)], 0.5, atol=1e-6)  # noqa: SLF001
        np.testing.assert_allclose(uc._buffer[int(SAMPLE_RATE * 0.5) :], 0.7, atol=1e-6)  # noqa: SLF001

    def test_reset_buffer(self):
        fake = FakeTranscriber()
        uc = TranscribeAudioUseCase(transcriber=fake, language='zh', chunk_duration=1.0)

        audio = speech_like(SAMPLE_RATE)
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
        audio = speech_like(SAMPLE_RATE * 3)
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

            def transcribe(self, audio, language, hints=None):
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

        audio = speech_like(SAMPLE_RATE)
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

        audio = speech_like(SAMPLE_RATE)
        uc.set_session_offset(1.0)
        uc.feed_audio(audio)

        result = uc.prepare_buffer()

        assert result is not None
        snapshot, hints, buf_wall_start, is_first = result
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

    def test_apply_result_updates_hint_chain(self):
        """apply_result() should update _current_hints with the last segment text."""
        fake = FakeTranscriber()
        uc = TranscribeAudioUseCase(transcriber=fake, language='zh', overlap=0.0)

        segments = [
            TranscriptSegment(text='first', wall_start=0.0, wall_end=1.0),
            TranscriptSegment(text='last', wall_start=1.0, wall_end=2.0),
        ]
        uc.apply_result(segments, buffer_wall_start=0.0, is_first_chunk=True)

        assert uc._current_hints == ['last']

    def test_prepare_then_apply_equivalent_to_process_buffer(self):
        """prepare_buffer + apply_result should produce same output as process_buffer."""
        segs = [TranscriptSegment(text='speech', wall_start=0.0, wall_end=2.0)]
        fake_a = FakeTranscriber(segments=segs)
        fake_b = FakeTranscriber(segments=segs)

        uc_sync = TranscribeAudioUseCase(transcriber=fake_a, language='zh', chunk_duration=1.0, overlap=0.0)
        uc_async = TranscribeAudioUseCase(transcriber=fake_b, language='zh', chunk_duration=1.0, overlap=0.0)

        audio = speech_like(SAMPLE_RATE)
        uc_sync.set_session_offset(1.0)
        uc_sync.feed_audio(audio)
        uc_async.set_session_offset(1.0)
        uc_async.feed_audio(audio)

        sync_result = uc_sync.process_buffer()

        prepared = uc_async.prepare_buffer()
        assert prepared is not None
        buf, hints, buf_wall_start, is_first = prepared
        raw = fake_b.transcribe(buf, 'zh', hints)
        async_result = uc_async.apply_result(raw, buf_wall_start, is_first)

        assert len(sync_result) == len(async_result)
        assert sync_result[0].text == async_result[0].text
        assert sync_result[0].wall_start == pytest.approx(async_result[0].wall_start)

    def test_vad_triggers_on_speech_then_silence(self):
        """Buffer with speech body + silent tail should trigger via VAD pause detection."""
        fake = FakeTranscriber(segments=[TranscriptSegment(text='Speech', wall_start=0.0, wall_end=2.0)])
        uc = TranscribeAudioUseCase(
            transcriber=fake,
            language='zh',
            chunk_duration=100.0,  # very large — won't trigger on size
            overlap=0.0,
            silence_threshold=0.01,
            pause_duration=1.5,
        )

        # Feed 3s of speech (min_speech = 2s) + 1.5s of silence (pause_duration)
        speech = speech_like(SAMPLE_RATE * 3)
        silence = np.zeros(int(SAMPLE_RATE * 1.5), dtype=np.float32)
        uc.feed_audio(np.concatenate([speech, silence]))
        uc.set_session_offset(4.5)

        assert uc.should_trigger()

    def test_flush_silent_buffer_returns_empty(self):
        """Flush with a silent buffer should return empty — not waste CPU on transcription."""
        fake = FakeTranscriber()
        uc = TranscribeAudioUseCase(
            transcriber=fake,
            language='zh',
            chunk_duration=100.0,
            overlap=0.0,
            silence_threshold=0.01,
        )

        uc.feed_audio(np.zeros(SAMPLE_RATE * 3, dtype=np.float32))
        segments = uc.flush()
        assert segments == []
        assert len(fake.transcribe_calls) == 0

    def test_flush_with_too_little_data(self):
        fake = FakeTranscriber()
        uc = TranscribeAudioUseCase(transcriber=fake, language='zh', chunk_duration=10.0)

        # Feed less than min_speech_samples
        audio = speech_like(SAMPLE_RATE // 2)
        uc.feed_audio(audio)

        segments = uc.flush()
        assert segments == []


def _use_case(segments: list[TranscriptSegment]) -> tuple[TranscribeAudioUseCase, FakeTranscriber]:
    fake = FakeTranscriber(segments=segments)
    return TranscribeAudioUseCase(transcriber=fake, language='en', chunk_duration=25.0, overlap=0.0), fake


def _seg(text: str, start: float = 0.0) -> TranscriptSegment:
    return TranscriptSegment(text=text, wall_start=start, wall_end=start + 1.0)


class TestSpeechGate:
    real_gate = True

    def test_steady_noise_is_not_speech(self):
        noise = np.random.default_rng(1).standard_normal(SAMPLE_RATE * 10).astype(np.float32) * 0.02
        assert voiced_seconds(noise, 0.01) == 0.0

    def test_speech_over_a_quiet_floor_counts(self):
        assert voiced_seconds(speech_like(SAMPLE_RATE * 5), 0.01) == pytest.approx(4.0, abs=0.2)

    def test_a_click_in_silence_is_not_sent(self):
        audio = np.zeros(SAMPLE_RATE * 10, dtype=np.float32)
        audio[SAMPLE_RATE * 5 : SAMPLE_RATE * 5 + 4000] = 0.5  # 0.25 s click
        uc, fake = _use_case([_seg('Thank you.')])
        uc.feed_audio(audio)
        assert uc.prepare_buffer() is None
        assert uc.process_buffer() == []
        assert fake.transcribe_calls == []

    def test_a_short_sentence_in_a_quiet_room_is_sent(self):
        audio = np.zeros(SAMPLE_RATE * 20, dtype=np.float32)
        audio[SAMPLE_RATE * 5 : SAMPLE_RATE * 7] = speech_like(SAMPLE_RATE * 2, amplitude=0.2)
        uc, fake = _use_case([_seg('Yes, ship it on Thursday.')])
        uc.feed_audio(audio)
        assert [segment.text for segment in uc.process_buffer()] == ['Yes, ship it on Thursday.']

    def test_speech_after_a_long_quiet_lead_is_sent(self):
        # 20 s of quiet pulls the whole-chunk average under the threshold; the speech is still there.
        audio = np.zeros(SAMPLE_RATE * 25, dtype=np.float32)
        audio[SAMPLE_RATE * 20 :] = speech_like(SAMPLE_RATE * 5, amplitude=0.02)
        assert np.sqrt(np.mean(audio**2)) < 0.01
        uc, fake = _use_case([_seg('Good morning everyone.', 20.0)])
        uc.feed_audio(audio)
        assert uc.prepare_buffer() is not None

    def test_flush_sends_speech_after_a_quiet_lead(self):
        audio = np.zeros(SAMPLE_RATE * 20, dtype=np.float32)
        audio[SAMPLE_RATE * 17 :] = speech_like(SAMPLE_RATE * 3, amplitude=0.02)
        assert np.sqrt(np.mean(audio**2)) < 0.01
        uc, _ = _use_case([_seg('Last words.', 17.0)])
        uc.feed_audio(audio)
        assert [s.text for s in uc.flush()] == ['Last words.']


class TestNotSpeechLines:
    @pytest.mark.parametrize('text', ['*clap*', '*sad music*', '[BLANK_AUDIO]', '(laughs)', '♪', '.', '...', ' - '])
    def test_an_annotation_or_punctuation_line_is_dropped(self, text):
        uc, _ = _use_case([_seg(text), _seg('Real words.', 1.0)])
        uc.feed_audio(speech_like(SAMPLE_RATE * 3))
        assert [segment.text for segment in uc.process_buffer()] == ['Real words.']

    def test_a_dropped_line_is_not_chained_into_the_next_prompt(self):
        uc, fake = _use_case([_seg('Real words.'), _seg('*clap*', 1.0)])
        uc.feed_audio(speech_like(SAMPLE_RATE * 3))
        uc.process_buffer()
        uc.feed_audio(speech_like(SAMPLE_RATE * 3))
        uc.process_buffer()
        assert fake.transcribe_calls[1][2] == ['Real words.']


class TestDecodingLoops:
    def test_a_sentence_repeated_three_times_collapses(self):
        loop = 'And you can write the search plan by next Friday. ' * 51
        uc, _ = _use_case([_seg(f'We agreed. {loop}')])
        uc.feed_audio(speech_like(SAMPLE_RATE * 3))
        assert [s.text for s in uc.process_buffer()] == ['We agreed. And you can write the search plan by next Friday.']

    def test_an_unpunctuated_loop_collapses(self):
        loop = 'and you can write the search plan by next friday ' * 20
        uc, _ = _use_case([_seg(f'we agreed {loop}')])
        uc.feed_audio(speech_like(SAMPLE_RATE * 3))
        assert [s.text for s in uc.process_buffer()] == ['we agreed and you can write the search plan by next friday']

    def test_a_quoted_echo_across_lines_is_dropped(self):
        # The loop of a real session: the chained prompt came back quoted, then twice in one line.
        line = '"We\'re going to show you a couple of things."'
        uc, _ = _use_case(
            [_seg(f'{line}', 0.0), _seg(f'{line} {line}', 1.0), _seg(f'{line}', 2.0), _seg('Back to the agenda.', 3.0)]
        )
        uc.feed_audio(speech_like(SAMPLE_RATE * 5))
        assert [s.text for s in uc.process_buffer()] == [line, 'Back to the agenda.']

    def test_a_sentence_said_twice_is_kept(self):
        uc, _ = _use_case([_seg('No. No. I mean Friday.')])
        uc.feed_audio(speech_like(SAMPLE_RATE * 3))
        assert [s.text for s in uc.process_buffer()] == ['No. No. I mean Friday.']

    def test_a_line_equal_to_the_previous_one_is_dropped(self):
        uc, _ = _use_case([_seg('We are going to show you a couple of things.')])
        uc.feed_audio(speech_like(SAMPLE_RATE * 3))
        first = uc.process_buffer()
        uc.feed_audio(speech_like(SAMPLE_RATE * 3))
        assert len(first) == 1
        assert uc.process_buffer() == []


class TestACutSentence:
    """A chunk cut at the size limit can end mid-word; whisper then invents an ending."""

    def _use_case(self, replies):
        fake = FakeTranscriber()

        def transcribe(audio, language, hints=None):
            fake.transcribe_calls.append((audio, language, hints))
            return replies.pop(0)

        fake.transcribe = transcribe
        return TranscribeAudioUseCase(transcriber=fake, language='en', chunk_duration=10.0, overlap=1.0), fake

    def test_the_last_segment_waits_for_the_next_chunk(self):
        uc, fake = self._use_case(
            [
                [_seg('First sentence.', 0.0), _seg('I want to review the budget for the new', 7.0)],
                [_seg('I want to review the budget for the new servers before Friday.', 0.0)],
            ]
        )
        uc.feed_audio(speech_like(SAMPLE_RATE * 10))
        uc.set_session_offset(10.0)
        assert [s.text for s in uc.process_buffer()] == ['First sentence.']
        assert len(uc._buffer) == SAMPLE_RATE * 3  # noqa: SLF001 -- from the held segment's start
        uc.feed_audio(speech_like(SAMPLE_RATE * 2))
        uc.set_session_offset(12.0)
        second = uc.flush()
        assert [s.text for s in second] == ['I want to review the budget for the new servers before Friday.']
        assert second[0].wall_start == pytest.approx(7.0)
        assert fake.transcribe_calls[1][2] == ['First sentence.']

    def test_a_chunk_that_ends_at_a_pause_keeps_its_last_segment(self):
        uc, _ = self._use_case([[_seg('First.', 0.0), _seg('Second.', 2.0)]])
        uc.feed_audio(speech_like(SAMPLE_RATE * 5))
        assert [s.text for s in uc.process_buffer()] == ['First.', 'Second.']

    def test_flush_keeps_its_last_segment(self):
        uc, _ = self._use_case([[_seg('First.', 0.0), _seg('Second.', 8.0)]])
        uc.feed_audio(speech_like(SAMPLE_RATE * 10))
        assert [s.text for s in uc.flush()] == ['First.', 'Second.']

    def test_the_live_path_holds_the_cut_segment_too(self):
        # The live worker prepares a chunk, keeps feeding audio while whisper runs, then applies the result.
        uc, _ = self._use_case([])
        uc.feed_audio(speech_like(SAMPLE_RATE * 10))
        uc.set_session_offset(10.0)
        audio, _hints, wall_start, first = uc.prepare_buffer()
        uc.feed_audio(np.full(SAMPLE_RATE * 2, 0.25, dtype=np.float32))  # arrives during the transcription
        kept = uc.apply_result([_seg('First sentence.', 0.0), _seg('I want to review the', 7.0)], wall_start, first)
        assert [s.text for s in kept] == ['First sentence.']
        buffered = uc._buffer  # noqa: SLF001 -- the held audio, then the audio that arrived
        np.testing.assert_array_equal(buffered[: SAMPLE_RATE * 3], audio[SAMPLE_RATE * 7 :])
        np.testing.assert_allclose(buffered[SAMPLE_RATE * 3 :], 0.25)
        assert len(buffered) == SAMPLE_RATE * 5

    def test_a_final_apply_keeps_its_last_segment(self):
        uc, _ = self._use_case([])
        uc.feed_audio(speech_like(SAMPLE_RATE * 10))
        _audio, _hints, wall_start, first = uc.prepare_buffer()
        kept = uc.apply_result([_seg('First.', 0.0), _seg('Second.', 7.0)], wall_start, first, final=True)
        assert [s.text for s in kept] == ['First.', 'Second.']
