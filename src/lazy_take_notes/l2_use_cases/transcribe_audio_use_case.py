"""Use case: transcribe audio buffers — buffer management, VAD, overlap handling."""

from __future__ import annotations

import re

import numpy as np

from lazy_take_notes.l1_entities.audio_constants import SAMPLE_RATE
from lazy_take_notes.l1_entities.transcript import TranscriptSegment
from lazy_take_notes.l2_use_cases.ports.transcriber import Transcriber

# Whisper writes a sentence for a click, a cough or steady noise ("Thank you.",
# "*sad*", a line from the prompt). A chunk is sent only when it holds this much
# audio clearly above its own noise floor; a frame counts when it is louder than
# the silence threshold and twice the chunk's quietest tenth.
MIN_VOICED_S = 1.0
_FRAME = SAMPLE_RATE // 10
# A line that is only an annotation (*clap*, [music], (laughs), ♪) or only punctuation.
_NOT_SPEECH = re.compile(r'\s*(?:[*\[(♪][^\]*)♪]*[*\])♪]\s*|[\W_]*)')
# Hold back a cut chunk's last segment only when another one is sent; one long segment is sent as is.
_MIN_SEGMENTS_TO_HOLD = 2
# A sentence said this many times in a row is a decoding loop, not speech.
_LOOP_REPEATS = 3
_LOOP = re.compile(rf'(\S.{{8,}}?)(?:\s*\1){{{_LOOP_REPEATS - 1},}}')


def voiced_seconds(audio: np.ndarray, silence_threshold: float) -> float:
    """Seconds of *audio* clearly louder than both the threshold and the chunk's noise floor."""
    count = len(audio) // _FRAME
    if count == 0:
        return 0.0
    rms = np.sqrt(np.mean(audio[: count * _FRAME].reshape(count, _FRAME) ** 2, axis=1))
    floor = float(np.percentile(rms, 10))
    return float(np.count_nonzero(rms >= max(silence_threshold, 2 * floor))) * _FRAME / SAMPLE_RATE


def _collapse_loops(text: str) -> str:
    # A backreference, not a sentence split: whisper's loops come quoted or with no punctuation at all.
    return ' '.join(_LOOP.sub(r'\1', text).split())


class TranscribeAudioUseCase:
    """Encapsulates buffer management, VAD triggering, overlap, and prompt chaining.

    Does NO I/O itself — audio data is fed in via ``feed_audio()``,
    transcript segments come out via ``process_buffer()``.
    """

    def __init__(
        self,
        transcriber: Transcriber,
        language: str,
        chunk_duration: float = 25.0,
        overlap: float = 1.0,
        silence_threshold: float = 0.01,
        pause_duration: float = 1.5,
        recognition_hints: list[str] | None = None,
    ) -> None:
        self._transcriber = transcriber
        self._language = language
        self._recognition_hints: list[str] = list(recognition_hints) if recognition_hints else []
        self._current_hints: list[str] = list(self._recognition_hints)

        self._chunk_samples = int(SAMPLE_RATE * chunk_duration)
        self._overlap_samples = int(SAMPLE_RATE * overlap)
        self._pause_samples = int(SAMPLE_RATE * pause_duration)
        self._min_speech_samples = int(SAMPLE_RATE * 2.0)
        self._silence_threshold = silence_threshold

        # Audio buffer is stored in a pre-allocated array that grows exponentially
        # on demand. The valid prefix is `_storage[:_buffer_size]`; reads go through
        # the `_buffer` view. This makes feed_audio amortised O(1) in the chunk
        # size instead of O(buffer_size) — the previous np.concatenate-per-feed
        # turned long meetings under whisper backpressure into a quadratic CPU
        # sink that saturated the audio_worker consumer loop and caused live
        # level-meter lag.
        self._storage = np.zeros(SAMPLE_RATE * 10, dtype=np.float32)
        self._buffer_size = 0
        self._is_first_chunk = True
        self._session_offset: float = 0.0  # wall-clock offset of buffer start
        self._sent = np.array([], dtype=np.float32)  # the chunk handed out by prepare_buffer
        self._last_text: str | None = None  # the newest line sent out, to drop a repeat of it

    @property
    def overlap(self) -> float:
        return self._overlap_samples / SAMPLE_RATE

    @property
    def _buffer(self) -> np.ndarray:
        """View of the current valid audio samples — read-only conceptually.

        Callers that previously did `self._buffer = X` must instead use
        `_replace_buffer(X)` so the underlying storage stays in sync.
        """
        return self._storage[: self._buffer_size]

    def _replace_buffer(self, value: np.ndarray) -> None:
        """Reset buffer contents to `value`. Handles overlap-safe self-copies."""
        n = len(value)
        if n == 0:
            self._buffer_size = 0
            return
        if n > len(self._storage):
            new_cap = max(n, len(self._storage) * 2)
            self._storage = np.zeros(new_cap, dtype=np.float32)
            self._storage[:n] = value
        else:
            # `value` may be a view into self._storage (e.g. self._buffer[-overlap:]).
            # Slice-assigning across overlapping regions on the same array is undefined,
            # so take a copy when the source aliases our storage.
            src = value.copy() if value.base is self._storage else value
            self._storage[:n] = src
        self._buffer_size = n

    def set_session_offset(self, offset: float) -> None:
        """Set the current wall-clock offset in seconds from session start."""
        self._session_offset = offset

    def feed_audio(self, data: np.ndarray) -> None:
        """Append raw audio samples to the internal buffer (amortised O(1))."""
        flat = data.flatten() if data.ndim > 1 else data
        n = len(flat)
        if n == 0:
            return
        needed = self._buffer_size + n
        if needed > len(self._storage):
            new_cap = max(needed, len(self._storage) * 2)
            new_storage = np.zeros(new_cap, dtype=np.float32)
            new_storage[: self._buffer_size] = self._storage[: self._buffer_size]
            self._storage = new_storage
        self._storage[self._buffer_size : needed] = flat
        self._buffer_size = needed

    def reset_buffer(self) -> None:
        """Discard accumulated audio (e.g. after pause)."""
        self._buffer_size = 0

    def _not_worth_sending(self, buf: np.ndarray) -> bool:
        # Not the average over the chunk: a quiet lead pulls it under the threshold and loses the speech after it.
        voiced = voiced_seconds(buf, self._silence_threshold)
        return voiced == 0 or voiced < MIN_VOICED_S

    def _keep_tail(self, buf: np.ndarray) -> None:
        self._replace_buffer(
            buf[-self._overlap_samples :] if self._overlap_samples > 0 else np.array([], dtype=np.float32)
        )

    def _clean(self, segment: TranscriptSegment) -> TranscriptSegment | None:
        """Drop what whisper writes for non-speech, and collapse a decoding loop."""
        text = _collapse_loops(segment.text.strip())
        if _NOT_SPEECH.fullmatch(text) or self._repeats_last_line(text):
            return None
        self._last_text = text
        return TranscriptSegment(text=text, wall_start=segment.wall_start, wall_end=segment.wall_end)

    def _repeats_last_line(self, text: str) -> bool:
        # Prompt chaining feeds the last line back in, and whisper can echo it, once or several times.
        last = self._last_text
        return last is not None and re.fullmatch(rf'(?:\s*{re.escape(last)})+\s*', text) is not None

    def should_trigger(self) -> bool:
        """Check if the buffer should be processed."""
        if len(self._buffer) >= self._chunk_samples:
            return True

        if len(self._buffer) >= self._min_speech_samples + self._pause_samples:
            tail_rms = np.sqrt(np.mean(self._buffer[-self._pause_samples :] ** 2))
            body_rms = np.sqrt(np.mean(self._buffer[: -self._pause_samples] ** 2))
            if tail_rms < self._silence_threshold and body_rms >= self._silence_threshold:
                return True

        return False

    def prepare_buffer(self) -> tuple[np.ndarray, list[str], float, bool] | None:
        """Extract the current buffer for off-thread transcription.

        Returns (audio_snapshot, hints, buffer_wall_start, is_first_chunk), or None
        if the buffer is silent or empty (nothing to transcribe).

        Resets the internal buffer to the overlap tail immediately so the caller
        can continue feeding audio without waiting for transcription to finish.
        Call apply_result() from the audio loop thread once transcription completes.
        """
        buf = self._buffer

        if len(buf) == 0:
            return None

        if self._not_worth_sending(buf):
            self._keep_tail(buf)
            return None

        snapshot = buf.copy()
        self._sent = snapshot
        buffer_wall_start = self._session_offset - len(buf) / SAMPLE_RATE
        current_hints = list(self._current_hints)
        is_first = self._is_first_chunk

        self._replace_buffer(
            buf[-self._overlap_samples :] if self._overlap_samples > 0 else np.array([], dtype=np.float32)
        )
        self._is_first_chunk = False

        return snapshot, current_hints, buffer_wall_start, is_first

    def apply_result(
        self,
        segments: list[TranscriptSegment],
        buffer_wall_start: float,
        is_first_chunk: bool,
        *,
        final: bool = False,
    ) -> list[TranscriptSegment]:
        """Filter and adjust segments from an off-thread transcription.

        Deduplicates the overlap region, adjusts wall times to absolute session
        offsets, and updates the prompt chain.  Must be called from the audio
        loop thread (single-threaded with feed_audio / prepare_buffer).

        A chunk cut at the size limit can end mid-word, and whisper then invents
        an ending for it. Unless *final*, the last segment of such a chunk waits:
        its audio goes back in front of the buffer and is sent whole next time.
        """
        min_start = 0.0 if is_first_chunk else self.overlap
        new_segments: list[TranscriptSegment] = []
        last_text = None
        segments = [seg for seg in segments if seg.wall_end > min_start]
        held_from = None if final else self._cut_segment_start(segments)
        if held_from is not None:
            segments = segments[:-1]

        for seg in segments:
            if seg.wall_end > min_start:
                adjusted = self._clean(
                    TranscriptSegment(
                        text=seg.text,
                        wall_start=buffer_wall_start + seg.wall_start,
                        wall_end=buffer_wall_start + seg.wall_end,
                    )
                )
                if adjusted is not None:
                    new_segments.append(adjusted)
                    last_text = adjusted.text

        if last_text:
            self._current_hints = list(self._recognition_hints) + [last_text]
        if held_from is not None:
            self._resend_from(held_from)

        return new_segments

    def _cut_segment_start(self, segments: list[TranscriptSegment]) -> int | None:
        """Sample where the last segment starts, when the sent chunk was cut at the size limit."""
        sent = self._sent
        if len(sent) < self._chunk_samples or len(segments) < _MIN_SEGMENTS_TO_HOLD:
            return None
        start = int(segments[-1].wall_start * SAMPLE_RATE)
        return start if 0 < start < len(sent) else None

    def _resend_from(self, start: int) -> None:
        # The buffer holds the overlap tail of the sent chunk plus what arrived since.
        tail = min(self._overlap_samples, len(self._sent))
        self._replace_buffer(np.concatenate([self._sent[start:], self._buffer[tail:]]))
        # The next chunk starts where the held segment starts, so nothing in it is a repeat.
        self._is_first_chunk = True

    def process_buffer(self, *, final: bool = False) -> list[TranscriptSegment]:
        """Transcribe the current buffer now and return new segments (see apply_result)."""
        prepared = self.prepare_buffer()
        if prepared is None:
            return []
        audio, hints, buffer_wall_start, is_first_chunk = prepared
        segments = self._transcriber.transcribe(audio=audio, language=self._language, hints=hints)
        return self.apply_result(segments, buffer_wall_start, is_first_chunk, final=final)

    def flush(self) -> list[TranscriptSegment]:
        """Process any remaining audio on shutdown."""
        if len(self._buffer) < self._min_speech_samples:
            return []
        return self.process_buffer(final=True)
