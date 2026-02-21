"""Use case: transcribe audio buffers — buffer management, VAD, overlap handling."""

from __future__ import annotations

import numpy as np

from lazy_take_notes.l1_entities.audio_constants import SAMPLE_RATE
from lazy_take_notes.l1_entities.transcript import TranscriptSegment
from lazy_take_notes.l2_use_cases.ports.transcriber import Transcriber


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

        self._buffer = np.array([], dtype=np.float32)
        self._is_first_chunk = True
        self._session_offset: float = 0.0  # wall-clock offset of buffer start

    @property
    def overlap(self) -> float:
        return self._overlap_samples / SAMPLE_RATE

    def set_session_offset(self, offset: float) -> None:
        """Set the current wall-clock offset in seconds from session start."""
        self._session_offset = offset

    def feed_audio(self, data: np.ndarray) -> None:
        """Append raw audio samples to the internal buffer."""
        self._buffer = np.concatenate([self._buffer, data.flatten()])

    def reset_buffer(self) -> None:
        """Discard accumulated audio (e.g. after pause)."""
        self._buffer = np.array([], dtype=np.float32)

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
        if the buffer is silent (nothing to transcribe).

        Resets the internal buffer to the overlap tail immediately so the caller
        can continue feeding audio without waiting for transcription to finish.
        Call apply_result() from the audio loop thread once transcription completes.
        """
        buf = self._buffer

        rms = np.sqrt(np.mean(buf**2))
        if rms < self._silence_threshold:
            self._buffer = (
                buf[-self._overlap_samples :] if self._overlap_samples > 0 else np.array([], dtype=np.float32)
            )
            return None

        snapshot = buf.copy()
        buffer_wall_start = self._session_offset - len(buf) / SAMPLE_RATE
        current_hints = list(self._current_hints)
        is_first = self._is_first_chunk

        self._buffer = buf[-self._overlap_samples :] if self._overlap_samples > 0 else np.array([], dtype=np.float32)
        self._is_first_chunk = False

        return snapshot, current_hints, buffer_wall_start, is_first

    def apply_result(
        self,
        segments: list[TranscriptSegment],
        buffer_wall_start: float,
        is_first_chunk: bool,
    ) -> list[TranscriptSegment]:
        """Filter and adjust segments from an off-thread transcription.

        Deduplicates the overlap region, adjusts wall times to absolute session
        offsets, and updates the prompt chain.  Must be called from the audio
        loop thread (single-threaded with feed_audio / prepare_buffer).
        """
        min_start = 0.0 if is_first_chunk else self.overlap
        new_segments: list[TranscriptSegment] = []
        last_text = None

        for seg in segments:
            if seg.wall_end > min_start:
                adjusted = TranscriptSegment(
                    text=seg.text,
                    wall_start=buffer_wall_start + seg.wall_start,
                    wall_end=buffer_wall_start + seg.wall_end,
                )
                new_segments.append(adjusted)
                last_text = seg.text

        if last_text:
            self._current_hints = list(self._recognition_hints) + [last_text]

        return new_segments

    def process_buffer(self) -> list[TranscriptSegment]:
        """Transcribe the current buffer and return new segments.

        Handles overlap dedup, silence skip, and prompt chaining.
        Retains overlap tail in the buffer for next cycle.
        """
        buf = self._buffer

        # Skip if entire buffer is silence
        rms = np.sqrt(np.mean(buf**2))
        if rms < self._silence_threshold:
            self._buffer = (
                buf[-self._overlap_samples :] if self._overlap_samples > 0 else np.array([], dtype=np.float32)
            )
            return []

        # Compute wall-clock start of this buffer
        buffer_wall_start = self._session_offset - len(buf) / SAMPLE_RATE

        segments = self._transcriber.transcribe(
            audio=buf,
            language=self._language,
            hints=self._current_hints,
        )

        # Filter out overlap region (except for first chunk)
        min_start = 0.0 if self._is_first_chunk else self.overlap
        new_segments: list[TranscriptSegment] = []
        last_text = None

        for seg in segments:
            if seg.wall_end > min_start:
                # Adjust wall times to absolute session offset
                adjusted = TranscriptSegment(
                    text=seg.text,
                    wall_start=buffer_wall_start + seg.wall_start,
                    wall_end=buffer_wall_start + seg.wall_end,
                )
                new_segments.append(adjusted)
                last_text = seg.text

        self._is_first_chunk = False

        # Prompt chaining
        if last_text:
            self._current_hints = list(self._recognition_hints) + [last_text]

        # Retain overlap tail
        if self._overlap_samples > 0:
            self._buffer = buf[-self._overlap_samples :]
        else:
            self._buffer = np.array([], dtype=np.float32)

        return new_segments

    def flush(self) -> list[TranscriptSegment]:
        """Process any remaining audio on shutdown."""
        if len(self._buffer) < self._min_speech_samples:
            return []
        rms = np.sqrt(np.mean(self._buffer**2))
        if rms < self._silence_threshold:
            return []
        return self.process_buffer()
