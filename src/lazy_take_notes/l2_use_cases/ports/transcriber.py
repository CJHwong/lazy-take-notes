"""Port: speech-to-text transcription engine."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from lazy_take_notes.l1_entities.transcript import TranscriptSegment


class Transcriber(Protocol):
    """Abstract transcription engine. Zero framework types leak through."""

    def load_model(self, model_path: str) -> None:
        """Load the transcription model from the given path."""
        ...

    def transcribe(
        self,
        audio: np.ndarray,
        language: str,
        hints: list[str] | None = None,
    ) -> list[TranscriptSegment]:
        """Transcribe audio buffer into transcript segments."""
        ...

    def close(self) -> None:
        """Release underlying resources."""
        ...
