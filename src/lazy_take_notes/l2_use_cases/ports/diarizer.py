"""Port: speaker diarization."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from lazy_take_notes.l1_entities.speaker_turn import SpeakerTurn


class Diarizer(Protocol):
    """Splits a recording into anonymous speaker turns."""

    def diarize(self, audio_path: Path, timeout: float = 300.0) -> list[SpeakerTurn]:
        """Return the speaker turns of one 16 kHz mono WAV file.

        Raises RuntimeError when the backend is missing, fails, or times out.
        """
        ...
