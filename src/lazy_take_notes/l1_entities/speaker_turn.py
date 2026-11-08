"""Entity: one stretch of speech by one anonymous speaker."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SpeakerTurn:
    """A speaker's contiguous speech, in seconds from the start of the recording.

    The speaker number is local to one diarization run. It is not an identity:
    the same person gets a different number in another session.
    """

    start: float
    end: float
    speaker: int
