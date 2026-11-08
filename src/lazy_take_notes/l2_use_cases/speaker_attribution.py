"""Use case: label timestamped transcript lines with anonymous speakers."""

from __future__ import annotations

from lazy_take_notes.l1_entities.speaker_turn import SpeakerTurn


def speaker_at(turns: list[SpeakerTurn], at: float) -> int | None:
    """Speaker talking at a given time, or None when no turn covers it.

    Turns may overlap. The one that started most recently wins, because an
    overlapping stretch usually means a backchannel from the previous speaker.
    """
    covering = [turn for turn in turns if turn.start <= at < turn.end]
    if not covering:
        return None
    return max(covering, key=lambda turn: turn.start).speaker
