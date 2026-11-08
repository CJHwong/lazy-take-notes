"""Use case: label timestamped transcript lines with anonymous speakers."""

from __future__ import annotations

from lazy_take_notes.l1_entities.speaker_turn import SpeakerTurn


def speaker_for_span(turns: list[SpeakerTurn], start: float, end: float, tolerance: float = 1.5) -> int | None:
    """Speaker for a transcript line, using its whole span rather than its start.

    A line that starts in a short gap belongs to the speaker it overlaps. A line
    with no overlap at all stays unlabeled unless a turn sits within `tolerance`
    seconds, which covers the usual few-hundred-millisecond mismatch between
    whisper segment edges and diarization turn edges.
    """
    if end <= start:
        end = start + 0.01

    best: SpeakerTurn | None = None
    best_overlap = 0.0
    for turn in turns:
        overlap = min(end, turn.end) - max(start, turn.start)
        if overlap > best_overlap:
            best_overlap = overlap
            best = turn
    if best is not None:
        return best.speaker

    nearest: SpeakerTurn | None = None
    distance = float('inf')
    for turn in turns:
        gap = max(turn.start - end, start - turn.end, 0.0)
        if gap < distance:
            distance = gap
            nearest = turn
    if nearest is not None and distance <= tolerance:
        return nearest.speaker
    return None


def speaker_at(turns: list[SpeakerTurn], at: float) -> int | None:
    """Speaker talking at a given time, or None when no turn covers it.

    Turns may overlap. The one that started most recently wins, because an
    overlapping stretch usually means a backchannel from the previous speaker.
    """
    covering = [turn for turn in turns if turn.start <= at < turn.end]
    if not covering:
        return None
    return max(covering, key=lambda turn: turn.start).speaker
