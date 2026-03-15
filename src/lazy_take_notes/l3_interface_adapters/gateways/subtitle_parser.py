"""Gateway: parse WebVTT subtitle files into TranscriptSegment list."""

from __future__ import annotations

import re
from pathlib import Path

from lazy_take_notes.l1_entities.transcript import TranscriptSegment

# Matches WebVTT timestamps in either form:
#   HH:MM:SS.mmm --> HH:MM:SS.mmm
#   MM:SS.mmm    --> MM:SS.mmm      (hours field is optional per the WebVTT spec)
_TIMESTAMP_RE = re.compile(
    r'(?:(?P<sh>\d{2}):)?(?P<sm>\d{2}):(?P<ss>\d{2})\.(?P<sms>\d{3})\s+-->\s+'
    r'(?:(?P<eh>\d{2}):)?(?P<em>\d{2}):(?P<es>\d{2})\.(?P<ems>\d{3})'
)


def _to_seconds(h: str | None, m: str, s: str, ms: str) -> float:
    hours = int(h) if h is not None else 0
    return hours * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def _strip_inline_tags(text: str) -> str:
    """Remove VTT inline tags: <c>, <00:00:01.000>, </c>, etc."""
    return re.sub(r'<[^>]+>', '', text).strip()


def parse_vtt_to_segments(vtt_path: Path) -> list[TranscriptSegment]:
    """Parse a WebVTT file into TranscriptSegment list with auto-caption dedup."""
    raw = vtt_path.read_text(encoding='utf-8', errors='replace')
    lines = raw.splitlines()

    # Collect raw cues: (start, end, text)
    cues: list[tuple[float, float, str]] = []
    idx = 0
    while idx < len(lines):
        match = _TIMESTAMP_RE.match(lines[idx])
        if match:
            start = _to_seconds(match.group('sh'), match.group('sm'), match.group('ss'), match.group('sms'))
            end = _to_seconds(match.group('eh'), match.group('em'), match.group('es'), match.group('ems'))
            idx += 1
            text_lines: list[str] = []
            while idx < len(lines) and lines[idx].strip():
                text_lines.append(lines[idx])
                idx += 1
            text = _strip_inline_tags(' '.join(text_lines))
            if text:
                cues.append((start, end, text))
        else:
            idx += 1

    if not cues:
        return []

    # Auto-caption dedup: collapse rolling-window cues where prev is a substring of next
    segments: list[TranscriptSegment] = []
    prev_start, prev_end, prev_text = cues[0]

    for start, end, text in cues[1:]:
        if prev_text in text:
            # Current cue is a superset — extend the window, keep longer text
            prev_end = max(prev_end, end)
            prev_text = text
        else:
            segments.append(TranscriptSegment(text=prev_text, wall_start=prev_start, wall_end=prev_end))
            prev_start, prev_end, prev_text = start, end, text

    segments.append(TranscriptSegment(text=prev_text, wall_start=prev_start, wall_end=prev_end))
    return segments
