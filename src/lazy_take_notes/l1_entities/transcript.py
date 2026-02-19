"""Transcript segment entity."""

from __future__ import annotations

from pydantic import BaseModel, Field


def format_wall_time(seconds: float) -> str:
    """Format seconds as HH:MM:SS for wall-clock display."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f'{hours:02d}:{minutes:02d}:{secs:02d}'


class TranscriptSegment(BaseModel):
    """A single transcribed speech segment."""

    text: str
    wall_start: float = Field(description='Wall-clock offset in seconds from session start')
    wall_end: float = Field(description='Wall-clock offset in seconds from session start')
