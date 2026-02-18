"""Transcript segment entity."""

from __future__ import annotations

from pydantic import BaseModel, Field


class TranscriptSegment(BaseModel):
    """A single transcribed speech segment."""

    text: str
    wall_start: float = Field(description='Wall-clock offset in seconds from session start')
    wall_end: float = Field(description='Wall-clock offset in seconds from session start')
