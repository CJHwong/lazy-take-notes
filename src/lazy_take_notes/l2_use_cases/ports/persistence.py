"""Port: persistence gateway for saving session artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from lazy_take_notes.l1_entities.transcript import TranscriptSegment


class PersistenceGateway(Protocol):
    """Abstract persistence for transcripts, digests, and history."""

    def save_transcript_lines(self, segments: list[TranscriptSegment], *, append: bool = True) -> Path:
        """Save transcript segments to disk."""
        ...

    def save_digest_md(self, markdown: str, digest_number: int) -> Path:
        """Save the latest digest markdown."""
        ...

    def save_history(self, markdown: str, digest_number: int, *, is_final: bool = False) -> Path:
        """Save numbered history file."""
        ...
