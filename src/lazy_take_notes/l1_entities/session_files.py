"""Canonical filenames for session artifacts, with legacy fallback support."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SessionFile:
    """A session output file with its current and legacy name."""

    name: str
    legacy: str

    def resolve(self, directory: Path) -> Path | None:
        """Return path if current or legacy name exists in *directory*, else None."""
        path = directory / self.name
        if path.exists():
            return path
        legacy_path = directory / self.legacy
        return legacy_path if legacy_path.exists() else None


TRANSCRIPT = SessionFile('transcript.txt', 'transcript_raw.txt')
NOTES = SessionFile('notes.md', 'digest.md')
CONTEXT = SessionFile('context.txt', 'session_context.txt')
DEBUG_LOG = SessionFile('debug.log', 'ltn_debug.log')
