"""Read-only access to saved session directories on disk.

Shared by the session picker (TUI), the ``view`` command, and the read-only
``ls`` / ``transcript`` / ``notes`` CLI commands. Sessions are timestamped
subdirectories; lexical sort on the timestamp-prefixed name equals
chronological order, so the newest session is the current one.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from lazy_take_notes.l1_entities.session_files import NOTES, TRANSCRIPT


@dataclass(frozen=True)
class SessionInfo:
    """A saved session directory and what it contains."""

    dir: Path
    name: str
    has_notes: bool


def list_sessions(base_dir: Path) -> list[SessionInfo]:
    """Return sessions under *base_dir*, newest-first.

    A session is any subdirectory containing a transcript file (current or
    legacy name). Sort is lexical on the directory name, which equals
    chronological order because names are timestamp-prefixed.
    """
    if not base_dir.exists():
        return []

    sessions = []
    for child in sorted(base_dir.iterdir(), reverse=True):
        if not child.is_dir() or not TRANSCRIPT.resolve(child):
            continue
        sessions.append(SessionInfo(dir=child, name=child.name, has_notes=NOTES.resolve(child) is not None))
    return sessions


def latest_session(base_dir: Path) -> SessionInfo | None:
    """Return the most recent session, or None when there are none."""
    sessions = list_sessions(base_dir)
    return sessions[0] if sessions else None


def read_transcript(session_dir: Path) -> str | None:
    """Return the transcript text for *session_dir*, or None if absent."""
    path = TRANSCRIPT.resolve(session_dir)
    return path.read_text(encoding='utf-8') if path else None


def read_notes(session_dir: Path) -> str | None:
    """Return the notes markdown for *session_dir*, or None if absent."""
    path = NOTES.resolve(session_dir)
    return path.read_text(encoding='utf-8') if path else None
