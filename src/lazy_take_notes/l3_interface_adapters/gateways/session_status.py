"""Read/write the headless-session status file and check process liveness."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

from lazy_take_notes.l1_entities.session_status import SessionStatus

STATUS_FILE = '.status.json'


def write_status(session_dir: Path, status: SessionStatus) -> Path:
    """Atomically write *status* to ``<session_dir>/.status.json``."""
    path = session_dir / STATUS_FILE
    tmp = session_dir / (STATUS_FILE + '.tmp')
    tmp.write_text(json.dumps(asdict(status)), encoding='utf-8')
    tmp.replace(path)
    return path


def read_status(session_dir: Path) -> SessionStatus | None:
    """Return the session's status, or None if absent or unreadable."""
    path = session_dir / STATUS_FILE
    if not path.exists():
        return None
    try:
        return SessionStatus(**json.loads(path.read_text(encoding='utf-8')))
    except (OSError, ValueError, TypeError):
        return None


def is_pid_alive(pid: int) -> bool:
    """Return True if *pid* names a running process.

    Used to tell a still-running session apart from one that crashed without
    updating its status file.
    """
    if not isinstance(pid, int) or pid <= 0:  # guards a corrupt status file (e.g. pid stored as a string)
        return False
    if sys.platform == 'win32':  # pragma: no cover -- liveness check is POSIX-only; assume alive on Windows
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # exists but owned by another user
        return True
    except OSError:
        return False
    return True
