"""Status record for a headless session, persisted so the CLI can check in."""

from __future__ import annotations

from dataclasses import dataclass

# Lifecycle states a headless session moves through.
STARTING = 'starting'
LOADING_MODEL = 'loading_model'
DOWNLOADING = 'downloading'
RECORDING = 'recording'
DIGESTING = 'digesting'
STOPPED = 'stopped'
ERROR = 'error'

# States that imply a live process should be running; if the PID is dead while
# in one of these, the session crashed.
LIVE_STATES = frozenset({STARTING, LOADING_MODEL, DOWNLOADING, RECORDING, DIGESTING})


@dataclass
class SessionStatus:
    """Mutable snapshot of a headless session's progress."""

    state: str
    pid: int
    started_at: str
    updated_at: str
    segment_count: int = 0
    digest_count: int = 0
    error: str = ''
