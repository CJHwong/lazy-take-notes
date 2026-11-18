"""An exclusive lock on a file, held across processes and across open files in one process.

Two downloads of one asset write the same .part file, and both then fail their
digest check. The kiosk runs the preflight and Settings in one process with two
stores, and a second terminal can run another session, so a thread lock is not
enough. flock() locks the open file, so two opens in one process also conflict;
POSIX lockf() would not.
"""

from __future__ import annotations

import os
import sys
import threading
from collections.abc import Callable, Generator
from contextlib import contextmanager
from pathlib import Path

POLL_S = 0.2


@contextmanager
def exclusive(
    path: Path,
    *,
    cancelled: Callable[[], bool] = lambda: False,
    on_wait: Callable[[], None] | None = None,
) -> Generator[None]:
    """Hold an exclusive lock on *path*. While another holder has it, call *on_wait* every POLL_S.

    Raises InterruptedError when *cancelled* returns True during the wait.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        while not _try_lock(fd):
            if cancelled():
                raise InterruptedError(f'cancelled while waiting for {path.name}')
            if on_wait is not None:
                on_wait()
            threading.Event().wait(POLL_S)
        try:
            yield
        finally:
            _unlock(fd)
    finally:
        os.close(fd)


if sys.platform == 'win32':  # pragma: no cover -- Windows only
    import msvcrt

    def _try_lock(fd: int) -> bool:
        os.lseek(fd, 0, os.SEEK_SET)  # msvcrt locks bytes from the current position
        try:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        except OSError:
            return False
        return True

    def _unlock(fd: int) -> None:
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)

else:
    import fcntl

    def _try_lock(fd: int) -> bool:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        return True

    def _unlock(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_UN)
