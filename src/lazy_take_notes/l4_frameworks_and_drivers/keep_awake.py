"""Prevent the machine from sleeping while a live recording is active.

macOS only (uses ``caffeinate``); a no-op on other platforms. The inhibitor is
tied to this process via ``-w <pid>``, so the assertion is released even if the
process is killed without a clean shutdown.
"""

from __future__ import annotations

import os
import subprocess  # noqa: S404 -- launches caffeinate with a fixed arg list, not shell=True
import sys
from collections.abc import Iterator
from contextlib import contextmanager


def inhibit_sleep() -> subprocess.Popen | None:
    """Start a display/idle-sleep inhibitor for this process, or None if unsupported."""
    if sys.platform != 'darwin':
        return None
    try:
        return subprocess.Popen(  # noqa: S603 -- fixed arg list, not shell=True
            ['caffeinate', '-di', '-w', str(os.getpid())],  # noqa: S607 -- caffeinate is a macOS system binary on PATH
            # Own session so a terminal Ctrl-C doesn't kill the inhibitor; it stays
            # up through the graceful-stop / final-digest window and is released when
            # this process exits (the -w pid tie), or by release_sleep().
            start_new_session=True,
        )
    except (OSError, ValueError):  # pragma: no cover -- caffeinate missing or unspawnable
        return None


def release_sleep(handle: subprocess.Popen | None) -> None:
    """Release a previously started inhibitor."""
    if handle is not None:
        handle.terminate()


@contextmanager
def keep_awake() -> Iterator[None]:
    """Keep the machine (and display) awake for the duration of the block."""
    handle = inhibit_sleep()
    try:
        yield
    finally:
        release_sleep(handle)
