"""The session folder, shared by the app and the WAV writer."""

from __future__ import annotations

import errno
import threading
from pathlib import Path

# The writer holds the lock for one chunk. Longer means the disk is stalled.
RENAME_WAIT_S = 2.0


class SessionFolder:
    """The session folder, shared by the app and the WAV writer.

    The user can rename the folder while it records. The writer checks that its
    file is still at its path, to catch Move to Trash, so a rename it does not
    know about looks like a removed file and starts a second recording. A rename
    through this object holds the writer's lock and moves the writer's path too.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.lock = threading.Lock()

    def rename(self, new_path: Path) -> None:
        # Called on the UI thread: a stalled disk must not freeze the screen.
        if not self.lock.acquire(timeout=RENAME_WAIT_S):
            raise OSError(errno.EBUSY, 'the disk is busy writing the recording; try again')
        try:
            self.path.rename(new_path)
            self.path = new_path
        finally:
            self.lock.release()
