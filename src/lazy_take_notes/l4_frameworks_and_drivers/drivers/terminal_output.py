"""Keep Textual's terminal output alive through a brief EBADF on fd 2.

sounddevice re-initializes PortAudio with dup2() onto fd 2, to hide PortAudio
messages. On macOS, a write to fd 2 from another thread during that dup2()
fails with EBADF. Textual's writer thread does not catch the error. The thread
dies, the screen freezes, and the app hangs at exit on the writer's full queue.

TerminalOutput holds the text itself and writes it with os.write(). A failed
os.write() writes nothing, so a retry writes each byte once. Python's text
stream cannot promise that: it drops a large write that fails.
"""

from __future__ import annotations

import errno
import logging
import os
import time
from typing import Any, TextIO

from textual.driver import Driver

log = logging.getLogger('ltn.app')

# The dup2() window lasts microseconds. A longer EBADF means fd 2 is really gone.
RETRY_INTERVAL_S = 0.001
RETRY_LIMIT_S = 1.0


class TerminalOutput:
    """The file Textual's writer thread writes to. Never raises."""

    def __init__(self, stream: TextIO) -> None:
        self._stream = stream
        self._encoding = stream.encoding or 'utf-8'
        self._pending = bytearray()

    def write(self, text: str) -> None:
        self._pending += text.encode(self._encoding, 'backslashreplace')

    def flush(self) -> None:
        deadline = time.monotonic() + RETRY_LIMIT_S
        while self._pending:
            try:
                written = os.write(self._stream.fileno(), self._pending)
            except OSError as error:
                if error.errno == errno.EBADF and time.monotonic() < deadline:
                    time.sleep(RETRY_INTERVAL_S)
                    continue
                self._drop(error)
                return
            del self._pending[:written]

    def _drop(self, error: OSError) -> None:
        log.warning('Terminal output failed (%s); dropped %d bytes', error, len(self._pending))
        self._pending.clear()

    def fileno(self) -> int:
        return self._stream.fileno()

    def isatty(self) -> bool:
        return self._stream.isatty()


def with_terminal_output(driver_class: type[Driver]) -> type[Driver]:
    """Subclass a Textual driver so its writer thread writes through TerminalOutput."""
    base: Any = driver_class  # Textual picks the class at run time, so the base has no static type

    class TerminalOutputDriver(base):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self._file = TerminalOutput(self._file)

    return TerminalOutputDriver
