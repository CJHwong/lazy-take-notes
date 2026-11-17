"""Textual's terminal output survives a brief EBADF on fd 2."""

from __future__ import annotations

import errno
import io
import logging
import os
import sys
import threading
from typing import Any

import pytest

from lazy_take_notes.l4_frameworks_and_drivers.drivers import terminal_output
from lazy_take_notes.l4_frameworks_and_drivers.drivers.terminal_output import TerminalOutput, with_terminal_output
from lazy_take_notes.l4_frameworks_and_drivers.screen_base import SessionHostApp

# The tests patch os.write itself, so keep the real one.
REAL_WRITE = os.write


@pytest.fixture
def pipe():
    read_end, write_end = os.pipe()
    stream = io.TextIOWrapper(io.FileIO(write_end, 'w'), encoding='utf-8')
    yield read_end, stream
    stream.close()
    os.close(read_end)


def _read_all(read_end: int, size: int) -> bytes:
    chunks = b''
    while len(chunks) < size:
        chunks += os.read(read_end, size - len(chunks))
    return chunks


class FailingWrite:
    """os.write that fails with `errors` first, then writes at most `limit` bytes per call."""

    def __init__(self, errors: list[int], limit: int | None = None) -> None:
        self.errors = list(errors)
        self.limit = limit
        self.calls = 0

    def __call__(self, fd: int, data) -> int:
        self.calls += 1
        if self.errors:
            code = self.errors.pop(0)
            raise OSError(code, os.strerror(code))
        return REAL_WRITE(fd, bytes(data[: self.limit]) if self.limit else data)


class TestWrite:
    def test_text_waits_for_the_flush(self, pipe):
        read_end, stream = pipe
        output = TerminalOutput(stream)
        output.write('frame 1 ')
        output.write('frame 2 ✓')
        os.set_blocking(read_end, False)
        with pytest.raises(BlockingIOError):
            os.read(read_end, 100)
        output.flush()
        expected = 'frame 1 frame 2 ✓'.encode()
        assert _read_all(read_end, len(expected)) == expected

    def test_a_brief_ebadf_is_retried_and_writes_each_byte_once(self, pipe, monkeypatch):
        read_end, stream = pipe
        failing = FailingWrite([errno.EBADF, errno.EBADF])
        monkeypatch.setattr(terminal_output.os, 'write', failing)
        output = TerminalOutput(stream)
        output.write('x' * 100_000)
        threading.Thread(target=output.flush).start()
        assert _read_all(read_end, 100_000) == b'x' * 100_000
        assert failing.calls >= 3

    def test_a_partial_write_sends_the_rest(self, pipe, monkeypatch):
        read_end, stream = pipe
        monkeypatch.setattr(terminal_output.os, 'write', FailingWrite([], limit=7))
        output = TerminalOutput(stream)
        output.write('abcdefghijklmnopqrstuvwxyz')
        output.flush()
        assert _read_all(read_end, 26) == b'abcdefghijklmnopqrstuvwxyz'


class TestNeverRaises:
    def test_a_lasting_ebadf_drops_the_text_and_logs_it(self, pipe, monkeypatch, caplog):
        read_end, stream = pipe
        monkeypatch.setattr(terminal_output, 'RETRY_LIMIT_S', 0.01)
        monkeypatch.setattr(terminal_output.os, 'write', FailingWrite([errno.EBADF] * 10_000))
        output = TerminalOutput(stream)
        output.write('lost')
        with caplog.at_level(logging.WARNING, logger='ltn.app'):
            output.flush()
        assert 'Bad file descriptor' in caplog.text
        assert '4 bytes' in caplog.text

        monkeypatch.setattr(terminal_output.os, 'write', REAL_WRITE)
        output.write('next')
        output.flush()
        assert _read_all(read_end, 4) == b'next'

    def test_another_os_error_drops_the_text_at_once(self, pipe, monkeypatch):
        _read_end, stream = pipe
        failing = FailingWrite([errno.EIO, errno.EIO])
        monkeypatch.setattr(terminal_output.os, 'write', failing)
        output = TerminalOutput(stream)
        output.write('gone')
        output.flush()
        assert failing.calls == 1


class TestDriver:
    def test_the_driver_writes_through_terminal_output(self, pipe):
        _read_end, stream = pipe

        class Driver:
            def __init__(self, app) -> None:
                self.app = app
                self._file = stream

        wrapped: Any = with_terminal_output  # the stand-in driver has only what the wrapper touches
        driver = wrapped(Driver)('app')
        assert isinstance(driver._file, TerminalOutput)
        assert driver.app == 'app'
        assert driver._file.fileno() == stream.fileno()
        assert driver._file.isatty() is False

    @pytest.mark.skipif(sys.platform == 'win32', reason='LinuxDriver needs termios')
    async def test_every_session_host_uses_it_with_textual_s_own_driver(self):
        driver_class = SessionHostApp().get_driver_class()
        driver = driver_class(app=None)  # ty: ignore[invalid-argument-type] -- the driver only stores the app
        assert isinstance(driver._file, TerminalOutput)  # ty: ignore[missing-attribute] -- Textual's private writer file

    def test_windows_keeps_textual_s_driver(self, monkeypatch):
        # The Windows console stream converts text for the console; raw os.write would not.
        from textual.app import App

        from lazy_take_notes.l4_frameworks_and_drivers import screen_base

        monkeypatch.setattr(screen_base.sys, 'platform', 'win32')
        assert SessionHostApp().get_driver_class() is App().get_driver_class()
