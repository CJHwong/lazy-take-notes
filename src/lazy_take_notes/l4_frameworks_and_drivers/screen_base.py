"""Base Screen for the single-app shell.

Every picker and mode UI is a Screen, so the kiosk runs one Textual App for the
whole session. Standalone commands and tests still need to run one Screen
without the shell, so ShellScreen provides run() and run_test() backed by a
minimal host App.
"""

from __future__ import annotations

import logging
import os
import signal
import sys
import threading
import time
from collections import deque
from typing import TypeVar

from textual.app import App
from textual.driver import Driver
from textual.screen import Screen
from textual.worker import WorkerFailed

from lazy_take_notes.l4_frameworks_and_drivers.drivers.terminal_output import with_terminal_output

T = TypeVar('T')

log = logging.getLogger('ltn.app')

# How long a shutdown signal waits for the audio worker to flush its last chunk.
SHUTDOWN_GRACE_S = 3.0
# Extra time for Textual's own shutdown before the watchdog ends the process.
FORCE_EXIT_AFTER_S = 5.0
# More unexpected errors than this inside the window means an error loop.
ERROR_BURST_LIMIT = 5
ERROR_BURST_WINDOW_S = 10.0
# During an error loop, log a count of the errors this often instead of each traceback.
STORM_LOG_INTERVAL_S = 60.0


def shutdown_signals() -> tuple[signal.Signals, ...]:
    """SIGTERM and SIGHUP where the platform has them. Windows has no SIGHUP."""
    names = ('SIGTERM', 'SIGHUP')
    return tuple(getattr(signal, name) for name in names if hasattr(signal, name))


def arm_exit_watchdog(signum: int) -> None:
    """End the process if the clean shutdown hangs.

    When the window closes, the terminal is gone and Textual's writer thread
    dies; Textual's shutdown then blocks forever on its full write queue. By
    the time this fires, the audio flush and context save have had their grace
    period, so a hard exit loses nothing that was going to be saved.
    """
    watchdog = threading.Timer(SHUTDOWN_GRACE_S + FORCE_EXIT_AFTER_S, _force_exit, (signum,))
    watchdog.daemon = True
    watchdog.start()


def _force_exit(signum: int) -> None:
    log.error('Shutdown after %s did not finish; forcing exit', signal.Signals(signum).name)
    for handler in logging.getLogger('ltn').handlers:
        handler.flush()
    os._exit(128 + signum)


class SessionHostApp(App):
    """Behaviour every host of a session screen needs, kiosk or standalone.

    1. An unexpected exception goes to the session debug log with its full
       traceback, the user sees an error toast, and the app keeps running: a
       meeting recording must not end because of a bug. More than
       ERROR_BURST_LIMIT errors within ERROR_BURST_WINDOW_S means something fails
       on every redraw; then one alert replaces the toasts, and the recording
       still continues until the user stops it.
    2. SIGTERM and SIGHUP (window closed) end the session cleanly: the audio
       worker flushes, the context is saved, and the app exits. The final LLM
       digest is skipped because it can take minutes.
    """

    _error_times: deque[float]
    _storm_reported = False
    _storm_errors = 0
    _storm_logged_at = 0.0

    def get_driver_class(self) -> type[Driver]:
        driver_class = super().get_driver_class()
        if sys.platform == 'win32':
            # The console stream converts text for the console; raw os.write would not.
            return driver_class
        # Textual's writer thread dies on the first failed terminal write.
        return with_terminal_output(driver_class)

    def _handle_exception(self, error: Exception) -> None:
        # Textual wraps a worker exception in WorkerFailed without raising it,
        # so the wrapper has no traceback. Log the original one.
        cause = error.error if isinstance(error, WorkerFailed) else error
        if self._storm_reported:
            self._count_storm_error(cause)
            return
        log.critical('Unexpected error; the session continues', exc_info=cause)
        if not hasattr(self, '_error_times'):
            self._error_times = deque(maxlen=ERROR_BURST_LIMIT + 1)
        now = time.monotonic()
        self._error_times.append(now)
        storm = len(self._error_times) > ERROR_BURST_LIMIT and now - self._error_times[0] <= ERROR_BURST_WINDOW_S
        if not storm:
            self.notify(
                f'Unexpected error: {type(cause).__name__}: {cause} (see debug.log). The session continues.',
                severity='error',
                timeout=15,
            )
            return
        self._storm_reported = True
        self._storm_logged_at = now
        log.critical(
            'Errors repeat %d times in %.0fs; the recording continues',
            len(self._error_times),
            now - self._error_times[0],
        )
        self.notify(
            'Errors keep repeating (see debug.log). The audio recording continues; press s to stop and save.',
            severity='error',
            timeout=3600,
        )

    def _count_storm_error(self, cause: BaseException) -> None:
        # A traceback per error would fill the disk in a long meeting, and a full
        # disk pauses the recording. Log a count instead.
        self._storm_errors += 1
        now = time.monotonic()
        if now - self._storm_logged_at < STORM_LOG_INTERVAL_S:
            return
        log.critical(
            '%d more errors in %.0fs; the last: %s: %s',
            self._storm_errors,
            now - self._storm_logged_at,
            type(cause).__name__,
            cause,
        )
        self._storm_errors = 0
        self._storm_logged_at = now

    _previous_handlers: dict[int, object]
    _watchdog_armed = False

    def on_mount(self) -> None:
        # signal.signal, not loop.add_signal_handler: an asyncio handler runs only
        # when the loop turns, so a hung loop would ignore the closed window and
        # keep the tap and whisper processes alive. This handler runs on the main
        # thread even while it waits on a lock, and arms the exit watchdog first.
        self._previous_handlers = {}
        for signum in shutdown_signals():
            try:
                self._previous_handlers[signum] = signal.signal(signum, self._on_os_signal)
            except (ValueError, OSError, RuntimeError) as e:
                # Only the main thread can install handlers (tests, embedded hosts).
                log.debug('No handler for %s: %s', signum, e)

    def on_unmount(self) -> None:
        for signum, previous in getattr(self, '_previous_handlers', {}).items():
            try:
                signal.signal(signum, previous)
            except (ValueError, OSError, RuntimeError, TypeError) as e:
                log.debug('Could not restore the handler for %s: %s', signum, e)
        self._previous_handlers = {}

    def _on_os_signal(self, signum: int, _frame) -> None:
        self._arm_watchdog_once(signum)
        loop = self._loop
        if loop is None:
            log.error('No event loop after %s; the watchdog will end the process', signal.Signals(signum).name)
            return
        try:
            loop.call_soon_threadsafe(self.handle_shutdown_signal, signum)
        except RuntimeError as e:
            log.error('Event loop closed after %s: %s', signal.Signals(signum).name, e)

    def _arm_watchdog_once(self, signum: int) -> None:
        if self._watchdog_armed:
            return
        self._watchdog_armed = True
        arm_exit_watchdog(signum)

    def handle_shutdown_signal(self, signum: int) -> None:
        log.warning('Received %s; ending the session without a final digest', signal.Signals(signum).name)
        self._arm_watchdog_once(signum)
        self.end_sessions()

    def end_sessions(self) -> None:
        """Stop every recording (audio flush, context save), then exit."""
        waiting = False
        for screen in self.screen_stack:
            stop = getattr(screen, 'stop_for_shutdown', None)
            if callable(stop) and stop():
                waiting = True
        if waiting:
            self.set_timer(SHUTDOWN_GRACE_S, self.exit)
        else:
            self.exit()


class _HostApp(SessionHostApp):
    """Minimal one-screen host for a standalone Screen."""

    def __init__(self, screen: Screen) -> None:
        super().__init__()
        self._hosted_screen = screen

    def on_mount(self) -> None:
        self.push_screen(self._hosted_screen, callback=self.exit)


class ShellScreen(Screen[T]):
    """Screen that can run on its own for a one-shot command or a test."""

    _host: _HostApp | None = None

    @property
    def return_value(self) -> T | None:
        """Result passed to dismiss(), mirrored from the standalone host App."""
        return self._host.return_value if self._host is not None else None

    def run(self) -> T | None:
        self._host = _HostApp(self)
        self._host.run()
        return self._host.return_value

    def run_test(self, *args, **kwargs):
        self._host = _HostApp(self)
        return self._host.run_test(*args, **kwargs)

    async def _dispatch_message(self, message):
        # Textual ends a node's message loop after a handler raises. For a
        # recording screen that would freeze transcript and audio updates, so
        # report the error and keep the loop alive.
        try:
            return await super()._dispatch_message(message)
        except Exception as error:
            self.app._handle_exception(error)  # noqa: SLF001 -- route to the host's logging safety net
            return None
