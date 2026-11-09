"""Base Screen for the single-app shell.

Every picker and mode UI is a Screen, so the kiosk runs one Textual App for the
whole session. Standalone commands and tests still need to run one Screen
without the shell, so ShellScreen provides run() and run_test() backed by a
minimal host App.
"""

from __future__ import annotations

from typing import TypeVar

from textual.app import App
from textual.screen import Screen

T = TypeVar('T')


class _HostApp(App):
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
