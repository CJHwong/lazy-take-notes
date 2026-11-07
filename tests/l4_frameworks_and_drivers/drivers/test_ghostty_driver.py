"""Tests for the Trolley/libghostty Textual driver."""

from __future__ import annotations

import pytest
from textual.driver import Driver
from textual.drivers.linux_driver import LinuxDriver

from lazy_take_notes.l4_frameworks_and_drivers.drivers.ghostty_driver import (
    DISABLE_KITTY_KEYBOARD,
    GhosttyDriver,
)


class RecordingDriver(GhosttyDriver):
    """GhosttyDriver with the write path captured instead of sent to a terminal."""

    def __init__(self) -> None:
        self.written: list[str] = []
        self.flushes = 0

    def write(self, data: str) -> None:
        self.written.append(data)

    def flush(self) -> None:
        self.flushes += 1


@pytest.fixture
def driver(monkeypatch: pytest.MonkeyPatch) -> RecordingDriver:
    """A driver whose parent start-up does nothing (no real terminal attached)."""
    monkeypatch.setattr(LinuxDriver, 'start_application_mode', lambda self: None)
    return RecordingDriver()


def test_driver_is_a_textual_driver() -> None:
    assert issubclass(GhosttyDriver, Driver)


def test_start_application_mode_disables_kitty_keyboard(driver: RecordingDriver) -> None:
    driver.start_application_mode()

    assert driver.written == [DISABLE_KITTY_KEYBOARD]


def test_start_application_mode_flushes(driver: RecordingDriver) -> None:
    driver.start_application_mode()

    assert driver.flushes == 1


def test_disable_sequence_is_legacy_keyboard_encoding() -> None:
    assert DISABLE_KITTY_KEYBOARD == '\x1b[<u'
