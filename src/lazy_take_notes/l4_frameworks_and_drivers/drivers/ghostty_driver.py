"""Textual driver for the Trolley/libghostty runtime.

Trolley's libghostty runtime handles the Kitty keyboard protocol badly: after
Textual enables it, arrow keys stop arriving. Textual sends the enable
sequence unconditionally, so the only place to intercept is the driver.

Activated by the bundle through ``TEXTUAL_DRIVER``:

    TEXTUAL_DRIVER=lazy_take_notes.l4_frameworks_and_drivers.drivers.ghostty_driver:GhosttyDriver

Nothing else uses this class. Plain terminals keep Textual's default driver.
"""

from __future__ import annotations

from textual.drivers.linux_driver import LinuxDriver

#: Pop the terminal back to legacy keyboard encoding (CSI < u).
DISABLE_KITTY_KEYBOARD = '\x1b[<u'


class GhosttyDriver(LinuxDriver):
    """Textual driver that turns the Kitty keyboard protocol back off."""

    def start_application_mode(self) -> None:
        super().start_application_mode()
        self.write(DISABLE_KITTY_KEYBOARD)
        self.flush()
