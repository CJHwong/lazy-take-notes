"""Textual driver for Trolley/libghostty runtime.

Trolley's libghostty runtime has issues with the Kitty keyboard protocol
(CSI > 1 u) — arrow keys stop working. This driver disables the protocol
immediately after Textual enables it, falling back to legacy escape sequences.

Activated via: TEXTUAL_DRIVER=lazy_take_notes.l4_frameworks_and_drivers.drivers.ghostty_driver:GhosttyDriver
"""

from __future__ import annotations

from textual.drivers.linux_driver import LinuxDriver


class GhosttyDriver(LinuxDriver):
    def start_application_mode(self) -> None:
        super().start_application_mode()
        # Textual unconditionally sends \x1b[>1u to enable the Kitty keyboard
        # protocol. Ghostty/libghostty supports it, but arrow key encoding
        # breaks under Trolley's runtime. Pop the protocol back to legacy mode.
        self.write('\x1b[<u')
        self.flush()
