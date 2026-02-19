"""Transcript panel — scrolling RichLog of transcribed speech segments."""

from __future__ import annotations

import pyperclip
from textual.binding import Binding
from textual.widgets import RichLog

from lazy_take_notes.l1_entities.transcript import TranscriptSegment, format_wall_time


class TranscriptPanel(RichLog):
    """Auto-scrolling transcript display using RichLog."""

    DEFAULT_CSS = """
    TranscriptPanel {
        border: solid $primary;
        scrollbar-size: 1 1;
    }
    TranscriptPanel:focus {
        border: solid $accent;
    }
    """

    BINDINGS = [Binding('c', 'copy_content', 'Copy', show=False)]

    def __init__(self, title: str = 'Transcript', **kwargs) -> None:
        super().__init__(highlight=True, markup=True, wrap=True, auto_scroll=True, **kwargs)
        self.border_title = title
        self._all_text: list[str] = []

    def append_segments(self, segments: list[TranscriptSegment]) -> None:
        """Append new transcript segments to the log."""
        for seg in segments:
            timestamp = format_wall_time(seg.wall_start)
            self._all_text.append(f'[{timestamp}] {seg.text}')
            self.write(f'[dim]\\[{timestamp}][/dim] {seg.text}')

    def action_copy_content(self) -> None:
        """Copy full transcript text to system clipboard."""
        if not self._all_text:
            self.app.notify('No transcript to copy', severity='warning', timeout=2)
            return
        pyperclip.copy('\n'.join(self._all_text))
        self.app.notify('Transcript copied', timeout=2)
