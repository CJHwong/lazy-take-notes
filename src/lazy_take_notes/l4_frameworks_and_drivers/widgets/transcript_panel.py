"""Transcript panel — scrolling RichLog of transcribed speech segments."""

from __future__ import annotations

from collections.abc import Callable

import pyperclip
from textual.binding import Binding
from textual.widgets import RichLog, TextArea

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

    def render_segments(
        self,
        segments: list[TranscriptSegment],
        label_for: Callable[[TranscriptSegment], str | None] | None = None,
    ) -> None:
        """Rewrite the whole log, prefixing each line with a speaker label.

        RichLog is append-only, so a diarization update redraws every line. Keep
        auto-scroll only when the reader was already at the bottom.
        """

        def line_for(seg: TranscriptSegment) -> tuple[str, str]:
            timestamp = format_wall_time(seg.wall_start)
            label = label_for(seg) if label_for is not None else None
            prefix = f'{label}: ' if label else ''
            return f'[{timestamp}] {prefix}{seg.text}', f'[dim]\\[{timestamp}][/dim] {prefix}{seg.text}'

        at_bottom = self.scroll_offset.y >= max(0, self.max_scroll_y - 1)
        self.clear()
        self._all_text = []
        for seg in segments:
            plain, markup = line_for(seg)
            self._all_text.append(plain)
            self.write(markup)
        if at_bottom:
            self.scroll_end(animate=False)

    def _session_context_suffix(self) -> str:
        """Return session context text to append when copying, or empty string."""
        try:
            ctx = self.app.query_one('#context-input', TextArea)
            if ctx.read_only and ctx.text.strip():
                return f'\n\n---\nSession Context:\n{ctx.text.strip()}'
        except Exception:  # noqa: S110 -- widget may not exist in all app modes
            pass
        return ''

    def action_copy_content(self) -> None:
        """Copy full transcript text to system clipboard."""
        if not self._all_text:
            self.app.notify('No transcript to copy', severity='warning', timeout=2)
            return
        pyperclip.copy('\n'.join(self._all_text) + self._session_context_suffix())
        self.app.notify('Transcript copied', timeout=2)
