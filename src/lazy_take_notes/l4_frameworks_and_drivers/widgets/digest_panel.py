"""Digest panel — renders LLM markdown output."""

from __future__ import annotations

import pyperclip
from textual.binding import Binding
from textual.widgets import Markdown, TextArea


class DigestPanel(Markdown):
    """Scrollable digest display that renders LLM markdown directly."""

    can_focus = True

    DEFAULT_CSS = """
    DigestPanel {
        height: 1fr;
        overflow-y: auto;
        border: solid $secondary;
        scrollbar-size: 1 1;
    }
    DigestPanel:focus {
        border: solid $accent;
    }
    """

    BINDINGS = [
        Binding('c', 'copy_content', 'Copy', show=False),
        Binding('up', 'scroll_up', 'Scroll up', show=False),
        Binding('down', 'scroll_down', 'Scroll down', show=False),
        Binding('pageup', 'page_up', 'Page up', show=False),
        Binding('pagedown', 'page_down', 'Page down', show=False),
        Binding('home', 'scroll_home', 'Home', show=False),
        Binding('end', 'scroll_end', 'End', show=False),
    ]

    def __init__(self, title: str = 'Digest', **kwargs) -> None:
        super().__init__('', **kwargs)
        self.border_title = title
        self._current_markdown: str = ''

    def update_digest(self, markdown: str) -> None:
        """Replace content with the latest markdown from the LLM."""
        self._current_markdown = markdown
        self.update(markdown)

    def _session_context_suffix(self) -> str:
        """Return session context text to append when copying, or empty string."""
        try:
            ctx = self.app.query_one('#context-input', TextArea)
            if ctx.read_only and ctx.text.strip():
                return f'\n\n---\n\n**Session Context**\n\n{ctx.text.strip()}'
        except Exception:  # noqa: S110 -- widget may not exist in all app modes
            pass
        return ''

    def action_copy_content(self) -> None:
        """Copy digest markdown to system clipboard."""
        if not self._current_markdown:
            self.app.notify('No digest to copy', severity='warning', timeout=2)
            return
        pyperclip.copy(self._current_markdown + self._session_context_suffix())
        self.app.notify('Digest copied', timeout=2)
