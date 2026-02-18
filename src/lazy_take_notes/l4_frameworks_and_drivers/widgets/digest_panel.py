"""Digest panel — renders LLM markdown output."""

from __future__ import annotations

import pyperclip
from textual.binding import Binding
from textual.widgets import Markdown


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

    BINDINGS = [Binding('c', 'copy_content', 'Copy', show=False)]

    def __init__(self, title: str = 'Digest', **kwargs) -> None:
        super().__init__('', **kwargs)
        self.border_title = title
        self._current_markdown: str = ''

    def update_digest(self, markdown: str) -> None:
        """Replace content with the latest markdown from the LLM."""
        self._current_markdown = markdown
        self.update(markdown)

    def action_copy_content(self) -> None:
        """Copy digest markdown to system clipboard."""
        if not self._current_markdown:
            self.app.notify('No digest to copy', severity='warning', timeout=2)
            return
        pyperclip.copy(self._current_markdown)
        self.app.notify('Digest copied', timeout=2)
