"""Help modal — dismissible overlay showing template info and keybindings."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Markdown, Static


class HelpModal(ModalScreen[None]):
    """Modal screen that displays template info and keybinding reference."""

    DEFAULT_CSS = """
    HelpModal {
        align: center middle;
    }

    HelpModal > VerticalScroll {
        width: 60%;
        max-width: 80;
        height: auto;
        max-height: 80%;
        background: $surface;
        border: thick $primary;
        padding: 1 2;
    }

    HelpModal > VerticalScroll > #help-title {
        text-style: bold;
        margin-bottom: 1;
    }

    HelpModal > VerticalScroll > #help-body {
        height: auto;
    }

    HelpModal > VerticalScroll > #help-hint {
        dock: bottom;
        text-align: center;
        color: $text-muted;
        margin-top: 1;
    }
    """

    BINDINGS = [
        ('escape', 'dismiss', 'Close'),
        ('h', 'dismiss', 'Close'),
    ]

    def __init__(self, body_md: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self._body_md = body_md

    def compose(self) -> ComposeResult:
        with VerticalScroll():
            yield Static('Help', id='help-title')
            yield Markdown(self._body_md, id='help-body')
            yield Static('Press Escape or h to close', id='help-hint')
