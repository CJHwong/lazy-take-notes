"""Query modal — dismissible modal screen for quick action results."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Markdown, Static


class QueryModal(ModalScreen[None]):
    """Modal screen that displays quick-action query results. Escape to dismiss."""

    DEFAULT_CSS = """
    QueryModal {
        align: center middle;
    }

    QueryModal > VerticalScroll {
        width: 80%;
        max-width: 100;
        height: auto;
        max-height: 80%;
        background: $surface;
        border: thick $primary;
        padding: 1 2;
    }

    QueryModal > VerticalScroll > #query-title {
        text-style: bold;
        margin-bottom: 1;
    }

    QueryModal > VerticalScroll > #query-body {
        height: auto;
    }

    QueryModal > VerticalScroll > #query-hint {
        dock: bottom;
        text-align: center;
        color: $text-muted;
        margin-top: 1;
    }
    """

    BINDINGS = [('escape', 'dismiss', 'Close')]

    def __init__(self, title: str, body: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self._title = title
        self._body = body

    def compose(self) -> ComposeResult:
        with VerticalScroll():
            yield Static(self._title, id='query-title')
            yield Markdown(self._body, id='query-body')
            yield Static('Press Escape to close', id='query-hint')
