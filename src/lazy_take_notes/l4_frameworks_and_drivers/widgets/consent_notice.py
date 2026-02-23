"""Recording consent notice — non-blocking overlay reminding the user to inform attendees."""

from __future__ import annotations

from collections.abc import Callable

from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Markdown, Static

NOTICE_BODY = """\
This session will **record audio** and generate a transcript.

- **Inform all participants** that recording is active.
- Recording without consent may violate local laws.
- You are responsible for compliance.
"""


class ConsentNotice(ModalScreen[None]):
    """One-time, non-blocking notice shown when a recording session starts."""

    DEFAULT_CSS = """
    ConsentNotice {
        align: center middle;
    }

    ConsentNotice > VerticalScroll {
        width: 60%;
        max-width: 72;
        height: auto;
        max-height: 60%;
        background: $surface;
        border: thick $warning;
        padding: 1 2;
    }

    ConsentNotice > VerticalScroll > #notice-title {
        text-style: bold;
        color: $warning;
        margin-bottom: 1;
    }

    ConsentNotice > VerticalScroll > #notice-body {
        height: auto;
    }

    ConsentNotice > VerticalScroll > #notice-hint {
        dock: bottom;
        text-align: center;
        color: $text-muted;
        margin-top: 1;
    }
    """

    BINDINGS = [
        ('escape', 'dismiss', 'Close'),
    ]

    def __init__(
        self,
        *args,
        on_suppress: Callable[[], None] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._on_suppress = on_suppress

    def compose(self) -> ComposeResult:
        with VerticalScroll():
            yield Static('Recording Notice', id='notice-title')
            yield Markdown(NOTICE_BODY, id='notice-body')
            yield Static('Press any key to dismiss \u00b7 \\[n] never show again', id='notice-hint')

    def on_key(self, event) -> None:  # noqa: ANN001 -- Textual Key event; type not needed
        if event.key == 'n' and self._on_suppress is not None:
            self._on_suppress()
        self.dismiss()
