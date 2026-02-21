"""Label modal — text input for renaming the session on the fly."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, Static


class LabelModal(ModalScreen[str | None]):
    """Modal that prompts for a session label. Enter → return text, Escape → None."""

    DEFAULT_CSS = """
    LabelModal {
        align: center middle;
    }

    LabelModal > Vertical {
        width: 60;
        height: auto;
        background: $surface;
        border: thick $primary;
        padding: 1 2;
    }

    LabelModal > Vertical > #label-title {
        text-style: bold;
        margin-bottom: 1;
    }

    LabelModal > Vertical > #label-hint {
        color: $text-muted;
        margin-top: 1;
        text-align: center;
    }
    """

    BINDINGS = [('escape', 'cancel', 'Cancel')]

    def __init__(self, current_label: str = '', **kwargs) -> None:
        super().__init__(**kwargs)
        self._current_label = current_label

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static('Session label', id='label-title')
            yield Input(
                value=self._current_label,
                placeholder='Session label...',
                id='label-input',
            )
            yield Static('Enter to confirm · Escape to cancel', id='label-hint')

    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        self.dismiss(text if text else None)

    def action_cancel(self) -> None:
        self.dismiss(None)
