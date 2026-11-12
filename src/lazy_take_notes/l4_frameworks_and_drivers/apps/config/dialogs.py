"""Modal dialogs shared by the settings screen."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Static


def _megabytes(size: int) -> str:
    return f'{size / (1024 * 1024):.0f} MB'


class _DiscardChangesModal(ModalScreen[bool]):
    """Ask before leaving Settings with unsaved edits. True means discard."""

    DEFAULT_CSS = """
    _DiscardChangesModal {
        align: center middle;
    }

    _DiscardChangesModal > Vertical {
        width: 54;
        height: auto;
        background: $surface;
        border: thick $warning;
        padding: 1 2;
    }

    _DiscardChangesModal #discard-title {
        text-style: bold;
        margin-bottom: 1;
    }

    _DiscardChangesModal #discard-body {
        margin-bottom: 1;
    }

    _DiscardChangesModal #discard-actions {
        height: auto;
        align: right middle;
    }

    _DiscardChangesModal Button {
        margin-left: 2;
    }
    """

    BINDINGS = [
        Binding('escape', 'keep_editing', 'Keep editing', priority=True),
        Binding('enter', 'discard', 'Discard', priority=True),
    ]

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static('Unsaved changes', id='discard-title')
            yield Static('Your changes to Settings have not been saved. Leave anyway?', id='discard-body')
            with Horizontal(id='discard-actions'):
                yield Button('Keep editing', id='discard-cancel')
                yield Button('Discard and exit', id='discard-confirm', variant='error')

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == 'discard-confirm')

    def action_keep_editing(self) -> None:
        self.dismiss(False)

    def action_discard(self) -> None:
        self.dismiss(True)


class RemoveModelModal(_DiscardChangesModal):
    """Confirm model removal and state the freed bytes.

    It shares the discard dialog layout and ids.
    """

    def __init__(self, label: str, size: int) -> None:
        super().__init__()
        self._asset_label = label
        self._freed_bytes = size

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static(f'Remove {self._asset_label}?', id='discard-title')
            yield Static(
                f'This frees {_megabytes(self._freed_bytes)}. It downloads again if you need it.',
                id='discard-body',
            )
            with Horizontal(id='discard-actions'):
                yield Button('Cancel', id='discard-cancel')
                yield Button('Remove', id='discard-confirm', variant='error')
