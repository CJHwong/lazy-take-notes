"""Welcome picker — TUI to choose Record / Transcribe / View at app startup."""

from __future__ import annotations

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import ListItem, ListView, Static

_MODES = [
    ('record', 'Record', 'Live transcription with AI digest'),
    ('transcribe', 'Transcribe', 'Transcribe an existing audio file'),
    ('view', 'View', 'Browse previously saved sessions'),
    ('create-template', 'Create Template', 'Build a custom template with AI'),
    ('config', 'Settings', 'Edit app configuration'),
]

# Color palette
_P = '#E8A820'  # pig body (golden yellow)
_T = '#4A9E9E'  # teal headphones
_D = '#2A1818'  # dark outline
_S = '#C47A10'  # snout (amber-brown)
_B = None  # terminal background (transparent)

_PIXEL_ROWS: list[list[str | None]] = [
    # rows 0-1: headphone arch
    [_B, _B, _T, _T, _T, _T, _T, _T, _T, _T, _T, _T, _T, _T, _T, _T, _T, _T, _B, _B],
    [_B, _T, _T, _B, _D, _D, _D, _D, _D, _D, _D, _D, _D, _D, _D, _D, _B, _T, _T, _B],
    # rows 2-3: ear cups (extend down) + face upper (eyes)
    [_T, _T, _B, _D, _P, _P, _P, _P, _P, _P, _P, _P, _P, _P, _P, _P, _D, _B, _T, _T],
    [_T, _T, _B, _D, _P, _D, _D, _P, _P, _P, _P, _P, _P, _D, _D, _P, _D, _B, _T, _T],
    # rows 4-5: ear cups (extend down) + cheeks + snout top
    [_T, _T, _B, _D, _P, _P, _P, _P, _P, _P, _P, _P, _P, _P, _P, _P, _D, _B, _T, _T],
    [_T, _T, _B, _D, _P, _P, _P, _S, _S, _S, _S, _S, _S, _P, _P, _P, _D, _B, _T, _T],
    # rows 6-7: ear cups taper off + nostrils
    [_B, _T, _B, _D, _P, _P, _P, _S, _D, _S, _S, _D, _S, _P, _P, _P, _D, _B, _T, _B],
    [_B, _B, _B, _D, _P, _P, _P, _S, _S, _S, _S, _S, _S, _P, _P, _P, _D, _B, _B, _B],
    # rows 8-9: lower face
    [_B, _B, _B, _D, _P, _P, _P, _P, _P, _P, _P, _P, _P, _P, _P, _P, _D, _B, _B, _B],
    [_B, _B, _B, _B, _D, _P, _P, _P, _P, _P, _P, _P, _P, _P, _P, _D, _B, _B, _B, _B],
    # rows 10-11: chin / base
    [_B, _B, _B, _B, _B, _D, _P, _P, _P, _P, _P, _P, _P, _P, _D, _B, _B, _B, _B, _B],
    [_B, _B, _B, _B, _B, _B, _D, _D, _D, _D, _D, _D, _D, _D, _B, _B, _B, _B, _B, _B],
]


def _cell(top: str | None, bot: str | None) -> str:
    if top is None and bot is None:
        return ' '
    if top is not None and bot is None:
        return f'[{top}]▀[/]'
    if top is None and bot is not None:
        return f'[{bot}]▄[/]'
    if top == bot:
        return f'[{top}]█[/]'
    return f'[{top} on {bot}]▀[/]'


def _render_banner() -> str:
    lines = []
    for i in range(0, 12, 2):
        row = _PIXEL_ROWS[i]
        row_below = _PIXEL_ROWS[i + 1]
        lines.append(''.join(_cell(row[c], row_below[c]) for c in range(20)))
    lines.append('[bold #E8A820]lazy-[/][bold #4A9E9E]take-notes[/]')
    return '\n'.join(lines)


_BANNER_TEXT = _render_banner()


class ModeItem(ListItem):
    """Selectable row representing a launch mode."""

    def __init__(self, mode: str, label: str, description: str) -> None:
        super().__init__()
        self.mode = mode
        self._label_text = f'[bold]{label}[/bold]  [dim]{description}[/dim]'

    def compose(self) -> ComposeResult:
        yield Static(self._label_text, markup=True)


class WelcomePicker(App[str | None]):
    CSS = """
    Screen {
        align: center middle;
    }
    #welcome-banner {
        text-align: center;
        width: 100%;
        height: auto;
        padding: 1 0 0 0;
    }
    #welcome-footer {
        dock: bottom;
        height: 1;
        background: $surface;
        color: $text-muted;
        text-align: center;
        padding: 0 1;
    }
    #welcome-list {
        border: solid $primary;
        margin: 1 2;
        height: auto;
        max-height: 8;
    }
    #welcome-list Static {
        height: 1;
        overflow: hidden hidden;
    }
    """

    BINDINGS = [
        Binding('escape', 'cancel', 'Cancel', priority=True),
        Binding('q', 'cancel', 'Cancel'),
        Binding('enter', 'select_mode', 'Select', priority=True),
    ]

    def compose(self) -> ComposeResult:
        yield Static(_BANNER_TEXT, id='welcome-banner', markup=True)
        yield ListView(
            *[ModeItem(mode, label, desc) for mode, label, desc in _MODES],
            id='welcome-list',
        )
        yield Static(
            '\\[Enter] Select  \\[↑/↓] Navigate  \\[Esc] Cancel',
            id='welcome-footer',
            markup=True,
        )

    def on_mount(self) -> None:
        self.query_one('#welcome-list', ListView).focus()

    def action_select_mode(self) -> None:
        lv = self.query_one('#welcome-list', ListView)
        item = lv.highlighted_child
        if isinstance(item, ModeItem):
            self.exit(item.mode)

    def action_cancel(self) -> None:
        self.exit(None)
