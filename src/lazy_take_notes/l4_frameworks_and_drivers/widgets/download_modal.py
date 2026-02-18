"""Download modal — blocks interaction while model downloads or loads."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Static


class DownloadModal(ModalScreen[None]):
    """Modal overlay shown during model download and initial load."""

    DEFAULT_CSS = """
    DownloadModal {
        align: center middle;
    }

    DownloadModal > Vertical {
        width: auto;
        min-width: 40;
        max-width: 60;
        height: auto;
        background: $surface;
        border: thick $primary;
        padding: 1 3;
    }

    DownloadModal > Vertical > #dl-status {
        text-align: center;
        text-style: bold;
        margin-bottom: 1;
    }

    DownloadModal > Vertical > #dl-detail {
        text-align: center;
        color: $text-muted;
    }
    """

    def __init__(self, model_name: str = '', **kwargs) -> None:
        super().__init__(**kwargs)
        self.model_name = model_name
        self.percent = -1
        self.phase = 'downloading'

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static('Downloading model…', id='dl-status')
            yield Static(self.model_name, id='dl-detail')

    def update_progress(self, percent: int) -> None:
        self.percent = percent
        self.query_one('#dl-status', Static).update(f'Downloading model… {percent}%')

    def switch_to_loading(self) -> None:
        self.phase = 'loading'
        self.query_one('#dl-status', Static).update('Loading model…')
        self.query_one('#dl-detail', Static).update(self.model_name)
