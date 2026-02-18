"""Status bar — bottom bar showing elapsed time, segment count, digest status."""

from __future__ import annotations

import time

from textual.reactive import reactive
from textual.widgets import Static


class StatusBar(Static):
    """Bottom status bar with recording state, counters, and keybinding hints."""

    DEFAULT_CSS = """
    StatusBar {
        dock: bottom;
        height: 1;
        background: $surface;
        color: $text;
        padding: 0 1;
    }
    """

    segment_count: reactive[int] = reactive(0)
    digest_count: reactive[int] = reactive(0)
    recording: reactive[bool] = reactive(False)
    paused: reactive[bool] = reactive(False)
    stopped: reactive[bool] = reactive(False)
    audio_status: reactive[str] = reactive('')
    activity: reactive[str] = reactive('')
    download_percent: reactive[int] = reactive(-1)
    download_model: reactive[str] = reactive('')
    _start_time: float = 0.0
    _frozen_elapsed: float | None = None

    def __init__(self, keybinding_hints: str = '', **kwargs) -> None:
        super().__init__(**kwargs)
        self._keybinding_hints = keybinding_hints
        self._start_time = time.monotonic()

    def watch_download_percent(self, value: int) -> None:
        """Re-render immediately when download progress changes."""
        self.refresh()

    def watch_stopped(self, value: bool) -> None:
        """Freeze the elapsed timer when recording stops."""
        if value and self._frozen_elapsed is None:
            self._frozen_elapsed = time.monotonic() - self._start_time

    def _format_elapsed(self) -> str:
        if self._frozen_elapsed is not None:
            elapsed = self._frozen_elapsed
        else:
            elapsed = time.monotonic() - self._start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        secs = int(elapsed % 60)
        return f'{hours:02d}:{minutes:02d}:{secs:02d}'

    def render(self) -> str:
        if self.stopped:
            status_icon = '■ Stopped'
        elif self.paused:
            status_icon = '❚❚ Paused'
        elif self.recording:
            status_icon = '● Rec'
        elif self.download_percent >= 0:
            status_icon = f'⟳ Downloading {self.download_model}… {self.download_percent}%'
        elif self.audio_status == 'loading_model':
            status_icon = '⟳ Loading model'
        elif self.audio_status == 'error':
            status_icon = '✗ Error'
        else:
            status_icon = '○ Idle'

        elapsed = self._format_elapsed()
        parts = [
            status_icon,
            f'Seg: {self.segment_count}',
            f'Digest #{self.digest_count}',
            f'{elapsed}',
        ]
        if self.activity:
            parts.append(f'⟳ {self.activity}')
        if self._keybinding_hints:
            parts.append(self._keybinding_hints)

        return ' │ '.join(parts)
