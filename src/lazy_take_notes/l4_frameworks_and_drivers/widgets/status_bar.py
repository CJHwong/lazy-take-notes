"""Status bar — bottom bar showing elapsed time, digest buffer progress, and keybinding hints."""

from __future__ import annotations

import time
from collections import deque

from textual.reactive import reactive
from textual.widgets import Static

_WAVE_CHARS = '▁▂▃▄▅▆▇█'


def _rms_to_char(rms: float) -> str:
    idx = min(int(rms * 50), 7)  # saturates at ~0.14 RMS (typical speech peak)
    return _WAVE_CHARS[idx]


class StatusBar(Static):
    """Bottom status bar with recording state, digest buffer progress, and keybinding hints."""

    DEFAULT_CSS = """
    StatusBar {
        dock: bottom;
        height: 1;
        background: $surface;
        color: $text;
        padding: 0 1;
    }
    """

    recording: reactive[bool] = reactive(False)
    paused: reactive[bool] = reactive(False)
    stopped: reactive[bool] = reactive(False)
    audio_status: reactive[str] = reactive('')
    activity: reactive[str] = reactive('')
    download_percent: reactive[int] = reactive(-1)
    download_model: reactive[str] = reactive('')
    buf_count: reactive[int] = reactive(0)
    buf_max: reactive[int] = reactive(15)
    audio_level: reactive[float] = reactive(0.0)
    keybinding_hints: reactive[str] = reactive('')
    _start_time: float = 0.0
    _frozen_elapsed: float | None = None
    _pause_start: float | None = None
    _paused_total: float = 0.0

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._start_time = time.monotonic()
        self._level_history: deque[float] = deque([0.0] * 6, maxlen=6)

    def watch_download_percent(self, value: int) -> None:
        """Re-render immediately when download progress changes."""
        self.refresh()

    def watch_paused(self, value: bool) -> None:
        """Track pause start/end to exclude paused time from the elapsed timer."""
        if value:
            self._pause_start = time.monotonic()
        elif self._pause_start is not None:
            self._paused_total += time.monotonic() - self._pause_start
            self._pause_start = None

    def watch_stopped(self, value: bool) -> None:
        """Freeze the elapsed timer (recording time only) when recording stops."""
        if value and self._frozen_elapsed is None:
            paused = self._paused_total
            if self._pause_start is not None:
                paused += time.monotonic() - self._pause_start
            self._frozen_elapsed = time.monotonic() - self._start_time - paused

    def watch_audio_level(self, value: float) -> None:
        """Push new level into rolling history and re-render."""
        self._level_history.append(value)
        self.refresh()

    def _recording_elapsed(self) -> float:
        """Elapsed seconds excluding any paused periods."""
        paused = self._paused_total
        if self._pause_start is not None:
            paused += time.monotonic() - self._pause_start
        return time.monotonic() - self._start_time - paused

    def _format_elapsed(self) -> str:
        if self._frozen_elapsed is not None:
            elapsed = self._frozen_elapsed
        else:
            elapsed = self._recording_elapsed()
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
        left_parts = [
            status_icon,
            f'buf {self.buf_count}/{self.buf_max}',
            elapsed,
        ]
        if self.recording:
            wave = ''.join(_rms_to_char(v) for v in self._level_history)
            left_parts.append(wave)
        if self.activity:
            left_parts.append(f'⟳ {self.activity}')
        left = ' │ '.join(left_parts)

        right = self.keybinding_hints
        if not right:
            return left

        content_width = (self.size.width or 80) - 2
        gap = content_width - len(left) - len(right)
        padding = ' ' * max(gap, 2)
        return left + padding + right
