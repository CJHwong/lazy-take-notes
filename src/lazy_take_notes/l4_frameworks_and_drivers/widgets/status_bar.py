"""Status bar — bottom bar showing elapsed time, digest buffer progress, and keybinding hints."""

from __future__ import annotations

import math
import time
from collections import deque

from rich.cells import cell_len
from textual.reactive import reactive
from textual.widgets import Static

_WAVE_CHARS = '▁▂▃▄▅▆▇█'

# dB-scaled level meter: maps -60 dB (silence) to -11 dB (loud) across 8 bars.
# Works for both quiet mic input (~0.005–0.06 RMS) and loud system audio (~0.1–0.5 RMS).
_DB_FLOOR = -60.0
_DB_RANGE = 49.0  # -60 to -11


def _rms_to_char(rms: float) -> str:
    if rms < 1e-7:
        return _WAVE_CHARS[0]
    db = 20.0 * math.log10(rms)
    idx = int((db - _DB_FLOOR) / _DB_RANGE * 7)
    return _WAVE_CHARS[min(max(idx, 0), 7)]


class StatusBar(Static):
    """Bottom status bar with recording state, digest buffer progress, and keybinding hints."""

    DEFAULT_CSS = """
    StatusBar {
        dock: bottom;
        height: auto;
        background: $surface;
        color: $text;
        padding: 0 1;
        overflow: hidden hidden;
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
    last_digest_time: reactive[float] = reactive(0.0)
    mode_label: reactive[str] = reactive('')
    keybinding_hints: reactive[str] = reactive('')
    quick_action_hints: reactive[str] = reactive('')
    _start_time: float = 0.0
    _frozen_elapsed: float | None = None
    _pause_start: float | None = None
    _paused_total: float = 0.0

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._start_time = 0.0
        self._recording_started = False
        self._level_history: deque[float] = deque([0.0] * 6, maxlen=6)

    def watch_recording(self, value: bool) -> None:
        """Start the elapsed timer on the first recording=True transition."""
        if value and not self._recording_started:
            self._start_time = time.monotonic()
            self._recording_started = True

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
            if not self._recording_started:
                self._frozen_elapsed = 0.0
                return
            now = time.monotonic()
            paused = self._paused_total
            if self._pause_start is not None:
                paused += now - self._pause_start
            self._frozen_elapsed = now - self._start_time - paused

    def watch_audio_level(self, value: float) -> None:
        """Push new level into rolling history and re-render."""
        self._level_history.append(value)
        self.refresh()

    def _recording_elapsed(self, now: float) -> float:
        """Elapsed seconds excluding any paused periods."""
        paused = self._paused_total
        if self._pause_start is not None:
            paused += now - self._pause_start
        return now - self._start_time - paused

    def _format_elapsed(self, now: float) -> str:
        if not self._recording_started and self._frozen_elapsed is None:
            return '00:00:00'
        elapsed = self._frozen_elapsed if self._frozen_elapsed is not None else self._recording_elapsed(now)
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        secs = int(elapsed % 60)
        return f'{hours:02d}:{minutes:02d}:{secs:02d}'

    def render(self) -> str:
        now = time.monotonic()

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

        left_parts = []
        if self.mode_label:
            left_parts.append(self.mode_label)
        left_parts.extend(
            [
                status_icon,
                f'buf {self.buf_count}/{self.buf_max}',
                self._format_elapsed(now),
            ]
        )
        if self.last_digest_time > 0:
            since = now - self.last_digest_time
            if since < 60:
                left_parts.append(f'last {int(since)}s ago')
            else:
                left_parts.append(f'last {int(since / 60)}m ago')
        if self.recording:
            wave = ''.join(_rms_to_char(v) for v in self._level_history)
            left_parts.append(wave)
        if self.activity:
            left_parts.append(f'⟳ {self.activity}')
        left = ' │ '.join(left_parts)

        content_width = (self.size.width or 80) - 2

        hints = self.keybinding_hints
        if hints:
            hints_width = cell_len(hints.replace(r'\[', '['))
            gap = content_width - cell_len(left) - hints_width
            if gap >= 2:
                left = left + ' ' * gap + hints

        if self.quick_action_hints:
            qa = self.quick_action_hints
            qa_width = cell_len(qa.replace(r'\[', '['))
            qa_gap = content_width - qa_width
            qa_line = ' ' * qa_gap + qa if qa_gap >= 0 else qa
            return qa_line + '\n' + left
        return left
