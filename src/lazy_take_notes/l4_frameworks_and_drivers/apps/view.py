"""ViewApp — read-only TUI for browsing saved sessions."""

from __future__ import annotations

import logging
import subprocess  # noqa: S404 -- used for fire-and-forget OS file manager launch
import sys
from pathlib import Path

from rich.markup import escape
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Static

from lazy_take_notes.l1_entities.session_files import NOTES, TRANSCRIPT
from lazy_take_notes.l4_frameworks_and_drivers.config import load_theme
from lazy_take_notes.l4_frameworks_and_drivers.screen_base import ShellScreen
from lazy_take_notes.l4_frameworks_and_drivers.widgets.digest_panel import DigestPanel
from lazy_take_notes.l4_frameworks_and_drivers.widgets.status_bar import StatusBar
from lazy_take_notes.l4_frameworks_and_drivers.widgets.transcript_panel import TranscriptPanel

log = logging.getLogger('ltn.view')

# ViewApp does NOT inherit from BaseApp — it doesn't need a controller,
# template, digest workers, or quick actions. It's a standalone read-only shell.
CSS_PATH = 'app.tcss'


class ViewApp(ShellScreen):
    """Read-only TUI for browsing a saved session's transcript and digest."""

    CSS_PATH = 'app.tcss'

    BINDINGS = [
        Binding('q', 'quit_app', 'Quit', priority=True),
        Binding('o', 'open_session_dir', 'Open', show=False),
        Binding('tab', 'app.focus_next', 'Switch Panel', show=False),
    ]

    def __init__(self, session_dir: Path, **kwargs) -> None:
        super().__init__(**kwargs)
        self._session_dir = session_dir

    def compose(self) -> ComposeResult:
        label = self._session_dir.name
        yield Static(f'  lazy-take-notes | View: {label}', id='header')
        with Horizontal(id='main-panels'):
            yield TranscriptPanel(id='transcript-panel')
            with Vertical(id='digest-col'):
                yield DigestPanel(id='digest-panel')
        yield StatusBar(id='status-bar')

    def on_mount(self) -> None:
        self.app.theme = load_theme()
        bar = self.query_one('#status-bar', StatusBar)
        bar.mode_label = 'View'
        bar.keybinding_hints = r'\[c] copy  \[o] open  \[Tab] switch  \[q] back'

        # Load transcript — write raw lines directly; the saved file already
        # contains timestamps so we must NOT go through append_segments()
        # which would prepend a second [00:00:00] timestamp.
        text = self._read_saved(TRANSCRIPT.resolve(self._session_dir))
        transcript_panel = self.query_one('#transcript-panel', TranscriptPanel)
        for line in text.splitlines():
            if line.strip():
                transcript_panel._all_text.append(line)
                # Saved lines hold raw whisper text, which can look like markup.
                transcript_panel.write(escape(line))

        if TRANSCRIPT.resolve(self._session_dir) is None and next(self._session_dir.glob('recording*.wav'), None):
            # A failed transcription keeps only the audio.
            transcript_panel.write('No transcript. The audio is saved in this folder: press o to open it.')

        digest_text = self._read_saved(NOTES.resolve(self._session_dir))
        if digest_text:
            self.query_one('#digest-panel', DigestPanel).update_digest(digest_text)

        bar.stopped = True

    def _read_saved(self, path: Path | None) -> str:
        """Return the stripped file text, or '' after a toast when it cannot be read."""
        if path is None:
            return ''
        try:
            return path.read_text(encoding='utf-8').strip()
        except (OSError, UnicodeDecodeError) as exc:
            log.error('Cannot read %s: %s', path, exc, exc_info=True)
            self.notify(f'Cannot read {path.name}: {exc}', severity='error', markup=False, timeout=8)
            return ''

    def action_open_session_dir(self) -> None:
        if sys.platform == 'darwin':
            opener = 'open'
        elif sys.platform == 'win32':
            opener = 'explorer'
        else:
            opener = 'xdg-open'
        try:
            subprocess.Popen([opener, str(self._session_dir)])  # noqa: S603 -- fixed arg list, not shell=True
        except OSError as exc:
            log.error('Cannot open %s with %s: %s', self._session_dir, opener, exc, exc_info=True)
            self.notify(f'Cannot open the session folder: {exc}', severity='error', markup=False)

    def action_quit_app(self) -> None:
        self.dismiss()
