"""ViewApp — read-only TUI for browsing saved sessions."""

from __future__ import annotations

import subprocess  # noqa: S404 -- used for fire-and-forget OS file manager launch
import sys
from pathlib import Path

from textual.app import App as TextualApp
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Static

from lazy_take_notes.l4_frameworks_and_drivers.widgets.digest_panel import DigestPanel
from lazy_take_notes.l4_frameworks_and_drivers.widgets.status_bar import StatusBar
from lazy_take_notes.l4_frameworks_and_drivers.widgets.transcript_panel import TranscriptPanel

# ViewApp does NOT inherit from BaseApp — it doesn't need a controller,
# template, digest workers, or quick actions. It's a standalone read-only shell.
CSS_PATH = 'app.tcss'


class ViewApp(TextualApp):
    """Read-only TUI for browsing a saved session's transcript and digest."""

    CSS_PATH = 'app.tcss'

    BINDINGS = [
        Binding('q', 'quit_app', 'Quit', priority=True),
        Binding('o', 'open_session_dir', 'Open', show=False),
        Binding('tab', 'focus_next', 'Switch Panel', show=False),
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
        bar = self.query_one('#status-bar', StatusBar)
        bar.mode_label = 'View'
        bar.keybinding_hints = r'\[c] copy  \[o] open  \[Tab] switch  \[q] back'

        # Load transcript — write raw lines directly; the saved file already
        # contains timestamps so we must NOT go through append_segments()
        # which would prepend a second [00:00:00] timestamp.
        transcript_path = self._session_dir / 'transcript_raw.txt'
        if transcript_path.exists():
            text = transcript_path.read_text(encoding='utf-8').strip()
            if text:
                panel = self.query_one('#transcript-panel', TranscriptPanel)
                for line in text.splitlines():
                    if line.strip():
                        panel._all_text.append(line)
                        panel.write(line)

        # Load digest
        digest_path = self._session_dir / 'digest.md'
        if digest_path.exists():
            digest_text = digest_path.read_text(encoding='utf-8').strip()
            if digest_text:
                panel = self.query_one('#digest-panel', DigestPanel)
                panel.update_digest(digest_text)

        bar.stopped = True

    def action_open_session_dir(self) -> None:
        if sys.platform == 'darwin':
            opener = 'open'
        elif sys.platform == 'win32':
            opener = 'explorer'
        else:
            opener = 'xdg-open'
        subprocess.Popen([opener, str(self._session_dir)])  # noqa: S603 -- fixed arg list, not shell=True

    def action_quit_app(self) -> None:
        self.exit()
