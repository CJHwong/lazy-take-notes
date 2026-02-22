"""Session picker — small TUI to select a previously saved session for viewing."""

from __future__ import annotations

from pathlib import Path

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.events import Key
from textual.widgets import Input, ListItem, ListView, Markdown, Static


def discover_sessions(sessions_dir: Path) -> list[dict]:
    """Scan *sessions_dir* for session subdirs containing transcript_raw.txt.

    Returns a list of dicts sorted newest-first:
      {'dir': Path, 'name': str, 'has_digest': bool}
    """
    if not sessions_dir.exists():
        return []

    results = []
    for child in sorted(sessions_dir.iterdir(), reverse=True):
        if not child.is_dir():
            continue
        transcript = child / 'transcript_raw.txt'
        if not transcript.exists():
            continue
        results.append(
            {
                'dir': child,
                'name': child.name,
                'has_digest': (child / 'digest.md').exists(),
            }
        )
    return results


class SessionItem(ListItem):
    """Selectable row representing a saved session."""

    def __init__(self, session: dict) -> None:
        super().__init__()
        self.session_dir: Path = session['dir']
        digest_badge = '  [green]\u2713 digest[/green]' if session['has_digest'] else '  [dim]no digest[/dim]'
        self._label_text = f'{session["name"]}{digest_badge}'

    def compose(self) -> ComposeResult:
        yield Static(self._label_text, markup=True)


class _SessionListView(ListView):
    """ListView that pops focus back to the filter input when up is pressed on the first item."""

    def on_key(self, event: Key) -> None:
        if event.key != 'up':
            return
        first_selectable = next(
            (i for i, child in enumerate(self.children) if isinstance(child, SessionItem)),
            None,
        )
        if first_selectable is not None and self.index == first_selectable:
            self.app.query_one('#session-search', Input).focus()
            event.prevent_default()


class SessionPicker(App[Path | None]):
    CSS = """
    #session-header {
        dock: top;
        height: 1;
        background: $primary;
        color: $text;
        text-align: center;
        text-style: bold;
        padding: 0 1;
    }
    #session-footer {
        dock: bottom;
        height: 1;
        background: $surface;
        color: $text-muted;
        text-align: center;
        padding: 0 1;
    }
    #session-layout {
        height: 1fr;
    }
    #session-list-pane {
        width: 1fr;
        min-width: 24;
        max-width: 48;
    }
    #session-search {
        dock: top;
        margin: 0 0 1 0;
    }
    #session-list {
        border: solid $primary;
        scrollbar-size: 1 1;
    }
    #session-preview {
        width: 3fr;
        border: solid $secondary;
        padding: 1 2;
        scrollbar-size: 1 1;
    }
    #session-preview-md {
        height: auto;
    }
    #session-list Static {
        overflow: hidden hidden;
        height: 1;
    }
    """

    BINDINGS = [
        Binding('escape', 'cancel', 'Cancel', priority=True),
        Binding('q', 'cancel', 'Cancel'),
        Binding('enter', 'select_session', 'Select', priority=True),
    ]

    def __init__(self, sessions_dir: Path, **kwargs):
        super().__init__(**kwargs)
        self._sessions_dir = sessions_dir
        self._sessions = discover_sessions(sessions_dir)
        self._current_session: Path | None = None

    def compose(self) -> ComposeResult:
        count = len(self._sessions)
        yield Static(f'  Select a session ({count} found)', id='session-header')
        with Horizontal(id='session-layout'):
            with Vertical(id='session-list-pane'):
                yield Input(placeholder='Filter sessions...', id='session-search')
                yield _SessionListView(id='session-list')
            with VerticalScroll(id='session-preview', can_focus=False):
                yield Markdown('', id='session-preview-md')
        yield Static('\\[Enter] Select  \\[\u2191/\u2193] Navigate  \\[Esc] Cancel', id='session-footer', markup=True)

    def on_mount(self) -> None:
        self._rebuild_list()
        self.query_one('#session-search', Input).focus()

    def on_key(self, event: Key) -> None:
        if event.key == 'down' and self.focused is self.query_one('#session-search', Input):
            self.query_one('#session-list', _SessionListView).focus()
            event.prevent_default()

    def on_input_changed(self, event: Input.Changed) -> None:
        self._rebuild_list(event.value.strip().lower())

    def _rebuild_list(self, query: str = '') -> None:
        list_view = self.query_one('#session-list', _SessionListView)
        list_view.clear()

        first_item: SessionItem | None = None
        insert_idx: int = 0
        for session in self._sessions:
            if query and query not in session['name'].lower():
                continue
            item = SessionItem(session)
            list_view.append(item)
            if first_item is None:
                first_item = item
            insert_idx += 1

        if first_item is not None:
            list_view.index = 0
            self._current_session = first_item.session_dir
            self._show_preview(first_item.session_dir)
        else:
            self._current_session = None
            self.query_one('#session-preview-md', Markdown).update('*No sessions found*')

    def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
        if isinstance(event.item, SessionItem):
            self._current_session = event.item.session_dir
            self._show_preview(event.item.session_dir)

    def _show_preview(self, session_dir: Path) -> None:
        lines = [f'## {session_dir.name}', '']

        transcript_path = session_dir / 'transcript_raw.txt'
        if transcript_path.exists():
            text = transcript_path.read_text(encoding='utf-8')
            preview_lines = text.strip().splitlines()[:10]
            if preview_lines:
                lines.append('### Transcript (first 10 lines)')
                lines.append('```text')
                lines.extend(preview_lines)
                lines.append('```')
                total = len(text.strip().splitlines())
                if total > 10:
                    lines.append(f'*...{total - 10} more lines*')
            else:
                lines.append('*Empty transcript*')

        digest_path = session_dir / 'digest.md'
        if digest_path.exists():
            lines.extend(['', '---', '', '### Digest'])
            digest_text = digest_path.read_text(encoding='utf-8').strip()
            if digest_text:
                lines.append(digest_text)
            else:
                lines.append('*Empty digest*')

        self.query_one('#session-preview-md', Markdown).update('\n'.join(lines))

    def action_select_session(self) -> None:
        if self._current_session is None:
            return
        self.exit(self._current_session)

    def action_cancel(self) -> None:
        self.exit(None)
