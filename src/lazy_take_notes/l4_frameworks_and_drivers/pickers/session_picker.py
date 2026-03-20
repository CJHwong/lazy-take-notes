"""Session picker — small TUI to select a previously saved session for viewing."""

from __future__ import annotations

from pathlib import Path

from textual.app import ComposeResult
from textual.widgets import ListItem, Markdown, Static

from lazy_take_notes.l1_entities.session_files import NOTES, TRANSCRIPT
from lazy_take_notes.l4_frameworks_and_drivers.pickers.base import (
    PickerListView,
    SearchablePicker,
)


def discover_sessions(sessions_dir: Path) -> list[dict]:
    """Scan *sessions_dir* for session subdirs containing a transcript file.

    Recognises both current and legacy filenames.

    Returns a list of dicts sorted newest-first:
      {'dir': Path, 'name': str, 'has_digest': bool}
    """
    if not sessions_dir.exists():
        return []

    results = []
    for child in sorted(sessions_dir.iterdir(), reverse=True):
        if not child.is_dir():
            continue
        if not TRANSCRIPT.resolve(child):
            continue
        results.append(
            {
                'dir': child,
                'name': child.name,
                'has_digest': NOTES.resolve(child) is not None,
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


class _SessionListView(PickerListView):
    """ListView that pops focus back to #sp-search when ↑ is pressed on the first item."""

    _selectable_type = SessionItem


class SessionPicker(SearchablePicker[Path]):
    CSS = """
    #sp-list-pane { max-width: 48; }
    #sp-list Static { overflow: hidden hidden; height: 1; }
    #sp-preview-md { height: auto; }
    """

    def __init__(self, sessions_dir: Path, **kwargs):
        super().__init__(**kwargs)
        self._sessions_dir = sessions_dir
        self._sessions = discover_sessions(sessions_dir)
        self._current_session: Path | None = None

    def _make_list_view(self) -> _SessionListView:
        return _SessionListView(id='sp-list')

    def _compose_preview(self) -> ComposeResult:
        yield Markdown('', id='sp-preview-md')

    def _header_text(self) -> str:
        return f'  Select a session ({len(self._sessions)} found)'

    def _footer_text(self) -> str:
        return r'\[Enter] Select  \[↑/↓] Navigate  \[Esc] Cancel'

    def _search_placeholder(self) -> str:
        return 'Filter sessions...'

    def _rebuild_list(self, query: str = '') -> None:
        list_view = self.query_one('#sp-list', _SessionListView)
        list_view.clear()

        first_item: SessionItem | None = None
        for session in self._sessions:
            if query and query not in session['name'].lower():
                continue
            item = SessionItem(session)
            list_view.append(item)
            if first_item is None:
                first_item = item

        if first_item is not None:
            list_view.index = 0
            self._current_session = first_item.session_dir
            self._show_preview(first_item.session_dir)
        else:
            self._current_session = None
            self.query_one('#sp-preview-md', Markdown).update('*No sessions found*')

    def _on_item_highlighted(self, item: ListItem) -> None:
        if isinstance(item, SessionItem):
            self._current_session = item.session_dir
            self._show_preview(item.session_dir)

    def _show_preview(self, session_dir: Path) -> None:
        lines = [f'## {session_dir.name}', '']

        transcript_path = TRANSCRIPT.resolve(session_dir)
        if transcript_path:
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

        digest_path = NOTES.resolve(session_dir)
        if digest_path:
            lines.extend(['', '---', '', '### Digest'])
            digest_text = digest_path.read_text(encoding='utf-8').strip()
            if digest_text:
                lines.append(digest_text)
            else:
                lines.append('*Empty digest*')

        self.query_one('#sp-preview-md', Markdown).update('\n'.join(lines))

    def action_select_item(self) -> None:
        if self._current_session is None:
            return
        self.exit(self._current_session)
