"""File picker — small TUI to browse the filesystem and select an audio file."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from textual.app import ComposeResult
from textual.binding import Binding
from textual.message import Message
from textual.widgets import Input, ListItem, Static

from lazy_take_notes.l4_frameworks_and_drivers.pickers.base import (
    PickerListView,
    SearchablePicker,
)

AUDIO_EXTS = frozenset({'.mp3', '.wav', '.m4a', '.mp4'})


def _human_size(n: int) -> str:
    for unit in ('B', 'KB', 'MB', 'GB'):
        if n < 1024:
            return f'{n:.0f} {unit}'
        n /= 1024  # type: ignore[assignment]
    return f'{n:.1f} TB'


class ParentItem(ListItem):
    def compose(self) -> ComposeResult:
        yield Static('[..]', markup=False)


class DirItem(ListItem):
    def __init__(self, path: Path) -> None:
        super().__init__()
        self.path = path

    def compose(self) -> ComposeResult:
        yield Static(f'[{self.path.name}/]', markup=False)


class FileItem(ListItem):
    def __init__(self, path: Path) -> None:
        super().__init__()
        self.path = path
        try:
            stat_result = path.stat()
        except OSError:
            size_str = 'unknown'
        else:
            size_str = _human_size(stat_result.st_size)
        self._label = f'{path.name}  [dim]{size_str}[/dim]'

    def compose(self) -> ComposeResult:
        yield Static(self._label, markup=True)


class _FileListView(PickerListView):
    """ListView that pops focus back to #sp-search when ↑ is pressed on the first item."""

    _selectable_type = ListItem  # ParentItem / DirItem / FileItem are all selectable


class FilePicker(SearchablePicker):
    CSS = """
    #sp-list-pane { width: 2fr; max-width: 9999; }
    #sp-preview   { width: 1fr; }
    #sp-list Static { overflow: hidden hidden; height: 1; }
    """

    BINDINGS = [
        *SearchablePicker.BINDINGS,
        Binding('backspace', 'parent', 'Parent'),
    ]

    class DirCountReady(Message):
        def __init__(self, path: Path, label: str) -> None:
            super().__init__()
            self.path = path
            self.label = label

    def __init__(self, start_dir: Path = Path.cwd(), audio_exts: frozenset[str] = AUDIO_EXTS, **kwargs) -> None:
        super().__init__(**kwargs)
        self._current_dir = start_dir.resolve()
        self._audio_exts = audio_exts
        self._highlighted_dir: Path | None = None

    def on_mount(self) -> None:
        # Override base: focus the list instead of the search input
        self._rebuild_list()
        self.query_one('#sp-list', _FileListView).focus()

    def _make_list_view(self) -> _FileListView:
        return _FileListView(id='sp-list')

    def _compose_preview(self) -> ComposeResult:
        yield Static('', id='sp-preview-info', markup=True)

    def _header_text(self) -> str:
        return f'  Select Audio File  {self._current_dir}'

    def _footer_text(self) -> str:
        return r'\[Enter] Open/Select  \[Backspace] Parent  \[↑/↓] Navigate  \[Esc] Cancel'

    def _search_placeholder(self) -> str:
        return 'Filter files...'

    def _rebuild_list(self, query: str = '') -> None:
        list_view = self.query_one('#sp-list', _FileListView)
        list_view.clear()
        list_view.append(ParentItem())

        try:
            children = list(self._current_dir.iterdir())
        except PermissionError:
            self._set_info('Permission denied')
            return

        dirs = sorted(
            (p for p in children if p.is_dir() and not p.name.startswith('.')),
            key=lambda p: p.name.lower(),
        )
        files = sorted(
            (p for p in children if p.is_file() and p.suffix.lower() in self._audio_exts),
            key=lambda p: p.name.lower(),
        )

        if query:
            dirs = [d for d in dirs if query in d.name.lower()]
            files = [f for f in files if query in f.name.lower()]

        for d in dirs:
            list_view.append(DirItem(d))
        for f in files:
            list_view.append(FileItem(f))

        if list_view.children:
            list_view.index = 0
            self._update_info(list_view.children[0])

    def _navigate_to(self, directory: Path) -> None:
        self._current_dir = directory
        self._refresh_header()
        query = self.query_one('#sp-search', Input).value.strip().lower()
        self._rebuild_list(query)

    def _on_item_highlighted(self, item: ListItem) -> None:
        self._update_info(item)

    def action_select_item(self) -> None:
        list_view = self.query_one('#sp-list', _FileListView)
        item = list_view.highlighted_child
        if isinstance(item, (ParentItem, DirItem)):
            target = self._current_dir.parent if isinstance(item, ParentItem) else item.path
            self._navigate_to(target)
        elif isinstance(item, FileItem):
            self.exit(item.path)

    def action_parent(self) -> None:
        self._navigate_to(self._current_dir.parent)

    def _update_info(self, item: ListItem | None) -> None:
        if item is None:
            self._set_info('')
            return
        if isinstance(item, ParentItem):
            parent = self._current_dir.parent
            self._set_info(f'[bold]..[/bold]\nParent directory\n{parent}')
        elif isinstance(item, DirItem):
            self._highlighted_dir = item.path
            self._set_info(f'[bold]{item.path.name}/[/bold]\n(counting…)')
            path = item.path

            def _count() -> None:
                try:
                    _CAP = 1000
                    count = 0
                    for _ in path.iterdir():
                        count += 1
                        if count > _CAP:
                            break
                    result_label = f'{_CAP}+' if count > _CAP else str(count)
                except PermissionError:
                    result_label = 'Permission denied'
                self.post_message(FilePicker.DirCountReady(path, result_label))

            self.run_worker(_count, thread=True, exclusive=True, group='dir-count')
        elif isinstance(item, FileItem):
            stat = item.path.stat()
            modified = datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M')
            size = _human_size(stat.st_size)
            self._set_info(f'[bold]{item.path.name}[/bold]\nSize: {size}\nModified: {modified}')

    def on_file_picker_dir_count_ready(self, msg: DirCountReady) -> None:
        if msg.path != self._highlighted_dir:
            return  # stale result — user has moved on
        item_name = msg.path.name
        if msg.label == 'Permission denied':
            self._set_info(f'[bold]{item_name}/[/bold]\nPermission denied')
        else:
            self._set_info(f'[bold]{item_name}/[/bold]\n{msg.label} items')

    def _set_info(self, text: str) -> None:
        self.query_one('#sp-preview-info', Static).update(text)
