"""File picker — small TUI to browse the filesystem and select an audio file."""

from __future__ import annotations

import os
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
_DIR_COUNT_CAP = 1000


def human_size(n: int) -> str:
    size = float(n)
    for unit in ('B', 'KB', 'MB', 'GB'):
        if size < 1024:
            fmt = '.0f' if unit == 'B' else '.1f'
            return f'{size:{fmt}} {unit}'
        size /= 1024
    return f'{size:.1f} TB'


def _count_dir_items(path: Path) -> str:
    try:
        count = 0
        for _ in path.iterdir():
            count += 1
            if count > _DIR_COUNT_CAP:
                break
        return f'{_DIR_COUNT_CAP}+' if count > _DIR_COUNT_CAP else str(count)
    except PermissionError:
        return 'Permission denied'


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
        self._stat: os.stat_result | None
        try:
            stat_result = path.stat()
        except OSError:
            self._stat = None
            size_str = 'unknown'
        else:
            self._stat = stat_result
            size_str = human_size(stat_result.st_size)
        self._size_str = size_str
        self._label = f'{path.name}  [dim]{self._size_str}[/dim]'

    def compose(self) -> ComposeResult:
        yield Static(self._label, markup=True)


class _FileListView(PickerListView):
    """ListView that pops focus back to #sp-search when ↑ is pressed on the first item."""

    _selectable_type = ListItem  # ParentItem / DirItem / FileItem are all selectable


class FilePicker(SearchablePicker[Path]):
    CSS = """
    #sp-list-pane { width: 2fr; max-width: 9999; }
    #sp-preview   { width: 1fr; }
    #sp-list Static { overflow: hidden hidden; height: 1; }
    """

    BINDINGS = [
        *SearchablePicker.BINDINGS,
        Binding('backspace', 'parent', 'Parent'),
        Binding('left', 'parent', 'Parent'),
        Binding('right', 'enter_dir', 'Enter dir'),
    ]

    class DirCountReady(Message):
        def __init__(self, path: Path, label: str) -> None:
            super().__init__()
            self.path = path
            self.label = label

    def __init__(self, start_dir: Path | None = None, audio_exts: frozenset[str] = AUDIO_EXTS, **kwargs) -> None:
        super().__init__(**kwargs)
        self._current_dir = (start_dir or Path.cwd()).resolve()
        self._audio_exts = audio_exts
        self._highlighted_dir: Path | None = None

    def on_mount(self) -> None:
        # Override base on_mount entirely: FilePicker starts with the list focused,
        # not the search input. call_after_refresh defers focus until after the
        # first render so the widget is ready to receive it.
        self._rebuild_list()
        self.call_after_refresh(self.query_one('#sp-list', _FileListView).focus)

    def _make_list_view(self) -> _FileListView:
        return _FileListView(id='sp-list')

    def _compose_preview(self) -> ComposeResult:
        yield Static('', id='sp-preview-info', markup=True)

    def _header_text(self) -> str:
        return f'  Select Audio File  {self._current_dir}'

    def _footer_text(self) -> str:
        return (
            r'\[Enter] Open/Select  \[Left/Backspace] Parent  '
            r'\[Right] Enter dir  \[↑/↓] Navigate  \[Esc] Cancel'
        )

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

        first_child = list_view.children[0] if list_view.children else None
        if isinstance(first_child, ListItem):
            list_view.index = 0
            self._update_info(first_child)

    def _navigate_to(self, directory: Path) -> None:
        self._current_dir = directory
        # Navigating to a new directory invalidates any in-flight dir counts.
        self._highlighted_dir = None
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

    def action_enter_dir(self) -> None:
        list_view = self.query_one('#sp-list', _FileListView)
        item = list_view.highlighted_child
        if isinstance(item, DirItem):
            self._navigate_to(item.path)

    def _update_info(self, item: ListItem | None) -> None:
        if item is None:
            self._highlighted_dir = None
            self._set_info('')
            return
        if isinstance(item, ParentItem):
            # No directory being counted; invalidate any in-flight DirCountReady.
            self._highlighted_dir = None
            parent = self._current_dir.parent
            self._set_info(f'[bold]..[/bold]\nParent directory\n{parent}')
        elif isinstance(item, DirItem):
            self._highlighted_dir = item.path
            self._set_info(f'[bold]{item.path.name}/[/bold]\n(counting…)')
            self._start_dir_count(item.path)
        elif isinstance(item, FileItem):
            # File highlighted; any outstanding dir count is now stale.
            self._highlighted_dir = None
            try:
                stat_result = item.path.stat()
            except OSError:
                stat_result = None
            if stat_result is None:
                modified = 'unknown'
                size = item._size_str
            else:
                modified = datetime.fromtimestamp(stat_result.st_mtime).strftime('%Y-%m-%d %H:%M')
                size = human_size(stat_result.st_size)
            self._set_info(f'[bold]{item.path.name}[/bold]\nSize: {size}\nModified: {modified}')

    def _start_dir_count(self, path: Path) -> None:
        def _count() -> None:
            result_label = _count_dir_items(path)
            self.post_message(FilePicker.DirCountReady(path, result_label))

        self.run_worker(_count, thread=True, exclusive=True, group='dir-count')

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
