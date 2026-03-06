"""File picker — small TUI to browse the filesystem and select an audio file."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import ListItem, ListView, Static

AUDIO_EXTS = frozenset({
    '.mp3', '.wav', '.m4a', '.ogg', '.flac', '.aac', '.opus',
    '.weba', '.webm', '.mp4', '.mov',
})


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
        size = _human_size(path.stat().st_size)
        self._label = f'{path.name}  [dim]{size}[/dim]'

    def compose(self) -> ComposeResult:
        yield Static(self._label, markup=True)


class FilePicker(App[Path | None]):
    CSS = """
    #file-header {
        dock: top;
        height: 1;
        background: $primary;
        color: $text;
        text-style: bold;
        padding: 0 1;
    }
    #file-footer {
        dock: bottom;
        height: 1;
        background: $surface;
        color: $text-muted;
        text-align: center;
        padding: 0 1;
    }
    #file-layout {
        height: 1fr;
    }
    #file-list {
        width: 2fr;
        border: solid $primary;
        scrollbar-size: 1 1;
    }
    #file-list Static {
        overflow: hidden hidden;
        height: 1;
    }
    #file-info {
        width: 1fr;
        border: solid $secondary;
        padding: 1 2;
    }
    """

    BINDINGS = [
        Binding('escape', 'cancel', 'Cancel', priority=True),
        Binding('q', 'cancel', 'Cancel'),
        Binding('enter', 'open', 'Open/Select', priority=True),
        Binding('backspace', 'parent', 'Parent'),
    ]

    def __init__(self, start_dir: Path = Path.cwd(), audio_exts: frozenset[str] = AUDIO_EXTS, **kwargs) -> None:
        super().__init__(**kwargs)
        self._current_dir = start_dir.resolve()
        self._audio_exts = audio_exts

    def compose(self) -> ComposeResult:
        yield Static('', id='file-header')
        with Horizontal(id='file-layout'):
            yield ListView(id='file-list')
            with Vertical(id='file-info'):
                yield Static('', id='file-info-text', markup=True)
        yield Static(
            '\\[Enter] Open/Select  \\[Backspace] Parent  \\[\u2191/\u2193] Navigate  \\[Esc] Cancel',
            id='file-footer',
            markup=True,
        )

    def on_mount(self) -> None:
        self._populate(self._current_dir)
        self.query_one('#file-list', ListView).focus()

    def _populate(self, directory: Path) -> None:
        self._current_dir = directory
        self.query_one('#file-header', Static).update(f'  Select Audio File  {directory}')

        list_view = self.query_one('#file-list', ListView)
        list_view.clear()

        list_view.append(ParentItem())

        try:
            children = list(directory.iterdir())
        except PermissionError:
            self._set_info('Permission denied')
            return

        dirs = sorted((p for p in children if p.is_dir() and not p.name.startswith('.')), key=lambda p: p.name.lower())
        files = sorted(
            (p for p in children if p.is_file() and p.suffix.lower() in self._audio_exts),
            key=lambda p: p.name.lower(),
        )

        for d in dirs:
            list_view.append(DirItem(d))
        for f in files:
            list_view.append(FileItem(f))

        if list_view.children:
            list_view.index = 0
            self._update_info(list_view.children[0])

    def _update_info(self, item: ListItem | None) -> None:
        if item is None:
            self._set_info('')
            return
        if isinstance(item, ParentItem):
            parent = self._current_dir.parent
            self._set_info(f'[bold]..[/bold]\nParent directory\n{parent}')
        elif isinstance(item, DirItem):
            try:
                count = sum(1 for _ in item.path.iterdir())
                self._set_info(f'[bold]{item.path.name}/[/bold]\n{count} items')
            except PermissionError:
                self._set_info(f'[bold]{item.path.name}/[/bold]\nPermission denied')
        elif isinstance(item, FileItem):
            stat = item.path.stat()
            modified = datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M')
            size = _human_size(stat.st_size)
            self._set_info(f'[bold]{item.path.name}[/bold]\nSize: {size}\nModified: {modified}')

    def _set_info(self, text: str) -> None:
        self.query_one('#file-info-text', Static).update(text)

    def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
        self._update_info(event.item)

    def action_open(self) -> None:
        list_view = self.query_one('#file-list', ListView)
        item = list_view.highlighted_child
        if isinstance(item, (ParentItem, DirItem)):
            target = self._current_dir.parent if isinstance(item, ParentItem) else item.path
            self._populate(target)
        elif isinstance(item, FileItem):
            self.exit(item.path)

    def action_parent(self) -> None:
        self._populate(self._current_dir.parent)

    def action_cancel(self) -> None:
        self.exit(None)
