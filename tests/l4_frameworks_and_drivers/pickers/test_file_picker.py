"""Tests for file picker — TUI to browse filesystem and select an audio file."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from textual.widgets import Input, Static

from lazy_take_notes.l4_frameworks_and_drivers.pickers.file_picker import (
    DirItem,
    FileItem,
    FilePicker,
    ParentItem,
    human_size,
)

# ── Fixture ───────────────────────────────────────────────────────────────────


@pytest.fixture()
def audio_dir(tmp_path: Path) -> Path:
    """Directory fixture with mixed content for FilePicker tests.

    Sorted list order after mount:
      index 0: ParentItem
      index 1: DirItem(music/)
      index 2: DirItem(podcasts/)
      index 3: FileItem(recording.wav)
      index 4: FileItem(song.mp3)
    """
    (tmp_path / 'music').mkdir()
    (tmp_path / 'podcasts').mkdir()
    (tmp_path / '.hidden').mkdir()
    (tmp_path / 'song.mp3').write_bytes(b'\xff\xfb' * 512)  # ~1 KB fake mp3
    (tmp_path / 'recording.wav').write_bytes(b'RIFF' + b'\x00' * 508)  # ~512 B fake wav
    (tmp_path / 'notes.txt').write_text('hello', encoding='utf-8')
    return tmp_path


def _preview_text(picker: FilePicker) -> str:
    """Return the raw text content of the preview Static widget."""
    return str(picker.query_one('#sp-preview-info', Static).content)


# ── 1. Pure Unit Tests ────────────────────────────────────────────────────────


class TestHumanSize:
    def test_bytes(self) -> None:
        assert human_size(500) == '500 B'

    def test_kilobytes(self) -> None:
        assert human_size(1024) == '1.0 KB'

    def test_megabytes(self) -> None:
        assert human_size(1024 * 1024) == '1.0 MB'

    def test_terabytes(self) -> None:
        assert human_size(1024**4) == '1.0 TB'


# ── 2. Mount & Initial State ──────────────────────────────────────────────────


class TestFilePickerMount:
    @pytest.mark.asyncio
    async def test_mount_focuses_list(self, audio_dir: Path) -> None:
        picker = FilePicker(start_dir=audio_dir)
        async with picker.run_test() as pilot:
            # Two pauses: first drains on_mount events, second lets call_after_refresh fire
            await pilot.pause()
            await pilot.pause()
            assert picker.focused is picker.query_one('#sp-list')

    @pytest.mark.asyncio
    async def test_mount_shows_parent_item(self, audio_dir: Path) -> None:
        picker = FilePicker(start_dir=audio_dir)
        async with picker.run_test() as pilot:
            await pilot.pause()
            list_view = picker.query_one('#sp-list')
            parent_items = [c for c in list_view.children if isinstance(c, ParentItem)]
            assert len(parent_items) == 1

    @pytest.mark.asyncio
    async def test_mount_shows_subdirs(self, audio_dir: Path) -> None:
        picker = FilePicker(start_dir=audio_dir)
        async with picker.run_test() as pilot:
            await pilot.pause()
            list_view = picker.query_one('#sp-list')
            dir_items = [c for c in list_view.children if isinstance(c, DirItem)]
            assert len(dir_items) == 2

    @pytest.mark.asyncio
    async def test_mount_shows_audio_files(self, audio_dir: Path) -> None:
        picker = FilePicker(start_dir=audio_dir)
        async with picker.run_test() as pilot:
            await pilot.pause()
            list_view = picker.query_one('#sp-list')
            file_items = [c for c in list_view.children if isinstance(c, FileItem)]
            assert len(file_items) == 2

    @pytest.mark.asyncio
    async def test_mount_excludes_hidden_dirs(self, audio_dir: Path) -> None:
        picker = FilePicker(start_dir=audio_dir)
        async with picker.run_test() as pilot:
            await pilot.pause()
            list_view = picker.query_one('#sp-list')
            dir_names = [c.path.name for c in list_view.children if isinstance(c, DirItem)]
            assert '.hidden' not in dir_names

    @pytest.mark.asyncio
    async def test_mount_excludes_non_audio_files(self, audio_dir: Path) -> None:
        picker = FilePicker(start_dir=audio_dir)
        async with picker.run_test() as pilot:
            await pilot.pause()
            list_view = picker.query_one('#sp-list')
            file_names = [c.path.name for c in list_view.children if isinstance(c, FileItem)]
            assert 'notes.txt' not in file_names

    @pytest.mark.asyncio
    async def test_header_shows_current_dir(self, audio_dir: Path) -> None:
        picker = FilePicker(start_dir=audio_dir)
        async with picker.run_test() as pilot:
            await pilot.pause()
            header = picker.query_one('#sp-header', Static)
            assert str(audio_dir) in str(header.content)


# ── 3. Navigation ─────────────────────────────────────────────────────────────


class TestFilePickerNavigation:
    @pytest.mark.asyncio
    async def test_enter_on_parent_item_navigates_up(self, audio_dir: Path) -> None:
        picker = FilePicker(start_dir=audio_dir)
        async with picker.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()  # wait for call_after_refresh to focus the list
            # List is focused at index 0 (ParentItem)
            await pilot.press('enter')
            await pilot.pause()
        assert picker._current_dir == audio_dir.parent

    @pytest.mark.asyncio
    async def test_enter_on_dir_navigates_into(self, audio_dir: Path) -> None:
        picker = FilePicker(start_dir=audio_dir)
        async with picker.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()  # wait for call_after_refresh
            await pilot.press('down')  # index 0 → 1: music/
            await pilot.pause()
            await pilot.press('enter')
            await pilot.pause()
        assert picker._current_dir == audio_dir / 'music'

    @pytest.mark.asyncio
    async def test_backspace_navigates_to_parent(self, audio_dir: Path) -> None:
        picker = FilePicker(start_dir=audio_dir)
        async with picker.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()  # wait for call_after_refresh (list must be focused)
            await pilot.press('backspace')
            await pilot.pause()
        assert picker._current_dir == audio_dir.parent

    @pytest.mark.asyncio
    async def test_navigate_updates_header(self, audio_dir: Path) -> None:
        picker = FilePicker(start_dir=audio_dir)
        async with picker.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            await pilot.press('backspace')
            await pilot.pause()
            header = picker.query_one('#sp-header', Static)
            assert str(audio_dir.parent) in str(header.content)

    @pytest.mark.asyncio
    async def test_navigate_rebuilds_list(self, audio_dir: Path) -> None:
        # music/ is empty — after navigating into it only ParentItem should remain
        picker = FilePicker(start_dir=audio_dir)
        async with picker.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            await pilot.press('down')  # index 0 → 1: music/
            await pilot.pause()
            await pilot.press('enter')
            await pilot.pause()
            children = list(picker.query_one('#sp-list').children)
            assert len(children) == 1
            assert isinstance(children[0], ParentItem)


# ── 4. File Selection & Cancellation ─────────────────────────────────────────


class TestFilePickerSelection:
    @pytest.mark.asyncio
    async def test_enter_on_file_exits_with_path(self, audio_dir: Path) -> None:
        picker = FilePicker(start_dir=audio_dir)
        async with picker.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()  # wait for list focus
            await pilot.press('down')  # index 0 → 1 (music/)
            await pilot.pause()
            await pilot.press('down')  # index 1 → 2 (podcasts/)
            await pilot.pause()
            await pilot.press('down')  # index 2 → 3 (recording.wav)
            await pilot.pause()
            await pilot.press('enter')
            await pilot.pause()
        assert picker.return_value == audio_dir / 'recording.wav'

    @pytest.mark.asyncio
    async def test_escape_returns_none(self, audio_dir: Path) -> None:
        picker = FilePicker(start_dir=audio_dir)
        async with picker.run_test() as pilot:
            await pilot.pause()
            await pilot.press('escape')
            await pilot.pause()
        assert picker.return_value is None

    @pytest.mark.asyncio
    async def test_q_cancels(self, audio_dir: Path) -> None:
        picker = FilePicker(start_dir=audio_dir)
        async with picker.run_test() as pilot:
            await pilot.pause()
            await pilot.press('q')
            await pilot.pause()
        assert picker.return_value is None


# ── 5. Search / Filter ────────────────────────────────────────────────────────


class TestFilePickerSearch:
    @pytest.mark.asyncio
    async def test_search_filters_dirs(self, audio_dir: Path) -> None:
        picker = FilePicker(start_dir=audio_dir)
        async with picker.run_test() as pilot:
            await pilot.pause()
            picker.query_one('#sp-search', Input).value = 'music'
            await pilot.pause()
            dir_items = [c for c in picker.query_one('#sp-list').children if isinstance(c, DirItem)]
            assert len(dir_items) == 1
            assert dir_items[0].path.name == 'music'

    @pytest.mark.asyncio
    async def test_search_filters_files(self, audio_dir: Path) -> None:
        picker = FilePicker(start_dir=audio_dir)
        async with picker.run_test() as pilot:
            await pilot.pause()
            picker.query_one('#sp-search', Input).value = 'song'
            await pilot.pause()
            file_items = [c for c in picker.query_one('#sp-list').children if isinstance(c, FileItem)]
            assert len(file_items) == 1
            assert file_items[0].path.name == 'song.mp3'

    @pytest.mark.asyncio
    async def test_search_case_insensitive(self, audio_dir: Path) -> None:
        picker = FilePicker(start_dir=audio_dir)
        async with picker.run_test() as pilot:
            await pilot.pause()
            picker.query_one('#sp-search', Input).value = 'MUSIC'
            await pilot.pause()
            dir_items = [c for c in picker.query_one('#sp-list').children if isinstance(c, DirItem)]
            assert len(dir_items) == 1
            assert dir_items[0].path.name == 'music'


# ── 6. Preview Info ───────────────────────────────────────────────────────────


class TestFilePickerPreview:
    @pytest.mark.asyncio
    async def test_preview_parent_item_on_mount(self, audio_dir: Path) -> None:
        picker = FilePicker(start_dir=audio_dir)
        async with picker.run_test() as pilot:
            await pilot.pause()
            info_text = _preview_text(picker)
            assert '..' in info_text
            assert str(audio_dir.parent) in info_text

    @pytest.mark.asyncio
    async def test_preview_file_item(self, audio_dir: Path) -> None:
        picker = FilePicker(start_dir=audio_dir)
        async with picker.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()  # wait for list focus
            await pilot.press('down')  # index 0 → 1: music/
            await pilot.pause()
            await pilot.press('down')  # index 1 → 2: podcasts/
            await pilot.pause()
            await pilot.press('down')  # index 2 → 3: recording.wav
            await pilot.pause()
            info_text = _preview_text(picker)
            assert 'recording.wav' in info_text
            assert 'Size:' in info_text
            assert 'Modified:' in info_text


# ── 7. Keyboard Shortcuts (base class integration) ────────────────────────────


class TestFilePickerKeyboard:
    @pytest.mark.asyncio
    async def test_up_arrow_on_first_item_focuses_search(self, audio_dir: Path) -> None:
        picker = FilePicker(start_dir=audio_dir)
        async with picker.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()  # wait for list focus via call_after_refresh
            # List is focused at index 0 (ParentItem) — Up should hand focus to search
            await pilot.press('up')
            await pilot.pause()
            assert picker.focused is picker.query_one('#sp-search', Input)


# ── 8. Edge Cases ─────────────────────────────────────────────────────────────


class TestFilePickerEdgeCases:
    @pytest.mark.asyncio
    async def test_permission_denied_shows_error(self, audio_dir: Path) -> None:
        picker = FilePicker(start_dir=audio_dir)
        async with picker.run_test() as pilot:
            await pilot.pause()
            with patch('pathlib.Path.iterdir', side_effect=PermissionError):
                picker._rebuild_list()
                await pilot.pause()
            assert 'Permission denied' in _preview_text(picker)

    @pytest.mark.asyncio
    async def test_dir_count_ready_stale_is_ignored(self, audio_dir: Path) -> None:
        picker = FilePicker(start_dir=audio_dir)
        async with picker.run_test() as pilot:
            await pilot.pause()
            info_before = _preview_text(picker)
            # Post a DirCountReady for a path that doesn't match _highlighted_dir (stale)
            wrong_path = audio_dir / 'nonexistent'
            picker.post_message(FilePicker.DirCountReady(wrong_path, '42'))
            await pilot.pause()
            assert _preview_text(picker) == info_before

    @pytest.mark.asyncio
    async def test_empty_directory(self, tmp_path: Path) -> None:
        empty_dir = tmp_path / 'empty'
        empty_dir.mkdir()
        picker = FilePicker(start_dir=empty_dir)
        async with picker.run_test() as pilot:
            await pilot.pause()
            children = list(picker.query_one('#sp-list').children)
            assert len(children) == 1
            assert isinstance(children[0], ParentItem)
            # Enter on ParentItem should navigate up without crashing
            await pilot.press('enter')
            await pilot.pause()
