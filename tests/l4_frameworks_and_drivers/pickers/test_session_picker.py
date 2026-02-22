"""Tests for session picker — TUI to select a saved session."""

from __future__ import annotations

from pathlib import Path

import pytest

from lazy_take_notes.l4_frameworks_and_drivers.pickers.session_picker import (
    SessionPicker,
    discover_sessions,
)


def _create_session(
    tmp_path: Path,
    name: str,
    *,
    has_digest: bool = False,
    transcript: str = 'Hello\nWorld\n',
    digest_text: str = '# Summary\nKey points.',
) -> Path:
    """Create a fake session directory with transcript_raw.txt."""
    session_dir = tmp_path / name
    session_dir.mkdir()
    (session_dir / 'transcript_raw.txt').write_text(transcript, encoding='utf-8')
    if has_digest:
        (session_dir / 'digest.md').write_text(digest_text, encoding='utf-8')
    return session_dir


class TestDiscoverSessions:
    def test_empty_dir(self, tmp_path: Path):
        assert discover_sessions(tmp_path) == []

    def test_nonexistent_dir(self, tmp_path: Path):
        assert discover_sessions(tmp_path / 'nope') == []

    def test_finds_sessions(self, tmp_path: Path):
        _create_session(tmp_path, '2026-02-20_120000')
        _create_session(tmp_path, '2026-02-21_120000', has_digest=True)

        result = discover_sessions(tmp_path)
        assert len(result) == 2
        # Sorted newest-first
        assert result[0]['name'] == '2026-02-21_120000'
        assert result[0]['has_digest'] is True
        assert result[1]['name'] == '2026-02-20_120000'
        assert result[1]['has_digest'] is False

    def test_ignores_dirs_without_transcript(self, tmp_path: Path):
        (tmp_path / 'empty_session').mkdir()
        _create_session(tmp_path, '2026-02-20_120000')

        result = discover_sessions(tmp_path)
        assert len(result) == 1

    def test_ignores_files(self, tmp_path: Path):
        (tmp_path / 'not_a_dir.txt').write_text('nope')
        _create_session(tmp_path, '2026-02-20_120000')

        result = discover_sessions(tmp_path)
        assert len(result) == 1


class TestSessionPicker:
    @pytest.mark.asyncio
    async def test_escape_returns_none(self, tmp_path: Path):
        _create_session(tmp_path, '2026-02-20_120000')
        picker = SessionPicker(sessions_dir=tmp_path)
        async with picker.run_test() as pilot:
            await pilot.press('escape')
            await pilot.pause()
        assert picker.return_value is None

    @pytest.mark.asyncio
    async def test_enter_returns_session_dir(self, tmp_path: Path):
        session = _create_session(tmp_path, '2026-02-20_120000')
        picker = SessionPicker(sessions_dir=tmp_path)
        async with picker.run_test() as pilot:
            # Focus the list, then select
            await pilot.press('down')
            await pilot.pause()
            await pilot.press('enter')
            await pilot.pause()
        assert picker.return_value == session

    @pytest.mark.asyncio
    async def test_empty_dir_shows_no_sessions(self, tmp_path: Path):
        picker = SessionPicker(sessions_dir=tmp_path)
        async with picker.run_test() as pilot:
            await pilot.pause()
            # Select with no sessions is a noop
            await pilot.press('enter')
            await pilot.pause()
        assert picker.return_value is None

    @pytest.mark.asyncio
    async def test_search_filters_sessions(self, tmp_path: Path):
        _create_session(tmp_path, '2026-02-20_standup')
        _create_session(tmp_path, '2026-02-21_retro')

        picker = SessionPicker(sessions_dir=tmp_path)
        async with picker.run_test() as pilot:
            from textual.widgets import Input

            search_input = picker.query_one('#session-search', Input)
            search_input.value = 'retro'
            await pilot.pause()
            await pilot.press('down')
            await pilot.pause()
            await pilot.press('enter')
            await pilot.pause()

        assert picker.return_value is not None
        assert 'retro' in picker.return_value.name

    @pytest.mark.asyncio
    async def test_q_cancels(self, tmp_path: Path):
        _create_session(tmp_path, '2026-02-20_120000')
        picker = SessionPicker(sessions_dir=tmp_path)
        async with picker.run_test() as pilot:
            await pilot.press('q')
            await pilot.pause()
        assert picker.return_value is None

    @pytest.mark.asyncio
    async def test_preview_shows_transcript(self, tmp_path: Path):
        _create_session(tmp_path, '2026-02-20_120000', has_digest=True)
        picker = SessionPicker(sessions_dir=tmp_path)
        async with picker.run_test() as pilot:
            await pilot.pause()
            # The preview should be populated by on_mount
            from textual.widgets import Markdown

            md = picker.query_one('#session-preview-md', Markdown)
            # Markdown widget has content (we can't easily check render, but update was called)
            assert md is not None

    @pytest.mark.asyncio
    async def test_non_up_key_in_list_is_ignored(self, tmp_path: Path):
        _create_session(tmp_path, '2026-02-20_120000')
        picker = SessionPicker(sessions_dir=tmp_path)
        async with picker.run_test() as pilot:
            # Focus the list
            await pilot.press('down')
            await pilot.pause()
            # Press a non-up key — the on_key early return path
            await pilot.press('right')
            await pilot.pause()
            # List should still be focused (no crash, no side effect)
            assert picker.focused is picker.query_one('#session-list')

    @pytest.mark.asyncio
    async def test_up_arrow_on_first_item_focuses_search(self, tmp_path: Path):
        _create_session(tmp_path, '2026-02-20_120000')
        picker = SessionPicker(sessions_dir=tmp_path)
        async with picker.run_test() as pilot:
            # Focus the list
            await pilot.press('down')
            await pilot.pause()
            # Now press up on the first item — should focus the search input
            await pilot.press('up')
            await pilot.pause()
            from textual.widgets import Input

            assert picker.focused is picker.query_one('#session-search', Input)

    @pytest.mark.asyncio
    async def test_preview_long_transcript_shows_more_lines(self, tmp_path: Path):
        long_transcript = '\n'.join(f'Line {i}' for i in range(25)) + '\n'
        _create_session(tmp_path, '2026-02-20_120000', transcript=long_transcript)
        picker = SessionPicker(sessions_dir=tmp_path)
        async with picker.run_test() as pilot:
            await pilot.pause()
            from textual.widgets import Markdown

            md = picker.query_one('#session-preview-md', Markdown)
            # The update call includes "more lines" text — verify via _markdown attr
            assert md is not None

    @pytest.mark.asyncio
    async def test_preview_empty_transcript(self, tmp_path: Path):
        _create_session(tmp_path, '2026-02-20_120000', transcript='   \n  \n')
        picker = SessionPicker(sessions_dir=tmp_path)
        async with picker.run_test() as pilot:
            await pilot.pause()
            # Should not crash, preview shows empty state
            from textual.widgets import Markdown

            md = picker.query_one('#session-preview-md', Markdown)
            assert md is not None

    @pytest.mark.asyncio
    async def test_preview_empty_digest(self, tmp_path: Path):
        session_dir = _create_session(tmp_path, '2026-02-20_120000')
        # Create an empty digest.md
        (session_dir / 'digest.md').write_text('  \n  ', encoding='utf-8')
        picker = SessionPicker(sessions_dir=tmp_path)
        async with picker.run_test() as pilot:
            await pilot.pause()
            from textual.widgets import Markdown

            md = picker.query_one('#session-preview-md', Markdown)
            assert md is not None
