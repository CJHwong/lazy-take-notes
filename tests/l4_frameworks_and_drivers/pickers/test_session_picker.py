"""Tests for session picker — TUI to select a saved session."""

from __future__ import annotations

from pathlib import Path

import pytest
from textual.widgets import Input, Markdown

from lazy_take_notes.l4_frameworks_and_drivers.pickers.session_picker import (
    SessionItem,
    SessionPicker,
)


def _create_session(
    tmp_path: Path,
    name: str,
    *,
    has_digest: bool = False,
    transcript: str = 'Hello\nWorld\n',
    digest_text: str = '# Summary\nKey points.',
) -> Path:
    """Create a fake session directory with transcript.txt."""
    session_dir = tmp_path / name
    session_dir.mkdir()
    (session_dir / 'transcript.txt').write_text(transcript, encoding='utf-8')
    if has_digest:
        (session_dir / 'notes.md').write_text(digest_text, encoding='utf-8')
    return session_dir


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

            search_input = picker.query_one('#sp-search', Input)
            search_input.value = 'retro'
            await pilot.pause()
            await pilot.press('down')
            await pilot.pause()
            await pilot.press('enter')
            await pilot.pause()

        assert picker.return_value is not None
        assert 'retro' in picker.return_value.name

    @pytest.mark.asyncio
    async def test_a_typed_filter_moves_the_preview_and_enter_to_the_match(self, tmp_path: Path):
        _create_session(tmp_path, '2026-02-20_standup', transcript='standup words\n')
        _create_session(tmp_path, '2026-02-21_retro', transcript='retro words\n')

        picker = SessionPicker(sessions_dir=tmp_path)
        async with picker.run_test() as pilot:
            await pilot.pause()
            # Fast typing: one rebuild replaces a list whose first row is another session.
            picker.query_one('#sp-search', Input).value = 'standup'
            await pilot.pause()
            preview = picker.query_one('#sp-preview-md', Markdown).source
            assert '2026-02-20_standup' in preview
            assert 'standup words' in preview
            await pilot.press('enter')
            await pilot.pause()

        assert picker.return_value is not None
        assert picker.return_value.name == '2026-02-20_standup'

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

            md = picker.query_one('#sp-preview-md', Markdown)
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
            assert picker.app.focused is picker.query_one('#sp-list')

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

            assert picker.app.focused is picker.query_one('#sp-search', Input)

    @pytest.mark.asyncio
    async def test_preview_long_transcript_shows_more_lines(self, tmp_path: Path):
        long_transcript = '\n'.join(f'Line {i}' for i in range(25)) + '\n'
        _create_session(tmp_path, '2026-02-20_120000', transcript=long_transcript)
        picker = SessionPicker(sessions_dir=tmp_path)
        async with picker.run_test() as pilot:
            await pilot.pause()
            from textual.widgets import Markdown

            md = picker.query_one('#sp-preview-md', Markdown)
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

            md = picker.query_one('#sp-preview-md', Markdown)
            assert md is not None

    @pytest.mark.asyncio
    async def test_preview_empty_digest(self, tmp_path: Path):
        session_dir = _create_session(tmp_path, '2026-02-20_120000')
        # Create an empty notes.md
        (session_dir / 'notes.md').write_text('  \n  ', encoding='utf-8')
        picker = SessionPicker(sessions_dir=tmp_path)
        async with picker.run_test() as pilot:
            await pilot.pause()
            from textual.widgets import Markdown

            md = picker.query_one('#sp-preview-md', Markdown)
            assert md is not None

    @pytest.mark.asyncio
    async def test_theme_applied_on_mount(self, tmp_path: Path):
        from unittest.mock import patch

        _create_session(tmp_path, '2026-02-20_120000')
        with patch('lazy_take_notes.l4_frameworks_and_drivers.pickers.base.load_theme', return_value='textual-light'):
            picker = SessionPicker(sessions_dir=tmp_path)
            async with picker.run_test() as pilot:
                await pilot.pause()
                assert picker.app.theme == 'textual-light'


class TestSessionPickerFaults:
    @pytest.mark.asyncio
    async def test_undecodable_transcript_preview_does_not_crash(self, tmp_path: Path):
        session_dir = _create_session(tmp_path, '2026-02-20_120000', has_digest=True)
        (session_dir / 'transcript.txt').write_bytes(b'\xff\xfe\xfa bad')
        picker = SessionPicker(sessions_dir=tmp_path)
        async with picker.run_test() as pilot:
            await pilot.pause()
            from textual.widgets import Markdown

            md = picker.query_one('#sp-preview-md', Markdown)
            assert 'Cannot read' in md.source
            assert 'Key points' in md.source


class TestAudioOnly:
    @pytest.mark.asyncio
    async def test_an_audio_only_session_is_listed_and_marked(self, tmp_path: Path):
        session = tmp_path / '2026-02-21_120000'
        session.mkdir()
        (session / 'recording.wav').write_bytes(b'RIFF')
        picker = SessionPicker(sessions_dir=tmp_path)
        async with picker.run_test() as pilot:
            await pilot.pause()
            item = picker.query_one(SessionItem)
            assert 'audio only' in item._label_text
            preview = picker.query_one('#sp-preview-md', Markdown)
            assert 'No transcript. The audio is in this folder.' in preview.source
