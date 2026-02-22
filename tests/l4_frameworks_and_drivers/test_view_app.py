"""Tests for ViewApp — read-only session viewer TUI."""

from __future__ import annotations

from pathlib import Path

import pytest

from lazy_take_notes.l4_frameworks_and_drivers.apps.view import ViewApp
from lazy_take_notes.l4_frameworks_and_drivers.widgets.digest_panel import DigestPanel
from lazy_take_notes.l4_frameworks_and_drivers.widgets.status_bar import StatusBar
from lazy_take_notes.l4_frameworks_and_drivers.widgets.transcript_panel import TranscriptPanel


def _make_session(tmp_path: Path, *, transcript: str = '', digest: str = '') -> Path:
    session_dir = tmp_path / '2026-02-20_120000'
    session_dir.mkdir()
    if transcript:
        (session_dir / 'transcript_raw.txt').write_text(transcript, encoding='utf-8')
    if digest:
        (session_dir / 'digest.md').write_text(digest, encoding='utf-8')
    return session_dir


class TestViewAppComposition:
    @pytest.mark.asyncio
    async def test_has_required_widgets(self, tmp_path):
        session_dir = _make_session(tmp_path)
        app = ViewApp(session_dir=session_dir)
        async with app.run_test():
            assert app.query_one('#transcript-panel', TranscriptPanel)
            assert app.query_one('#digest-panel', DigestPanel)
            assert app.query_one('#status-bar', StatusBar)


class TestViewAppLoadsContent:
    @pytest.mark.asyncio
    async def test_loads_transcript(self, tmp_path):
        session_dir = _make_session(tmp_path, transcript='Line one\nLine two\n')
        app = ViewApp(session_dir=session_dir)
        async with app.run_test() as pilot:
            await pilot.pause()
            panel = app.query_one('#transcript-panel', TranscriptPanel)
            assert len(panel._all_text) == 2

    @pytest.mark.asyncio
    async def test_loads_digest(self, tmp_path):
        session_dir = _make_session(tmp_path, digest='# Summary\nKey points here.')
        app = ViewApp(session_dir=session_dir)
        async with app.run_test() as pilot:
            await pilot.pause()
            panel = app.query_one('#digest-panel', DigestPanel)
            assert 'Summary' in panel._current_markdown

    @pytest.mark.asyncio
    async def test_empty_session_no_crash(self, tmp_path):
        session_dir = _make_session(tmp_path)
        app = ViewApp(session_dir=session_dir)
        async with app.run_test() as pilot:
            await pilot.pause()
            # No crash = pass

    @pytest.mark.asyncio
    async def test_both_transcript_and_digest(self, tmp_path):
        session_dir = _make_session(tmp_path, transcript='Hello\nWorld\n', digest='# Notes\nSome notes.')
        app = ViewApp(session_dir=session_dir)
        async with app.run_test() as pilot:
            await pilot.pause()
            transcript_panel = app.query_one('#transcript-panel', TranscriptPanel)
            digest_panel = app.query_one('#digest-panel', DigestPanel)
            assert len(transcript_panel._all_text) == 2
            assert 'Notes' in digest_panel._current_markdown


class TestViewAppQuit:
    @pytest.mark.asyncio
    async def test_q_exits(self, tmp_path):
        from unittest.mock import patch

        session_dir = _make_session(tmp_path)
        app = ViewApp(session_dir=session_dir)
        async with app.run_test() as pilot:
            with patch.object(app, 'exit') as mock_exit:
                await pilot.press('q')
                await pilot.pause()
                mock_exit.assert_called_once()


class TestViewAppStatusBar:
    @pytest.mark.asyncio
    async def test_status_bar_shows_stopped(self, tmp_path):
        session_dir = _make_session(tmp_path)
        app = ViewApp(session_dir=session_dir)
        async with app.run_test() as pilot:
            await pilot.pause()
            bar = app.query_one('#status-bar', StatusBar)
            assert bar.stopped is True

    @pytest.mark.asyncio
    async def test_status_bar_hints(self, tmp_path):
        session_dir = _make_session(tmp_path)
        app = ViewApp(session_dir=session_dir)
        async with app.run_test() as pilot:
            await pilot.pause()
            bar = app.query_one('#status-bar', StatusBar)
            assert 'back' in bar.keybinding_hints
