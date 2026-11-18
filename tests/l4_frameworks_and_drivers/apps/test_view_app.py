"""Tests for ViewApp — read-only session viewer TUI."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from lazy_take_notes.l4_frameworks_and_drivers.apps.view import ViewApp
from lazy_take_notes.l4_frameworks_and_drivers.widgets.digest_panel import DigestPanel
from lazy_take_notes.l4_frameworks_and_drivers.widgets.status_bar import StatusBar
from lazy_take_notes.l4_frameworks_and_drivers.widgets.transcript_panel import TranscriptPanel


def _make_session(tmp_path: Path, *, transcript: str = '', digest: str = '') -> Path:
    session_dir = tmp_path / '2026-02-20_120000'
    session_dir.mkdir()
    if transcript:
        (session_dir / 'transcript.txt').write_text(transcript, encoding='utf-8')
    if digest:
        (session_dir / 'notes.md').write_text(digest, encoding='utf-8')
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
            with patch.object(app, 'dismiss') as mock_exit:
                await pilot.press('q')
                await pilot.pause()
                mock_exit.assert_called_once()


class TestOpenSessionDir:
    @pytest.mark.asyncio
    async def test_o_opens_session_dir(self, tmp_path):
        from unittest.mock import patch

        session_dir = _make_session(tmp_path)
        app = ViewApp(session_dir=session_dir)
        async with app.run_test() as pilot:
            await pilot.pause()

            with patch('subprocess.Popen') as mock_popen:
                await pilot.press('o')
                await pilot.pause()
                mock_popen.assert_called_once()
                args = mock_popen.call_args[0][0]
                assert str(session_dir) in args

    @pytest.mark.asyncio
    async def test_o_uses_xdg_open_on_linux(self, tmp_path):
        from unittest.mock import patch

        session_dir = _make_session(tmp_path)
        app = ViewApp(session_dir=session_dir)
        async with app.run_test() as pilot:
            await pilot.pause()

            with (
                patch('lazy_take_notes.l4_frameworks_and_drivers.apps.view.sys') as mock_sys,
                patch('subprocess.Popen') as mock_popen,
            ):
                mock_sys.platform = 'linux'
                await pilot.press('o')
                await pilot.pause()
                mock_popen.assert_called_once()
                assert mock_popen.call_args[0][0][0] == 'xdg-open'

    @pytest.mark.asyncio
    async def test_o_uses_explorer_on_win32(self, tmp_path):
        from unittest.mock import patch

        session_dir = _make_session(tmp_path)
        app = ViewApp(session_dir=session_dir)
        async with app.run_test() as pilot:
            await pilot.pause()

            with (
                patch('lazy_take_notes.l4_frameworks_and_drivers.apps.view.sys') as mock_sys,
                patch('subprocess.Popen') as mock_popen,
            ):
                mock_sys.platform = 'win32'
                await pilot.press('o')
                await pilot.pause()
                mock_popen.assert_called_once()
                assert mock_popen.call_args[0][0][0] == 'explorer'


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


class TestViewAppTheme:
    @pytest.mark.asyncio
    async def test_on_mount_applies_saved_theme(self, tmp_path):
        with patch('lazy_take_notes.l4_frameworks_and_drivers.apps.view.load_theme', return_value='nord'):
            session_dir = _make_session(tmp_path)
            app = ViewApp(session_dir=session_dir)
            async with app.run_test() as pilot:
                await pilot.pause()
                assert app.app.theme == 'nord'


class TestViewAppFaults:
    @pytest.mark.asyncio
    async def test_markup_like_transcript_lines_do_not_crash(self, tmp_path):
        session_dir = _make_session(tmp_path, transcript='[00:00:01] say [/INST] now\n[bold]x[/b]\n')
        app = ViewApp(session_dir=session_dir)
        async with app.run_test() as pilot:
            await pilot.pause()
            panel = app.query_one('#transcript-panel', TranscriptPanel)
            assert panel._all_text == ['[00:00:01] say [/INST] now', '[bold]x[/b]']

    @pytest.mark.asyncio
    async def test_undecodable_transcript_shows_error(self, tmp_path):
        session_dir = _make_session(tmp_path, digest='# ok')
        (session_dir / 'transcript.txt').write_bytes(b'\xff\xfe\xfa bad')
        app = ViewApp(session_dir=session_dir)
        async with app.run_test() as pilot:
            await pilot.pause()
            assert any('transcript' in str(n.message).lower() for n in app.app._notifications)
            assert 'ok' in app.query_one('#digest-panel', DigestPanel)._current_markdown

    @pytest.mark.asyncio
    async def test_unreadable_digest_shows_error(self, tmp_path):
        session_dir = _make_session(tmp_path, transcript='hi\n', digest='# ok')
        app = ViewApp(session_dir=session_dir)
        real_read = Path.read_text

        def _read(path, *args, **kwargs):
            if path.name == 'notes.md':
                raise PermissionError('denied')
            return real_read(path, *args, **kwargs)

        with patch.object(Path, 'read_text', _read):
            async with app.run_test() as pilot:
                await pilot.pause()
                assert any('notes' in str(n.message).lower() for n in app.app._notifications)

    @pytest.mark.asyncio
    async def test_open_failure_notifies(self, tmp_path):
        session_dir = _make_session(tmp_path)
        app = ViewApp(session_dir=session_dir)
        async with app.run_test() as pilot:
            with patch('subprocess.Popen', side_effect=FileNotFoundError('xdg-open')):
                await pilot.press('o')
                await pilot.pause()
            assert any('open' in str(n.message).lower() for n in app.app._notifications)


class TestCopyTheDigest:
    @pytest.mark.asyncio
    async def test_tab_then_c_copies_the_digest(self, tmp_path):
        session_dir = _make_session(tmp_path, transcript='[00:00:01] hello\n', digest='# Notes\nbody')
        app = ViewApp(session_dir=session_dir)
        with patch('pyperclip.copy') as copy:
            async with app.run_test(size=(140, 40)) as pilot:
                await pilot.pause()
                await pilot.press('tab')
                await pilot.press('c')
                await pilot.pause()
                assert app.app.focused is app.query_one('#digest-panel', DigestPanel)
                assert copy.call_args.args[0].startswith('# Notes')


class TestAudioOnlySession:
    @pytest.mark.asyncio
    async def test_says_where_the_audio_is(self, tmp_path):
        session_dir = _make_session(tmp_path)
        (session_dir / 'recording.wav').write_bytes(b'RIFF')
        app = ViewApp(session_dir=session_dir)
        async with app.run_test(size=(140, 40)) as pilot:
            await pilot.pause()
            # The panel wraps the line at its own width.
            shown = ' '.join(
                ' '.join(line.text for line in app.query_one('#transcript-panel', TranscriptPanel).lines).split()
            )
            assert 'No transcript. The audio is saved in this folder: press o to open it.' in shown
