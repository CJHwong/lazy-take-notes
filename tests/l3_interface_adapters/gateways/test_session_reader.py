"""Tests for the read-only session reader gateway."""

from __future__ import annotations

from pathlib import Path

from lazy_take_notes.l3_interface_adapters.gateways.session_reader import (
    SessionInfo,
    latest_session,
    list_sessions,
    read_notes,
    read_transcript,
)


def _create_session(
    base_dir: Path,
    name: str,
    *,
    has_notes: bool = False,
    transcript: str = 'Hello\nWorld\n',
    notes_text: str = '# Summary\nKey points.',
    transcript_name: str = 'transcript.txt',
    notes_name: str = 'notes.md',
) -> Path:
    session_dir = base_dir / name
    session_dir.mkdir()
    (session_dir / transcript_name).write_text(transcript, encoding='utf-8')
    if has_notes:
        (session_dir / notes_name).write_text(notes_text, encoding='utf-8')
    return session_dir


class TestListSessions:
    def test_empty_dir(self, tmp_path: Path):
        assert list_sessions(tmp_path) == []

    def test_nonexistent_dir(self, tmp_path: Path):
        assert list_sessions(tmp_path / 'nope') == []

    def test_finds_sessions_newest_first(self, tmp_path: Path):
        _create_session(tmp_path, '2026-02-20_120000')
        _create_session(tmp_path, '2026-02-21_120000', has_notes=True)

        result = list_sessions(tmp_path)

        assert [s.name for s in result] == ['2026-02-21_120000', '2026-02-20_120000']
        assert result[0].has_notes is True
        assert result[1].has_notes is False
        assert isinstance(result[0], SessionInfo)

    def test_ignores_dirs_without_transcript(self, tmp_path: Path):
        (tmp_path / 'empty_session').mkdir()
        _create_session(tmp_path, '2026-02-20_120000')

        assert len(list_sessions(tmp_path)) == 1

    def test_ignores_files(self, tmp_path: Path):
        (tmp_path / 'not_a_dir.txt').write_text('nope')
        _create_session(tmp_path, '2026-02-20_120000')

        assert len(list_sessions(tmp_path)) == 1

    def test_recognises_legacy_filenames(self, tmp_path: Path):
        _create_session(
            tmp_path,
            '2026-02-20_120000',
            has_notes=True,
            transcript_name='transcript_raw.txt',
            notes_name='digest.md',
        )

        result = list_sessions(tmp_path)

        assert len(result) == 1
        assert result[0].has_notes is True


class TestLatestSession:
    def test_none_when_empty(self, tmp_path: Path):
        assert latest_session(tmp_path) is None

    def test_returns_newest(self, tmp_path: Path):
        _create_session(tmp_path, '2026-02-20_120000')
        _create_session(tmp_path, '2026-02-21_120000')

        latest = latest_session(tmp_path)

        assert latest is not None
        assert latest.name == '2026-02-21_120000'


class TestReadTranscript:
    def test_returns_text(self, tmp_path: Path):
        session_dir = _create_session(tmp_path, 's', transcript='line one\n')
        assert read_transcript(session_dir) == 'line one\n'

    def test_reads_legacy_name(self, tmp_path: Path):
        session_dir = _create_session(tmp_path, 's', transcript='legacy\n', transcript_name='transcript_raw.txt')
        assert read_transcript(session_dir) == 'legacy\n'

    def test_none_when_absent(self, tmp_path: Path):
        empty = tmp_path / 'empty'
        empty.mkdir()
        assert read_transcript(empty) is None


class TestReadNotes:
    def test_returns_text(self, tmp_path: Path):
        session_dir = _create_session(tmp_path, 's', has_notes=True, notes_text='# Notes\n')
        assert read_notes(session_dir) == '# Notes\n'

    def test_reads_legacy_name(self, tmp_path: Path):
        session_dir = _create_session(tmp_path, 's', has_notes=True, notes_text='# Legacy\n', notes_name='digest.md')
        assert read_notes(session_dir) == '# Legacy\n'

    def test_none_when_absent(self, tmp_path: Path):
        session_dir = _create_session(tmp_path, 's')
        assert read_notes(session_dir) is None
