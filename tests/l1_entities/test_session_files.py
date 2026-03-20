"""Tests for SessionFile dataclass and legacy filename resolution."""

from __future__ import annotations

from pathlib import Path

from lazy_take_notes.l1_entities.session_files import (
    CONTEXT,
    DEBUG_LOG,
    NOTES,
    TRANSCRIPT,
    SessionFile,
)


class TestSessionFile:
    def test_resolve_current_name(self, tmp_path: Path):
        sf = SessionFile('transcript.txt', 'transcript_raw.txt')
        (tmp_path / 'transcript.txt').write_text('hello')
        assert sf.resolve(tmp_path) == tmp_path / 'transcript.txt'

    def test_resolve_legacy_fallback(self, tmp_path: Path):
        sf = SessionFile('transcript.txt', 'transcript_raw.txt')
        (tmp_path / 'transcript_raw.txt').write_text('hello')
        assert sf.resolve(tmp_path) == tmp_path / 'transcript_raw.txt'

    def test_resolve_prefers_current_over_legacy(self, tmp_path: Path):
        sf = SessionFile('notes.md', 'digest.md')
        (tmp_path / 'notes.md').write_text('current')
        (tmp_path / 'digest.md').write_text('legacy')
        assert sf.resolve(tmp_path) == tmp_path / 'notes.md'

    def test_resolve_returns_none_when_neither_exists(self, tmp_path: Path):
        sf = SessionFile('notes.md', 'digest.md')
        assert sf.resolve(tmp_path) is None

    def test_frozen(self):
        sf = SessionFile('a.txt', 'b.txt')
        import pytest

        with pytest.raises(AttributeError):
            sf.name = 'c.txt'  # type: ignore[misc]


class TestModuleConstants:
    def test_transcript(self):
        assert TRANSCRIPT.name == 'transcript.txt'
        assert TRANSCRIPT.legacy == 'transcript_raw.txt'

    def test_notes(self):
        assert NOTES.name == 'notes.md'
        assert NOTES.legacy == 'digest.md'

    def test_context(self):
        assert CONTEXT.name == 'context.txt'
        assert CONTEXT.legacy == 'session_context.txt'

    def test_debug_log(self):
        assert DEBUG_LOG.name == 'debug.log'
        assert DEBUG_LOG.legacy == 'ltn_debug.log'
