"""Tests for file persistence gateway."""

from __future__ import annotations

from pathlib import Path

from lazy_take_notes.l1_entities.transcript import TranscriptSegment
from lazy_take_notes.l3_interface_adapters.gateways.file_persistence import FilePersistenceGateway


def _make_segments() -> list[TranscriptSegment]:
    return [
        TranscriptSegment(text='Hello world', wall_start=0.0, wall_end=1.5),
        TranscriptSegment(text='Second line', wall_start=2.0, wall_end=3.0),
    ]


SAMPLE_MARKDOWN = """\
## Current Topic
Discussing REST vs GraphQL

## Completed Topics
### Intro
**Resolution:** Done
- Greeted

## Open Questions
- Which one?

## Suggested Actions
- **Benchmark both** (Alice): Need data

## Parking Lot
- CI/CD pipeline
"""


class TestSaveTranscriptLines:
    def test_creates_file(self, tmp_output_dir: Path):
        gw = FilePersistenceGateway(tmp_output_dir)
        segs = _make_segments()
        path = gw.save_transcript_lines(segs, append=False)
        assert path.exists()
        content = path.read_text(encoding='utf-8')
        assert '[00:00:00] Hello world' in content
        assert '[00:00:02] Second line' in content

    def test_append_mode(self, tmp_output_dir: Path):
        gw = FilePersistenceGateway(tmp_output_dir)
        segs1 = [TranscriptSegment(text='First', wall_start=0, wall_end=1)]
        segs2 = [TranscriptSegment(text='Second', wall_start=60, wall_end=61)]
        gw.save_transcript_lines(segs1, append=False)
        gw.save_transcript_lines(segs2, append=True)
        content = (tmp_output_dir / 'transcript_raw.txt').read_text(encoding='utf-8')
        assert 'First' in content
        assert 'Second' in content


class TestSaveDigestMd:
    def test_creates_markdown(self, tmp_output_dir: Path):
        gw = FilePersistenceGateway(tmp_output_dir)
        path = gw.save_digest_md(SAMPLE_MARKDOWN, 3)
        assert path.exists()
        content = path.read_text(encoding='utf-8')
        assert '# Digest #3' in content
        assert 'Discussing REST vs GraphQL' in content

    def test_overwrites_on_second_call(self, tmp_output_dir: Path):
        gw = FilePersistenceGateway(tmp_output_dir)
        gw.save_digest_md('First version', 1)
        gw.save_digest_md('Second version', 2)
        content = (tmp_output_dir / 'digest.md').read_text(encoding='utf-8')
        assert 'Second version' in content
        assert 'First version' not in content


class TestSaveSessionContext:
    def test_creates_file(self, tmp_output_dir: Path):
        gw = FilePersistenceGateway(tmp_output_dir)
        path = gw.save_session_context('Speaker A = Alice\nFix: "rec" → "wreck"')
        assert path.name == 'session_context.txt'
        assert path.exists()
        assert 'Speaker A = Alice' in path.read_text(encoding='utf-8')

    def test_overwrites_on_second_call(self, tmp_output_dir: Path):
        gw = FilePersistenceGateway(tmp_output_dir)
        gw.save_session_context('first')
        gw.save_session_context('second')
        assert (tmp_output_dir / 'session_context.txt').read_text(encoding='utf-8') == 'second'


class TestSaveHistory:
    def test_creates_numbered_file(self, tmp_output_dir: Path):
        gw = FilePersistenceGateway(tmp_output_dir)
        path = gw.save_history(SAMPLE_MARKDOWN, 3)
        assert path.name == 'digest_003.md'
        assert path.exists()

    def test_final_suffix(self, tmp_output_dir: Path):
        gw = FilePersistenceGateway(tmp_output_dir)
        path = gw.save_history(SAMPLE_MARKDOWN, 3, is_final=True)
        assert path.name == 'digest_003_final.md'

    def test_creates_history_dir(self, tmp_output_dir: Path):
        gw = FilePersistenceGateway(tmp_output_dir)
        gw.save_history(SAMPLE_MARKDOWN, 3)
        assert (tmp_output_dir / 'history').is_dir()

    def test_content_has_header(self, tmp_output_dir: Path):
        gw = FilePersistenceGateway(tmp_output_dir)
        path = gw.save_history(SAMPLE_MARKDOWN, 3)
        content = path.read_text(encoding='utf-8')
        assert '# Digest #3' in content
