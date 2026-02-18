"""Tests for TranscriptSegment entity."""

from lazy_take_notes.l1_entities.transcript import TranscriptSegment


class TestTranscriptSegment:
    def test_creation(self):
        seg = TranscriptSegment(text='Hello', wall_start=1.0, wall_end=2.0)
        assert seg.text == 'Hello'
        assert seg.wall_start == 1.0
        assert seg.wall_end == 2.0

    def test_empty_text(self):
        seg = TranscriptSegment(text='', wall_start=0.0, wall_end=0.0)
        assert seg.text == ''
