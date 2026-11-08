"""Tests for timestamp to speaker attribution."""

from __future__ import annotations

from lazy_take_notes.l1_entities.speaker_turn import SpeakerTurn
from lazy_take_notes.l2_use_cases.speaker_attribution import speaker_at


class TestSpeakerAt:
    def test_returns_the_speaker_covering_the_time(self):
        turns = [SpeakerTurn(0.0, 2.0, 1), SpeakerTurn(2.0, 5.0, 2)]

        assert speaker_at(turns, 1.5) == 1
        assert speaker_at(turns, 2.0) == 2

    def test_returns_none_outside_every_turn(self):
        turns = [SpeakerTurn(0.0, 2.0, 1)]

        assert speaker_at(turns, 9.0) is None

    def test_returns_none_for_an_empty_turn_list(self):
        assert speaker_at([], 1.0) is None

    def test_overlap_prefers_the_turn_that_started_last(self):
        # A backchannel usually starts after the main speaker began.
        turns = [SpeakerTurn(0.0, 10.0, 1), SpeakerTurn(4.0, 4.5, 2)]

        assert speaker_at(turns, 4.2) == 2
        assert speaker_at(turns, 6.0) == 1


class TestSpeakerForSpan:
    def test_span_overlap_wins_over_start_instant(self):
        from lazy_take_notes.l2_use_cases.speaker_attribution import speaker_for_span

        turns = [SpeakerTurn(0.0, 2.0, 1), SpeakerTurn(2.0, 6.0, 2)]

        # Starts in a 0.1 s gap, but almost all of the line is speaker 2.
        assert speaker_for_span(turns, 1.95, 4.0) == 2

    def test_line_without_overlap_uses_a_nearby_turn(self):
        from lazy_take_notes.l2_use_cases.speaker_attribution import speaker_for_span

        turns = [SpeakerTurn(0.0, 2.0, 1)]

        assert speaker_for_span(turns, 2.4, 3.0) == 1

    def test_line_far_from_every_turn_stays_unlabeled(self):
        from lazy_take_notes.l2_use_cases.speaker_attribution import speaker_for_span

        turns = [SpeakerTurn(0.0, 2.0, 1)]

        assert speaker_for_span(turns, 30.0, 31.0) is None

    def test_zero_length_span_is_handled(self):
        from lazy_take_notes.l2_use_cases.speaker_attribution import speaker_for_span

        turns = [SpeakerTurn(1.0, 2.0, 3)]

        assert speaker_for_span(turns, 1.5, 1.5) == 3
