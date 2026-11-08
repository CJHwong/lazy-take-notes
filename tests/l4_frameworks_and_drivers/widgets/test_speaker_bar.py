"""Tests for the speaker bar formatting."""

from __future__ import annotations

from lazy_take_notes.l4_frameworks_and_drivers.widgets.speaker_bar import format_spoken_time


class TestFormatSpokenTime:
    def test_seconds_under_a_minute(self):
        assert format_spoken_time(42) == '42s'

    def test_minutes_and_seconds(self):
        assert format_spoken_time(298) == '4m58s'

    def test_exact_minute(self):
        assert format_spoken_time(120) == '2m0s'

    def test_hours(self):
        assert format_spoken_time(3725) == '1h2m'
