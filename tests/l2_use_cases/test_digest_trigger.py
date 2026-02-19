"""Tests for digest trigger evaluation."""

import time

from lazy_take_notes.l1_entities.digest_state import DigestState
from lazy_take_notes.l2_use_cases.digest_use_case import should_trigger_digest


class TestShouldTriggerDigest:
    def test_not_enough_lines(self):
        state = DigestState(last_digest_time=0.0)
        state.buffer = ['line1', 'line2']
        assert not should_trigger_digest(state, min_lines=5, min_interval=0)

    def test_not_enough_time(self):
        # 6 lines: above min_lines=5 but below max_lines=100 — interval gate blocks it
        state = DigestState(last_digest_time=time.monotonic())
        state.buffer = ['line'] * 6
        assert not should_trigger_digest(state, min_lines=5, min_interval=9999, max_lines=100)

    def test_should_trigger(self):
        state = DigestState(last_digest_time=0.0)
        state.buffer = ['line'] * 20
        assert should_trigger_digest(state, min_lines=5, min_interval=0)

    def test_max_lines_bypasses_interval(self):
        # Buffer at 10 lines with recent digest (interval not elapsed) and max_lines=10
        state = DigestState(last_digest_time=time.monotonic())
        state.buffer = ['line'] * 10
        assert should_trigger_digest(state, min_lines=5, min_interval=9999, max_lines=10)

    def test_max_lines_default_is_2x_min_lines(self):
        # Default max_lines = 2 * min_lines = 10; buffer at 10 should force-trigger
        state = DigestState(last_digest_time=time.monotonic())
        state.buffer = ['line'] * 10
        assert should_trigger_digest(state, min_lines=5, min_interval=9999)

    def test_below_max_lines_still_requires_interval(self):
        # Buffer at 6 lines (> min_lines=5, < max_lines=10) with recent digest
        state = DigestState(last_digest_time=time.monotonic())
        state.buffer = ['line'] * 6
        assert not should_trigger_digest(state, min_lines=5, min_interval=9999, max_lines=10)
