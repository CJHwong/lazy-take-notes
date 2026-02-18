"""Tests for digest trigger evaluation."""

import time

from lazy_take_notes.l1_entities.digest_state import DigestState
from lazy_take_notes.l3_interface_adapters.controllers.session_controller import should_trigger_digest


class TestShouldTriggerDigest:
    def test_not_enough_lines(self):
        state = DigestState(last_digest_time=0.0)
        state.buffer = ['line1', 'line2']
        assert not should_trigger_digest(state, min_lines=5, min_interval=0)

    def test_not_enough_time(self):
        state = DigestState(last_digest_time=time.monotonic())
        state.buffer = ['line'] * 20
        assert not should_trigger_digest(state, min_lines=5, min_interval=9999)

    def test_should_trigger(self):
        state = DigestState(last_digest_time=0.0)
        state.buffer = ['line'] * 20
        assert should_trigger_digest(state, min_lines=5, min_interval=0)
