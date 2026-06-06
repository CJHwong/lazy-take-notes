"""Tests for the SessionStatus entity."""

from __future__ import annotations

from lazy_take_notes.l1_entities.session_status import (
    DIGESTING,
    ERROR,
    LIVE_STATES,
    RECORDING,
    STOPPED,
    SessionStatus,
)


def test_constructs_with_defaults():
    status = SessionStatus(state=RECORDING, pid=123, started_at='t0', updated_at='t1')
    assert status.segment_count == 0
    assert status.digest_count == 0
    assert not status.error


def test_live_states_membership():
    # Active states imply a process should be running...
    assert RECORDING in LIVE_STATES
    assert DIGESTING in LIVE_STATES
    # ...terminal states do not.
    assert STOPPED not in LIVE_STATES
    assert ERROR not in LIVE_STATES
