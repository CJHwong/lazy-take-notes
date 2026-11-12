"""Tests for the settings audio probe."""

from __future__ import annotations

import threading
import time

import numpy as np

from lazy_take_notes.l4_frameworks_and_drivers.messages import (
    AudioProbeDone,
    AudioProbeError,
    AudioProbeLevel,
    AudioProbePermission,
)
from lazy_take_notes.l4_frameworks_and_drivers.workers.audio_probe_worker import run_audio_probe


class FakeSource:
    def __init__(self, chunks, *, signal_seen: bool = True, awaiting: bool = False) -> None:
        self._chunks = list(chunks)
        self.system_signal_seen = signal_seen
        self.awaiting_permission = awaiting
        self.closed = False

    def open(self, _sample_rate, _channels) -> None:
        pass

    def read(self, timeout: float = 0.1):
        if not self._chunks:
            time.sleep(0.001)
            return None
        return self._chunks.pop(0)

    def close(self) -> None:
        self.closed = True


def run(mic, system, duration: float = 0.05):
    events: list = []
    stop = threading.Event()
    run_audio_probe(mic, system, post_message=events.append, stop=stop, duration=duration)
    return events


class TestAudioProbe:
    def test_reports_levels_and_allowed_permissions(self):
        mic = FakeSource([np.zeros(4), np.full(4, 0.2)])
        system = FakeSource([np.full(4, 0.3)])

        events = run(mic, system)

        levels = [event for event in events if isinstance(event, AudioProbeLevel)]
        assert {event.source for event in levels} == {'mic', 'system'}
        permissions = [event for event in events if isinstance(event, AudioProbePermission)]
        final = {event.source: event.state for event in permissions}
        assert final == {'mic': 'allowed', 'system': 'allowed'}
        assert isinstance(events[-1], AudioProbeDone)
        assert mic.closed and system.closed

    def test_zero_mic_is_no_signal(self):
        events = run(FakeSource([np.zeros(4)]), FakeSource([np.zeros(4)]))

        permissions = [event for event in events if isinstance(event, AudioProbePermission)]
        final = {event.source: event.state for event in permissions}
        assert final['mic'] == 'no_signal'

    def test_waiting_system_permission(self):
        system = FakeSource([np.zeros(4)], signal_seen=False, awaiting=True)

        events = run(FakeSource([np.zeros(4)]), system)

        permissions = [event for event in events if isinstance(event, AudioProbePermission)]
        system_states = [event.state for event in permissions if event.source == 'system']
        assert system_states[-1] == 'waiting'

    def test_open_failure_reports_error_and_done(self):
        class Broken(FakeSource):
            def open(self, _sample_rate, _channels) -> None:
                raise RuntimeError('no device')

        events = run(Broken([]), FakeSource([]))

        assert any(isinstance(event, AudioProbeError) for event in events)
        assert isinstance(events[-1], AudioProbeDone)
