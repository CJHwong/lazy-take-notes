"""Probe both audio sources for the settings screen.

The probe runs in a worker thread. It reports a level per source and a
permission state. A missing microphone grant returns exact zeros with no error,
so ``no_signal`` is the only evidence the UI can show.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

import numpy as np

from lazy_take_notes.l1_entities.audio_constants import SAMPLE_RATE
from lazy_take_notes.l4_frameworks_and_drivers.messages import (
    AudioProbeDone,
    AudioProbeError,
    AudioProbeLevel,
    AudioProbePermission,
)

_PROBE_SECONDS = 15.0
_PERMISSION_INTERVAL = 0.5


def run_audio_probe(
    mic,
    system,
    *,
    post_message: Callable,
    stop: threading.Event,
    duration: float = _PROBE_SECONDS,
) -> None:
    """Open both sources and report levels until stopped or the time is up."""
    try:
        mic.open(SAMPLE_RATE, 1)
        system.open(SAMPLE_RATE, 1)
    except Exception as exc:  # noqa: BLE001 -- a failed open is a UI error, not a crash
        post_message(AudioProbeError(str(exc)))
        _close(mic, system)
        post_message(AudioProbeDone())
        return

    mic_seen = False
    started = time.monotonic()
    last_permission = 0.0
    try:
        while not stop.is_set() and time.monotonic() - started < duration:
            for name, source in (('mic', mic), ('system', system)):
                chunk = source.read(timeout=0.05)
                if chunk is None:
                    continue
                samples = np.asarray(chunk, dtype=np.float64)
                rms = float(np.sqrt(np.mean(samples**2))) if samples.size else 0.0
                if name == 'mic' and samples.any():
                    mic_seen = True
                post_message(AudioProbeLevel(name, rms))
            now = time.monotonic()
            if now - last_permission >= _PERMISSION_INTERVAL:
                last_permission = now
                post_message(AudioProbePermission('mic', 'allowed' if mic_seen else 'no_signal'))
                post_message(AudioProbePermission('system', _system_state(system)))
        post_message(AudioProbePermission('mic', 'allowed' if mic_seen else 'no_signal'))
        post_message(AudioProbePermission('system', _system_state(system)))
    finally:
        _close(mic, system)
        post_message(AudioProbeDone())


def _system_state(source) -> str:
    if getattr(source, 'awaiting_permission', False):
        return 'waiting'
    if getattr(source, 'system_signal_seen', True):
        return 'allowed'
    return 'no_signal'


def _close(mic, system) -> None:
    for source in (mic, system):
        try:
            source.close()
        except Exception:  # noqa: BLE001, S110 -- teardown must not mask the probe result
            pass
