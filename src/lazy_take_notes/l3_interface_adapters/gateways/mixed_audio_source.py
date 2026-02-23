"""Gateway: MixedAudioSource — composites microphone and system audio into one stream."""

from __future__ import annotations

import logging
import queue
import threading
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from lazy_take_notes.l2_use_cases.ports.audio_source import AudioSource

log = logging.getLogger('ltn.audio.mixed')


class MixedAudioSource:
    """Mix microphone and system audio from two AudioSource instances.

    Chunk sizes may differ between sources: sounddevice fires at device block size (~512
    samples / 32 ms) while system capture sources produce fixed 1600-sample (100 ms)
    chunks. To handle this, _sys_buf accumulates all available system chunks on every
    read() call (non-blocking drain) and dispenses exactly len(mic_chunk) samples per
    mix — equal duration, not equal chunk count. Mic chunk timing drives output cadence
    because mic data arrives more frequently; both sources contribute equally to the
    final audio.

    The 0.5 attenuation is an anti-clipping guard, not normalization. The two sources
    are not amplitude-normalized: mic level depends on input gain, system audio level
    depends on OS system volume. If they differ significantly, one will dominate.
    """

    def __init__(self, mic: AudioSource, system_audio: AudioSource) -> None:
        self._mic = mic
        self._system = system_audio
        self._mic_q: queue.Queue[np.ndarray] = queue.Queue()
        self._sys_q: queue.Queue[np.ndarray] = queue.Queue()
        # Rolling accumulation buffer for system audio chunks awaiting consumption.
        # Filled by draining _sys_q on every read(); consumed in equal-duration slices.
        self._sys_buf: np.ndarray = np.array([], dtype=np.float32)
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []

    def open(self, sample_rate: int, channels: int) -> None:
        log.info(
            'opening mixed source: mic=%s, system=%s',
            type(self._mic).__name__,
            type(self._system).__name__,
        )
        self._mic.open(sample_rate, channels)
        self._system.open(sample_rate, channels)
        self._stop.clear()
        self._threads = [
            threading.Thread(target=self._reader, args=(self._mic, self._mic_q), daemon=True),
            threading.Thread(target=self._reader, args=(self._system, self._sys_q), daemon=True),
        ]
        for t in self._threads:
            t.start()

    def _reader(self, src, dest: queue.Queue) -> None:
        while not self._stop.is_set():
            chunk = src.read(timeout=0.05)
            if chunk is not None:
                dest.put(chunk)

    def read(self, timeout: float = 0.1) -> np.ndarray | None:
        # Mic chunk timing drives output cadence (see class docstring).
        try:
            mic = self._mic_q.get(timeout=timeout)
        except queue.Empty:
            return None

        # Drain ALL available system chunks into the rolling buffer non-blocking.
        # get_nowait() is intentional: blocking here would stall the mic path and
        # cause _mic_q to accumulate unboundedly.
        while True:
            try:
                self._sys_buf = np.concatenate([self._sys_buf, self._sys_q.get_nowait()])
            except queue.Empty:
                break

        if len(self._sys_buf) == 0:
            return mic  # no system audio yet; pass mic through at full amplitude

        # Consume exactly len(mic) samples from the system buffer so both sides
        # cover the same time window regardless of their native chunk sizes.
        n = len(mic)
        if len(self._sys_buf) >= n:
            sys = self._sys_buf[:n]
            self._sys_buf = self._sys_buf[n:]
        else:
            # System buffer is shorter than mic chunk; zero-pad the tail.
            sys = np.pad(self._sys_buf, (0, n - len(self._sys_buf)))
            self._sys_buf = np.array([], dtype=np.float32)

        # Attenuate by 0.5 to prevent clipping (not normalization — see class docstring).
        return (mic + sys) * 0.5

    def close(self) -> None:
        log.debug('closing mixed source')
        self._stop.set()
        self._mic.close()
        self._system.close()
        for t in self._threads:
            t.join(timeout=2)
        self._threads = []
