"""Gateway: MixedAudioSource — composites two AudioSource instances into a single stream."""

from __future__ import annotations

import queue
import threading

import numpy as np

from lazy_take_notes.l2_use_cases.ports.audio_source import AudioSource


class MixedAudioSource:
    """Mixes two AudioSource instances by summing their PCM output, clamped to [-1, 1]."""

    def __init__(self, primary: AudioSource, secondary: AudioSource) -> None:
        self._primary = primary
        self._secondary = secondary
        self._q1: queue.Queue[np.ndarray] = queue.Queue()
        self._q2: queue.Queue[np.ndarray] = queue.Queue()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []

    def open(self, sample_rate: int, channels: int) -> None:
        self._primary.open(sample_rate, channels)
        self._secondary.open(sample_rate, channels)
        self._stop.clear()
        self._threads = [
            threading.Thread(target=self._reader, args=(self._primary, self._q1), daemon=True),
            threading.Thread(target=self._reader, args=(self._secondary, self._q2), daemon=True),
        ]
        for t in self._threads:
            t.start()

    def _reader(self, src: AudioSource, dest: queue.Queue) -> None:
        while not self._stop.is_set():
            chunk = src.read(timeout=0.05)
            if chunk is not None:
                dest.put(chunk)

    def read(self, timeout: float = 0.1) -> np.ndarray | None:
        try:
            a = self._q1.get(timeout=timeout)
        except queue.Empty:
            return None
        try:
            b = self._q2.get_nowait()
            # Pad the shorter chunk with zeros so neither source loses audio.
            length = max(len(a), len(b))
            if len(a) < length:
                a = np.pad(a, (0, length - len(a)))
            if len(b) < length:
                b = np.pad(b, (0, length - len(b)))
            # Attenuate by 0.5 to stay within [-1, 1] without clipping.
            # This also keeps the combined "silence" level low enough that
            # the TranscribeAudioUseCase silence trigger still fires correctly.
            return (a + b) * 0.5
        except queue.Empty:
            return a  # only primary has data; pass it through

    def close(self) -> None:
        self._stop.set()
        self._primary.close()
        self._secondary.close()
        for t in self._threads:
            t.join(timeout=2)
        self._threads = []
