"""Gateway: sounddevice audio source — implements AudioSource port."""

from __future__ import annotations

import queue

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16000


class SounddeviceAudioSource:
    """Wraps sounddevice.InputStream to provide audio chunks."""

    def __init__(self) -> None:
        self._stream: sd.InputStream | None = None
        self._queue: queue.Queue[np.ndarray] = queue.Queue()

    def open(self, sample_rate: int = SAMPLE_RATE, channels: int = 1) -> None:
        def _callback(indata, frames, time_info, status):
            self._queue.put(indata.copy())

        self._stream = sd.InputStream(
            samplerate=sample_rate,
            channels=channels,
            dtype='float32',
            callback=_callback,
        )
        self._stream.start()

    def read(self, timeout: float = 0.1) -> np.ndarray | None:
        try:
            return self._queue.get(timeout=timeout).flatten()
        except queue.Empty:
            return None

    def drain(self) -> np.ndarray | None:
        """Read all available data from the queue."""
        chunks = []
        while not self._queue.empty():
            try:
                chunks.append(self._queue.get_nowait().flatten())
            except queue.Empty:
                break
        return np.concatenate(chunks) if chunks else None

    def close(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
