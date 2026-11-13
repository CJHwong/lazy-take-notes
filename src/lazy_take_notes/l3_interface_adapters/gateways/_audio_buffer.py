"""Bound queued audio by duration so slow consumers cannot retain a session."""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections import deque

import numpy as np

log = logging.getLogger('ltn.audio.buffer')


class AudioBuffer:
    def __init__(self, *, max_frames: int, sample_rate: int, label: str = 'audio') -> None:
        if max_frames <= 0 or sample_rate <= 0:
            raise ValueError('Audio buffer frame limit and sample rate must be positive')
        self._label = label
        self._max_frames = max_frames
        self._sample_rate = sample_rate
        self._chunks: deque[np.ndarray] = deque()
        self._condition = threading.Condition()
        self.frames = 0
        self.dropped_frames = 0
        self._last_drop_report = float('-inf')
        self._reported_dropped_frames = 0

    def put(self, samples: np.ndarray) -> None:
        if not len(samples):
            return
        trimmed = max(0, len(samples) - self._max_frames)
        if trimmed:
            samples = samples[-self._max_frames :].copy()
        with self._condition:
            dropped = trimmed
            while self._chunks and self.frames + len(samples) > self._max_frames:
                removed = self._chunks.popleft()
                self.frames -= len(removed)
                dropped += len(removed)
            self._chunks.append(samples)
            self.frames += len(samples)
            self.dropped_frames += dropped
            self._condition.notify()

    def _report_drop(self) -> None:
        if self.dropped_frames == self._reported_dropped_frames:
            return
        current_time = time.monotonic()
        if current_time - self._last_drop_report >= 1:
            self._last_drop_report = current_time
            self._reported_dropped_frames = self.dropped_frames
            log.warning(
                '%s buffer dropped %d frames (%.3fs total) to keep recent audio',
                self._label,
                self.dropped_frames,
                self.dropped_frames / self._sample_rate,
            )

    def get(self, timeout: float) -> np.ndarray:
        with self._condition:
            if not self._condition.wait_for(lambda: bool(self._chunks), timeout=max(0, timeout)):
                raise queue.Empty
            samples = self._chunks.popleft()
            self.frames -= len(samples)
        self._report_drop()
        return samples

    def get_nowait(self) -> np.ndarray:
        return self.get(0)

    def clear(self) -> None:
        with self._condition:
            self._chunks.clear()
            self.frames = 0

    def empty(self) -> bool:
        with self._condition:
            return not self._chunks

    def qsize(self) -> int:
        with self._condition:
            return len(self._chunks)
