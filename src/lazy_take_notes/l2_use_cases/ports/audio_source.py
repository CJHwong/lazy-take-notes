"""Port: audio capture source."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class AudioSource(Protocol):
    """Abstract audio input stream."""

    def open(self, sample_rate: int, channels: int) -> None:
        """Open the audio stream."""
        ...

    def read(self, timeout: float) -> np.ndarray | None:
        """Read a chunk of audio. Returns None on timeout."""
        ...

    def close(self) -> None:
        """Close the audio stream."""
        ...
