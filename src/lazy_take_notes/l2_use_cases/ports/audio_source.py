"""Port: audio capture source."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class AudioSource(Protocol):
    """Abstract audio input stream."""

    mic_muted: bool
    """When True, microphone contribution is silenced. Sources that do not
    support muting should accept the attribute but may ignore it."""

    def open(self, sample_rate: int, channels: int) -> None:
        """Open the audio stream."""
        ...

    def read(self, timeout: float) -> np.ndarray | None:
        """Read a chunk of audio. Returns None on timeout."""
        ...

    def drain(self) -> None:
        """Discard any buffered audio. No-op for sources without internal buffers."""
        ...

    def close(self) -> None:
        """Close the audio stream."""
        ...
