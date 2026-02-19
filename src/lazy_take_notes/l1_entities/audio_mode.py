"""L1 entity: audio capture mode."""

from __future__ import annotations

import enum


class AudioMode(enum.Enum):
    MIC_ONLY = 'mic_only'
    SYSTEM_ONLY = 'system_only'
    MIX = 'mix'
