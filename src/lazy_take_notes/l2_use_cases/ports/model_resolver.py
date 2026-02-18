"""Port: whisper model resolution."""

from __future__ import annotations

from typing import Protocol


class ModelResolver(Protocol):
    """Abstract model resolver — maps model name to local file path."""

    def resolve(self, model_name: str) -> str:
        """Resolve a model name to a usable local path. Raises on failure."""
        ...
