"""Port: configuration loader."""

from __future__ import annotations

from typing import Protocol


class ConfigLoader(Protocol):  # pragma: no cover -- abstract Protocol; never instantiated directly
    """Abstract configuration loader."""

    def load(
        self,
        config_path: str | None = None,
        overrides: dict | None = None,
    ) -> dict:
        """Load configuration from source, returning raw dict before validation."""
        ...
