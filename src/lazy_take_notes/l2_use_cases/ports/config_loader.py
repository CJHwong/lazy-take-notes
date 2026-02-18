"""Port: configuration loader."""

from __future__ import annotations

from typing import Protocol

from lazy_take_notes.l1_entities.config import AppConfig


class ConfigLoader(Protocol):
    """Abstract configuration loader."""

    def load(
        self,
        config_path: str | None = None,
        overrides: dict | None = None,
    ) -> AppConfig:
        """Load and validate configuration, merging overrides."""
        ...
