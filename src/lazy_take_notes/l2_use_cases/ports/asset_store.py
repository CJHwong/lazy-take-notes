"""Port: shared model asset store."""

from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from lazy_take_notes.l1_entities.model_asset import AssetProgress, AssetState, ModelAsset


class AssetStore(Protocol):
    """Manages downloaded models and runtimes in one shared cache.

    ``resolve`` also satisfies the ``ModelResolver`` port. It accepts an asset
    id, a known model alias, or an ``hf://`` URI.
    """

    def list_assets(self) -> list[AssetState]:
        """Return every catalogued asset and its current state."""
        ...

    def status(self, asset_id: str) -> AssetState:
        """Return one asset's state. Raises for an unknown id."""
        ...

    def install(
        self,
        asset_id: str,
        on_progress: Callable[[AssetProgress], None] | None = None,
        cancel: threading.Event | None = None,
    ) -> Path:
        """Ensure the asset is present. Returns the entry path."""
        ...

    def remove(self, asset_id: str) -> int:
        """Delete an installed asset. Returns the freed bytes."""
        ...

    def resolve(self, name: str) -> str:
        """Resolve a model name to a local path. Raises on failure."""
        ...


class ModelAdminStore(Protocol):
    """The narrower store surface the settings screen needs."""

    @property
    def cache_root(self) -> Path: ...

    def catalog(self) -> list[ModelAsset]: ...
    def list_assets(self) -> list[AssetState]: ...
    def status(self, asset_id: str) -> AssetState: ...

    def install(
        self,
        asset_id: str,
        on_progress: Callable[[AssetProgress], None] | None = None,
        cancel: threading.Event | None = None,
    ) -> Path: ...

    def remove(self, asset_id: str) -> int: ...
