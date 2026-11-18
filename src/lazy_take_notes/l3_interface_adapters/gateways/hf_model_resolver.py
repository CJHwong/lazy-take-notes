"""Gateway: model resolver backed by the shared asset cache."""

from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from lazy_take_notes.l1_entities.model_asset import AssetProgress
from lazy_take_notes.l3_interface_adapters.gateways.cached_model_store import CachedModelStore
from lazy_take_notes.l3_interface_adapters.gateways.model_catalog import asset_for_model


class ResolverStore(Protocol):
    """The store surface the resolver needs."""

    def install(
        self,
        asset_id: str,
        on_progress: Callable[[AssetProgress], None] | None = None,
        cancel: threading.Event | None = None,
    ) -> Path: ...

    def resolve(self, name: str, cancel: threading.Event | None = None) -> str: ...


class HfModelResolver:
    """Resolve a model name through the shared cache.

    The resolver keeps the old interface for the worker and plugin paths. The
    download, cache, and verification work lives in ``CachedModelStore``.
    """

    def __init__(
        self,
        on_progress: Callable[[int], None] | None = None,
        *,
        store: ResolverStore | None = None,
    ) -> None:
        self._on_progress = on_progress
        self._store = store or CachedModelStore()
        # Set by the caller to stop a download; the partial file is kept for the next try.
        self.cancel: threading.Event | None = None
        # Set by a caller that shows bytes, speed and time left. It replaces the percent callback,
        # which stays an int for plugin resolvers.
        self.on_detail: Callable[[AssetProgress], None] | None = None

    def resolve(self, model_name: str) -> str:
        asset = asset_for_model(model_name)
        if asset is not None:
            return str(self._store.install(asset.id, self._report, self.cancel))
        return self._store.resolve(model_name, cancel=self.cancel)

    def _report(self, progress: AssetProgress) -> None:
        if self.on_detail is not None:
            self.on_detail(progress)
            return
        if self._on_progress is None or progress.total <= 0:
            return
        self._on_progress(min(int(progress.downloaded * 100 / progress.total), 100))
