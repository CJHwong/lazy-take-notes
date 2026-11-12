"""Tests for the cache-backed model resolver."""

from __future__ import annotations

from pathlib import Path

import pytest

from lazy_take_notes.l1_entities.errors import ModelResolutionError
from lazy_take_notes.l1_entities.model_asset import AssetProgress
from lazy_take_notes.l3_interface_adapters.gateways.cached_model_store import CachedModelStore
from lazy_take_notes.l3_interface_adapters.gateways.hf_model_resolver import HfModelResolver


class StubStore:
    """Minimal store that records calls and reports fixed progress."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.installed: list[str] = []
        self.resolved: list[str] = []

    def install(self, asset_id, on_progress=None, cancel=None) -> Path:
        self.installed.append(asset_id)
        if on_progress is not None:
            on_progress(AssetProgress(downloaded=5, total=10))
        return self.path

    def resolve(self, name: str) -> str:
        self.resolved.append(name)
        return name


class TestHfModelResolver:
    def test_alias_installs_the_cataloged_asset(self, tmp_path: Path):
        store = StubStore(tmp_path / 'model.bin')
        resolver = HfModelResolver(store=store)

        result = resolver.resolve('large-v3-turbo-q8_0')

        assert result == str(tmp_path / 'model.bin')
        assert store.installed == ['whisper-large-v3-turbo-q8_0']

    def test_progress_maps_to_percent(self, tmp_path: Path):
        calls: list[int] = []
        store = StubStore(tmp_path / 'model.bin')

        HfModelResolver(on_progress=calls.append, store=store).resolve('large-v3-turbo-q8_0')

        assert calls == [50]

    def test_custom_name_uses_store_resolve(self, tmp_path: Path):
        store = StubStore(tmp_path / 'model.bin')
        resolver = HfModelResolver(store=store)

        assert resolver.resolve('some-custom-model') == 'some-custom-model'
        assert store.resolved == ['some-custom-model']

    def test_absolute_path_exists(self, tmp_path: Path):
        model = tmp_path / 'model.bin'
        model.touch()
        resolver = HfModelResolver(store=CachedModelStore(tmp_path / 'cache', assets={}))

        assert resolver.resolve(str(model)) == str(model)

    def test_absolute_path_missing_raises(self, tmp_path: Path):
        resolver = HfModelResolver(store=CachedModelStore(tmp_path / 'cache', assets={}))

        with pytest.raises(ModelResolutionError, match='not found'):
            resolver.resolve(str(tmp_path / 'nope.bin'))

    def test_passthrough_name(self, tmp_path: Path):
        resolver = HfModelResolver(store=CachedModelStore(tmp_path / 'cache', assets={}))

        assert resolver.resolve('some-custom-model') == 'some-custom-model'

    def test_malformed_hf_uri_raises(self, tmp_path: Path):
        resolver = HfModelResolver(store=CachedModelStore(tmp_path / 'cache', assets={}))

        with pytest.raises(ModelResolutionError, match='Malformed hf:// URI'):
            resolver.resolve('hf://')
