"""Tests for the Models page."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest
from textual.widgets import Button

import lazy_take_notes.l3_interface_adapters.gateways.yaml_config_loader as loader_mod
import lazy_take_notes.l3_interface_adapters.gateways.yaml_config_writer as writer_mod
from lazy_take_notes.l1_entities.model_asset import AssetProgress, AssetState, AssetStatus, ModelAsset
from lazy_take_notes.l4_frameworks_and_drivers.apps.config import ConfigApp
from lazy_take_notes.l4_frameworks_and_drivers.apps.config.models_tab import ModelChoiceRow

ASSET = ModelAsset(
    id='tiny',
    label='Tiny model',
    purpose='Test',
    group='transcription',
    kind='hf',
    url='https://example.test/tiny.bin',
    size=100,
    sha256='0' * 64,
    subdir='whisper/tiny',
    entry='tiny.bin',
    hf_repo='owner/repo',
)


class FakeStore:
    def __init__(self, status: AssetStatus) -> None:
        self._state = AssetState(ASSET, status)
        self.installs: list[str] = []
        self.removes: list[str] = []
        self.cache_root = Path('/tmp/ltn-cache')

    def catalog(self) -> list[ModelAsset]:
        return [ASSET]

    def list_assets(self) -> list[AssetState]:
        return [self._state]

    def status(self, asset_id: str) -> AssetState:
        return self._state

    def install(
        self,
        asset_id: str,
        on_progress: Callable[[AssetProgress], None] | None = None,
        cancel: threading.Event | None = None,
    ) -> Path:
        self.installs.append(asset_id)
        if on_progress is not None:
            on_progress(AssetProgress(50, 100, 1_000_000))
        self._state = replace(self._state, status=AssetStatus.READY, installed_bytes=100, path='/tmp/tiny.bin')
        return Path('/tmp/tiny.bin')

    def remove(self, asset_id: str) -> int:
        self.removes.append(asset_id)
        self._state = replace(self._state, status=AssetStatus.MISSING, installed_bytes=0, path=None)
        return 100


@pytest.fixture
def app_factory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    config_path = tmp_path / 'config.yaml'
    config_path.write_text('{}')
    monkeypatch.setattr(loader_mod, 'DEFAULT_CONFIG_PATHS', [config_path])
    monkeypatch.setattr(writer_mod, 'DEFAULT_CONFIG_PATHS', [config_path])
    monkeypatch.setattr(writer_mod, 'CONFIG_DIR', tmp_path)

    def build(status: AssetStatus) -> tuple[ConfigApp, FakeStore]:
        store = FakeStore(status)
        return ConfigApp(asset_store=store), store

    return build


class TestModelsTab:
    async def test_rows_render_from_the_catalog(self, app_factory):
        app, _store = app_factory(AssetStatus.READY)

        async with app.run_test(size=(140, 50)) as pilot:
            await pilot.pause()
            await app.workers.wait_for_complete()
            row = app.query_one('#choice-tiny', ModelChoiceRow)
            assert row.asset is ASSET
            assert app.query_one('#status-tiny').render().plain == 'Ready'

    async def test_missing_asset_shows_download(self, app_factory):
        app, _store = app_factory(AssetStatus.MISSING)

        async with app.run_test(size=(140, 50)) as pilot:
            await pilot.pause()
            await app.workers.wait_for_complete()
            assert app.query_one('#download-tiny').display is True
            assert app.query_one('#remove-tiny').display is False

    async def test_ready_asset_shows_remove(self, app_factory):
        app, _store = app_factory(AssetStatus.READY)

        async with app.run_test(size=(140, 50)) as pilot:
            await pilot.pause()
            await app.workers.wait_for_complete()
            assert app.query_one('#download-tiny').display is False
            assert app.query_one('#remove-tiny').display is True

    async def test_download_installs_and_updates_the_row(self, app_factory):
        app, store = app_factory(AssetStatus.MISSING)

        async with app.run_test(size=(140, 50)) as pilot:
            await pilot.pause()
            await app.workers.wait_for_complete()
            app.query_one('#download-tiny', Button).press()
            await pilot.pause()
            await app.workers.wait_for_complete()

            assert store.installs == ['tiny']
            assert app.query_one('#status-tiny').render().plain == 'Ready'

    async def test_remove_asks_for_confirmation(self, app_factory):
        app, store = app_factory(AssetStatus.READY)

        async with app.run_test(size=(140, 50)) as pilot:
            await pilot.pause()
            await app.workers.wait_for_complete()
            app.query_one('#remove-tiny', Button).press()
            await pilot.pause()
            await pilot.pause()
            app.app.screen.query_one('#discard-confirm', Button).press()
            await pilot.pause()
            await app.workers.wait_for_complete()

            assert store.removes == ['tiny']
            assert app.query_one('#status-tiny').render().plain == 'Missing'

    async def test_default_model_row_shows_size_and_purpose(self, app_factory):
        app, _store = app_factory(AssetStatus.MISSING)

        async with app.run_test(size=(140, 50)) as pilot:
            await pilot.pause()
            await app.workers.wait_for_complete()
            row = app.query_one('#choice-tiny', ModelChoiceRow)
            label = row.query_one('.choice-label').render().plain
            assert 'Tiny model' in label
            assert row.query_one('.choice-size').render().plain == '0 MB'
