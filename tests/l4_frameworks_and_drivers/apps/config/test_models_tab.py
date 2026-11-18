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
from lazy_take_notes.l3_interface_adapters.gateways.download_engine import DownloadCancelledError
from lazy_take_notes.l4_frameworks_and_drivers.apps.config import ConfigApp
from lazy_take_notes.l4_frameworks_and_drivers.apps.config.models_tab import ModelChoiceRow, ModelsTab

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
            assert str(app.query_one('#status-tiny').render()) == 'Ready'

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

    async def test_a_download_in_another_window_shows_and_can_be_followed(self, app_factory):
        app, store = app_factory(AssetStatus.BUSY)

        async with app.run_test(size=(140, 50)) as pilot:
            await pilot.pause()
            await app.workers.wait_for_complete()
            assert str(app.query_one('#status-tiny').render()) == 'Downloading'
            assert 'In another window' in str(app.query_one('#detail-tiny').render())
            assert app.query_one('#cancel-tiny').display is False
            assert app.query_one('#remove-tiny').display is False
            app.query_one('#download-tiny', Button).press()
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()

            assert store.installs == ['tiny']
            assert str(app.query_one('#status-tiny').render()) == 'Ready'

    async def test_download_installs_and_updates_the_row(self, app_factory):
        app, store = app_factory(AssetStatus.MISSING)

        async with app.run_test(size=(140, 50)) as pilot:
            await pilot.pause()
            await app.workers.wait_for_complete()
            app.query_one('#download-tiny', Button).press()
            await pilot.pause()
            await app.workers.wait_for_complete()

            assert store.installs == ['tiny']
            assert str(app.query_one('#status-tiny').render()) == 'Ready'

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
            assert str(app.query_one('#status-tiny').render()) == 'Missing'

    async def test_default_model_row_shows_size_and_purpose(self, app_factory):
        app, _store = app_factory(AssetStatus.MISSING)

        async with app.run_test(size=(140, 50)) as pilot:
            await pilot.pause()
            await app.workers.wait_for_complete()
            row = app.query_one('#choice-tiny', ModelChoiceRow)
            label = row.query_one('.choice-label').render().plain
            assert 'Tiny model' in label
            assert row.query_one('.choice-size').render().plain == '0 MB'


class TestModelsTabFaults:
    async def test_state_load_failure_notifies(self, app_factory):
        app, store = app_factory(AssetStatus.READY)

        def _boom() -> list[AssetState]:
            raise OSError('cache unreadable')

        store.list_assets = _boom

        async with app.run_test(size=(140, 50)) as pilot:
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert any('Cannot read the model cache' in str(n.message) for n in app.app._notifications)


SPEAKER_ASSETS = [
    replace(ASSET, id='nemo-runtime', label='Speaker runtime', group='speakers', size=200),
    replace(ASSET, id='nemotron-3-diarization', label='Speaker model', group='speakers', size=300),
]


class SpeakerStore(FakeStore):
    """A catalog with one speech model and both speaker assets, all missing."""

    def __init__(self) -> None:
        super().__init__(AssetStatus.MISSING)
        self._states = {asset.id: AssetState(asset, AssetStatus.MISSING) for asset in [ASSET, *SPEAKER_ASSETS]}
        self.progress_reports = 0

    def catalog(self) -> list[ModelAsset]:
        return [ASSET, *SPEAKER_ASSETS]

    def list_assets(self) -> list[AssetState]:
        return list(self._states.values())

    def status(self, asset_id: str) -> AssetState:
        return self._states[asset_id]

    def install(self, asset_id, on_progress=None, cancel=None) -> Path:
        self.installs.append(asset_id)
        state = self._states[asset_id]
        if on_progress is not None:
            on_progress(AssetProgress(state.asset.size // 2, state.asset.size, 1_000_000))
            self.progress_reports += 1
        self._states[asset_id] = replace(state, status=AssetStatus.READY, path=f'/tmp/{asset_id}')
        return Path(f'/tmp/{asset_id}')

    def remove(self, asset_id: str) -> int:
        self.removes.append(asset_id)
        state = self._states[asset_id]
        self._states[asset_id] = replace(state, status=AssetStatus.MISSING, installed_bytes=0, path=None)
        return state.installed_bytes


class TestSpeakerAssets:
    async def test_install_downloads_both_assets_and_shows_ready(self, app_factory):
        store = SpeakerStore()
        app = ConfigApp(asset_store=store)

        async with app.run_test(size=(140, 50)) as pilot:
            await pilot.pause()
            await app.workers.wait_for_complete()
            app.query_one('#speaker-install', Button).press()
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()

            assert sorted(store.installs) == ['nemo-runtime', 'nemotron-3-diarization']
            assert store.progress_reports == 2
            assert str(app.query_one('.speaker-status').render()).startswith('Ready')
            assert not [n for n in app.app._notifications if n.severity == 'error']

    async def test_download_all_missing_installs_every_asset(self, app_factory):
        store = SpeakerStore()
        app = ConfigApp(asset_store=store)

        async with app.run_test(size=(140, 50)) as pilot:
            await pilot.pause()
            await app.workers.wait_for_complete()
            app.query_one('#models-download-all', Button).press()
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()

            assert sorted(store.installs) == ['nemo-runtime', 'nemotron-3-diarization', 'tiny']
            assert str(app.query_one('#status-tiny').render()) == 'Ready'
            assert not [n for n in app.app._notifications if n.severity == 'error']

    async def test_speaker_progress_shows_in_the_status(self, app_factory):
        store = SpeakerStore()
        app = ConfigApp(asset_store=store)

        async with app.run_test(size=(140, 50)) as pilot:
            await pilot.pause()
            await app.workers.wait_for_complete()
            tab = app.query_one('#section-models', ModelsTab)
            tab.post_message(tab.Started('nemo-runtime'))
            tab.post_message(tab.Progress('nemo-runtime', AssetProgress(100, 200, 0)))
            await pilot.pause()
            assert 'Downloading' in str(app.query_one('.speaker-status').render())
            assert app.query_one('#speaker-install').display is False


class TestSpeakerSizes:
    async def test_ready_and_remove_state_the_size_on_disk(self, app_factory):
        store = SpeakerStore()
        # An extracted runtime takes more space than its archive download.
        for asset_id, size in (('nemo-runtime', 17 * 2**20), ('nemotron-3-diarization', 102 * 2**20)):
            state = store._states[asset_id]  # noqa: SLF001 -- seed the fake store
            store._states[asset_id] = replace(state, status=AssetStatus.READY, installed_bytes=size, path='/tmp/x')  # noqa: SLF001 -- seed the fake store
        app = ConfigApp(asset_store=store)

        async with app.run_test(size=(140, 50)) as pilot:
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert str(app.query_one('.speaker-status').render()) == 'Ready (119 MB)'
            app.query_one('#speaker-remove', Button).press()
            await pilot.pause()
            await pilot.pause()
            body = str(app.app.screen.query_one('#discard-body').render())
            assert body.startswith('This frees 119 MB.')
            app.app.screen.query_one('#discard-confirm', Button).press()
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()

            assert sorted(store.removes) == ['nemo-runtime', 'nemotron-3-diarization']
            freed = [n.message for n in app.app._notifications if n.message.startswith('Freed')]  # noqa: SLF001 -- read the toasts
            assert freed == ['Freed 119 MB.']
            assert str(app.query_one('.speaker-status').render()) == 'Missing'


class TestLeavingDuringADownload:
    async def test_closing_settings_cancels_a_running_download(self, app_factory):
        app, store = app_factory(AssetStatus.MISSING)
        started = threading.Event()
        seen: list[threading.Event] = []

        def slow_install(asset_id, on_progress=None, cancel=None):
            assert cancel is not None
            seen.append(cancel)
            started.set()
            cancel.wait(5)
            raise DownloadCancelledError('download cancelled')

        store.install = slow_install

        async with app.run_test(size=(140, 50)) as pilot:
            await pilot.pause()
            await app.workers.wait_for_complete()
            app.query_one('#download-tiny', Button).press()
            await pilot.pause()
            assert started.wait(2)
        assert seen[0].is_set()

    async def test_two_fast_presses_start_one_download(self, app_factory):
        app, store = app_factory(AssetStatus.MISSING)
        release = threading.Event()
        original = store.install

        def held_install(asset_id, on_progress=None, cancel=None):
            release.wait(5)
            return original(asset_id, on_progress, cancel)

        store.install = held_install

        async with app.run_test(size=(140, 50)) as pilot:
            await pilot.pause()
            await app.workers.wait_for_complete()
            tab = app.query_one('#section-models', ModelsTab)
            tab._start_download('tiny')
            tab._start_download('tiny')
            release.set()
            await app.workers.wait_for_complete()
            assert store.installs == ['tiny']


class TestErrorToastInSettings:
    async def test_an_error_in_settings_does_not_mention_a_session(self, app_factory):
        app, _store = app_factory(AssetStatus.READY)

        async with app.run_test(size=(140, 50)) as pilot:
            await pilot.pause()
            app.call_later(lambda: 1 / 0)
            await pilot.pause()
            await pilot.pause()
            messages = [str(n.message) for n in app.app._notifications if n.severity == 'error']
            assert any('ZeroDivisionError' in message for message in messages)
            assert not any('session' in message for message in messages)


class TestRemoveDialogKeys:
    async def test_enter_in_the_remove_dialog_keeps_the_model(self, app_factory):
        app, store = app_factory(AssetStatus.READY)

        async with app.run_test(size=(140, 50)) as pilot:
            await pilot.pause()
            await app.workers.wait_for_complete()
            app.query_one('#remove-tiny', Button).press()
            await pilot.pause()
            await pilot.press('enter')
            await pilot.pause()
            await app.workers.wait_for_complete()
            assert store.removes == []


class TestRowProgress:
    async def test_a_download_shows_size_speed_and_time_left(self, app_factory):
        app, _store = app_factory(AssetStatus.MISSING)
        megabyte = 1024 * 1024

        async with app.run_test(size=(140, 50)) as pilot:
            await pilot.pause()
            await app.workers.wait_for_complete()
            tab = app.query_one('#section-models', ModelsTab)
            tab.post_message(tab.Started('tiny'))
            tab.post_message(tab.Progress('tiny', AssetProgress(13 * megabyte, 1536 * megabyte, 2 * megabyte)))
            await pilot.pause()
            assert str(app.query_one('#status-tiny').render()) == 'Downloading'
            detail = str(app.query_one('#detail-tiny').render())
            assert detail == '13 MB of 1.5 GB (0%), 2.0 MB/s, about 13 min left'

    async def test_cancel_says_how_much_is_kept(self, app_factory):
        app, _store = app_factory(AssetStatus.MISSING)
        megabyte = 1024 * 1024

        async with app.run_test(size=(140, 50)) as pilot:
            await pilot.pause()
            await app.workers.wait_for_complete()
            tab = app.query_one('#section-models', ModelsTab)
            tab.post_message(tab.Started('tiny'))
            tab.post_message(tab.Progress('tiny', AssetProgress(134 * megabyte, 1536 * megabyte, 2 * megabyte)))
            tab.post_message(tab.Cancelled('tiny'))
            await pilot.pause()
            assert str(app.query_one('#status-tiny').render()) == 'Paused'
            assert str(app.query_one('#detail-tiny').render()) == '134 MB kept. Download continues from there.'
            assert app.query_one('#download-tiny').display is True


class TestSpeakerCancel:
    async def test_cancel_stops_both_speaker_downloads(self, app_factory):
        store = SpeakerStore()
        app = ConfigApp(asset_store=store)
        seen: list[threading.Event] = []
        started = threading.Event()

        def slow_install(asset_id, on_progress=None, cancel=None):
            assert cancel is not None
            seen.append(cancel)
            if len(seen) == 2:
                started.set()
            cancel.wait(5)
            raise DownloadCancelledError('download cancelled')

        store.install = slow_install

        async with app.run_test(size=(140, 50)) as pilot:
            await pilot.pause()
            await app.workers.wait_for_complete()
            app.query_one('#speaker-install', Button).press()
            await pilot.pause()
            assert started.wait(2)
            cancel = app.query_one('#speaker-cancel', Button)
            assert cancel.display is True
            cancel.press()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert all(event.is_set() for event in seen)
            assert str(app.query_one('.speaker-status').render()) == 'Missing'
            assert cancel.display is False


class TestLoadDuringADownload:
    async def test_the_first_state_load_keeps_a_running_download(self, app_factory):
        app, _store = app_factory(AssetStatus.MISSING)

        async with app.run_test(size=(140, 50)) as pilot:
            await pilot.pause()
            await app.workers.wait_for_complete()
            tab = app.query_one('#section-models', ModelsTab)
            tab._cancel['tiny'] = threading.Event()  # a download is running
            tab.post_message(tab.Started('tiny'))
            tab.post_message(tab.Loaded([AssetState(ASSET, AssetStatus.MISSING)]))
            await pilot.pause()
            assert str(app.query_one('#status-tiny').render()) == 'Downloading'
            assert app.query_one('#cancel-tiny').display is True
