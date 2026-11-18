"""Tests for the shared model cache."""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Iterator, Mapping
from dataclasses import replace
from pathlib import Path

import pytest

from lazy_take_notes.l1_entities.errors import ModelResolutionError
from lazy_take_notes.l1_entities.model_asset import AssetProgress, AssetStatus, ModelAsset
from lazy_take_notes.l3_interface_adapters.gateways.cached_model_store import CachedModelStore
from lazy_take_notes.l3_interface_adapters.gateways.download_engine import DownloadCancelledError, HttpSession
from lazy_take_notes.l3_interface_adapters.gateways.model_catalog import RUNTIME_ID


class FakeResponse:
    status_code: int
    headers: Mapping[str, str]

    def __init__(self, body: bytes) -> None:
        self._body = body
        self.status_code = 200
        self.headers = {'Content-Length': str(len(body))}

    def raise_for_status(self) -> None:
        pass

    def iter_content(self, chunk_size: int) -> Iterator[bytes]:
        for offset in range(0, len(self._body), chunk_size):
            yield self._body[offset : offset + chunk_size]

    def close(self) -> None:
        pass


class FakeSession(HttpSession):
    def __init__(self, body: bytes) -> None:
        self.body = body
        self.calls = 0

    def get(self, url: str, **kwargs) -> FakeResponse:
        self.calls += 1
        return FakeResponse(self.body)


BODY = b'model-data'
ASSET = ModelAsset(
    id='tiny',
    label='Tiny model',
    purpose='Test',
    group='transcription',
    kind='hf',
    url='https://example.test/tiny.bin',
    size=len(BODY),
    sha256=hashlib.sha256(BODY).hexdigest(),
    subdir='whisper/tiny',
    entry='tiny.bin',
    hf_repo='owner/repo',
    legacy_dirs=('legacy-dir',),
)

RUNTIME_ENTRY = b'#!/bin/sh\n'
RUNTIME = ModelAsset(
    id=RUNTIME_ID,
    label='Runtime',
    purpose='Test',
    group='speakers',
    kind='release',
    url='https://example.test/runtime.tar.gz',
    size=10,
    sha256='a' * 64,
    subdir='diarization/runtime/runtime',
    entry='bin/tool',
    archive=True,
    entry_sha256=hashlib.sha256(RUNTIME_ENTRY).hexdigest(),
)


def make_store(tmp_path: Path, session: FakeSession | None = None, assets: dict | None = None) -> CachedModelStore:
    return CachedModelStore(
        tmp_path / 'cache',
        assets=assets or {ASSET.id: ASSET},
        session=session or FakeSession(BODY),
    )


class TestInstall:
    def test_install_downloads_and_records_state(self, tmp_path: Path):
        session = FakeSession(BODY)
        store = make_store(tmp_path, session)
        progress: list[AssetProgress] = []

        entry = store.install(ASSET.id, progress.append)

        assert entry.read_bytes() == BODY
        assert session.calls == 1
        assert progress
        state = json.loads((tmp_path / 'cache' / 'state.json').read_text())
        assert state['schema'] == 'ltn-assets/v1'
        assert state['assets'][ASSET.id]['sha256'] == ASSET.sha256

    def test_second_install_reuses_the_cache(self, tmp_path: Path):
        session = FakeSession(BODY)
        store = make_store(tmp_path, session)

        store.install(ASSET.id)
        store.install(ASSET.id)

        assert session.calls == 1

    def test_status_reports_ready(self, tmp_path: Path):
        store = make_store(tmp_path)
        store.install(ASSET.id)

        state = store.status(ASSET.id)

        assert state.status is AssetStatus.READY
        assert state.installed_bytes == len(BODY)
        assert state.path is not None

    def test_corrupt_file_is_reported(self, tmp_path: Path):
        store = make_store(tmp_path)
        entry = tmp_path / 'cache' / 'assets' / ASSET.subdir / ASSET.entry
        entry.parent.mkdir(parents=True)
        entry.write_bytes(b'wrong')

        state = store.status(ASSET.id)

        assert state.status is AssetStatus.CORRUPT

    def test_remove_frees_bytes(self, tmp_path: Path):
        store = make_store(tmp_path)
        store.install(ASSET.id)

        freed = store.remove(ASSET.id)

        assert freed == len(BODY)
        assert store.status(ASSET.id).status is AssetStatus.MISSING

    def test_unsupported_platform(self, tmp_path: Path):
        asset = replace(ASSET, platforms=('nope',))
        store = make_store(tmp_path, assets={asset.id: asset})

        assert store.status(asset.id).status is AssetStatus.UNSUPPORTED

    def test_unknown_asset_raises(self, tmp_path: Path):
        store = make_store(tmp_path)
        with pytest.raises(ValueError, match='unknown model asset'):
            store.status('missing')


class TestResolve:
    def test_resolve_by_id(self, tmp_path: Path):
        store = make_store(tmp_path)
        assert Path(store.resolve('tiny')).read_bytes() == BODY

    def test_resolve_by_hf_uri(self, tmp_path: Path):
        store = make_store(tmp_path)
        assert Path(store.resolve('hf://owner/repo/tiny.bin')).read_bytes() == BODY

    @pytest.mark.parametrize('name', ['tiny', 'hf://owner/repo/tiny.bin'])
    def test_a_set_cancel_event_stops_the_download_and_keeps_nothing_installed(self, tmp_path: Path, name):
        import threading

        from lazy_take_notes.l3_interface_adapters.gateways.download_engine import DownloadCancelledError

        store = make_store(tmp_path)
        cancel = threading.Event()
        cancel.set()
        with pytest.raises(DownloadCancelledError):
            store.resolve(name, cancel=cancel)

    def test_absolute_path_exists(self, tmp_path: Path):
        model = tmp_path / 'model.bin'
        model.touch()
        store = make_store(tmp_path)
        assert store.resolve(str(model)) == str(model)

    def test_absolute_path_missing_raises(self, tmp_path: Path):
        store = make_store(tmp_path)
        with pytest.raises(ModelResolutionError, match='not found'):
            store.resolve(str(tmp_path / 'nope.bin'))

    def test_a_pywhispercpp_name_passes_through_for_its_own_download(self, tmp_path: Path):
        store = make_store(tmp_path)
        assert store.resolve('base.en') == 'base.en'

    def test_a_relative_model_file_passes_through(self, tmp_path: Path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / 'my-model.bin').write_bytes(b'x')
        assert make_store(tmp_path).resolve('my-model.bin') == 'my-model.bin'

    def test_an_unknown_name_raises_instead_of_loading_nothing(self, tmp_path: Path):
        # pywhispercpp loads an unknown name as no model, and the child dies on the first chunk.
        store = make_store(tmp_path)
        with pytest.raises(ModelResolutionError, match="Unknown speech model 'not-a-real-model'"):
            store.resolve('not-a-real-model')

    def test_malformed_hf_uri_raises(self, tmp_path: Path):
        store = make_store(tmp_path)
        with pytest.raises(ModelResolutionError, match='Malformed hf:// URI'):
            store.resolve('hf://')


class TestLegacyAdoption:
    def test_adopts_a_matching_legacy_file(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        legacy_root = tmp_path / 'pywhispercpp'
        legacy = legacy_root / 'legacy-dir' / ASSET.entry
        legacy.parent.mkdir(parents=True)
        legacy.write_bytes(BODY)
        monkeypatch.setattr('pywhispercpp.constants.MODELS_DIR', str(legacy_root))
        store = make_store(tmp_path)

        state = store.status(ASSET.id)

        assert state.status is AssetStatus.READY
        assert state.path is not None
        assert Path(state.path).read_bytes() == BODY
        assert legacy.exists(), 'legacy copy must not be deleted'

    def test_ignores_a_legacy_file_with_a_bad_digest(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        legacy_root = tmp_path / 'pywhispercpp'
        legacy = legacy_root / 'legacy-dir' / ASSET.entry
        legacy.parent.mkdir(parents=True)
        legacy.write_bytes(b'wrong')
        monkeypatch.setattr('pywhispercpp.constants.MODELS_DIR', str(legacy_root))
        store = make_store(tmp_path)

        state = store.status(ASSET.id)

        assert state.status is AssetStatus.MISSING
        assert state.legacy_bytes == len(b'wrong')

    def test_adopts_a_legacy_runtime_tree(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        legacy_tree = tmp_path / 'legacy-runtime'
        (legacy_tree / 'bin').mkdir(parents=True)
        (legacy_tree / 'bin' / 'tool').write_bytes(RUNTIME_ENTRY)
        (legacy_tree / 'lib').mkdir()
        (legacy_tree / 'lib' / 'extra').write_bytes(b'lib-data')
        legacy_store = tmp_path / 'legacy-store'
        legacy_store.mkdir()
        (legacy_store / 'runtime.json').write_text(
            json.dumps({'binary': str(legacy_tree / 'bin' / 'tool')}),
            encoding='utf-8',
        )
        monkeypatch.setattr(
            'lazy_take_notes.l3_interface_adapters.gateways.cached_model_store.LEGACY_NEMOTRON_STORE',
            legacy_store,
        )
        store = make_store(tmp_path, assets={RUNTIME.id: RUNTIME})

        state = store.status(RUNTIME.id)

        assert state.status is AssetStatus.READY
        assert state.path is not None
        assert Path(state.path).read_bytes() == RUNTIME_ENTRY
        assert state.legacy_bytes > len(RUNTIME_ENTRY)
        assert (legacy_tree / 'bin' / 'tool').exists()


class TestState:
    def test_corrupt_state_file_is_ignored(self, tmp_path: Path):
        cache = tmp_path / 'cache'
        cache.mkdir()
        (cache / 'state.json').write_text('{not json', encoding='utf-8')

        store = make_store(tmp_path)

        assert store.status(ASSET.id).status is AssetStatus.MISSING

    def test_non_mapping_state_file_is_ignored(self, tmp_path: Path):
        cache = tmp_path / 'cache'
        cache.mkdir()
        (cache / 'state.json').write_text('[1, 2]', encoding='utf-8')

        store = make_store(tmp_path)

        assert store.status(ASSET.id).status is AssetStatus.MISSING

    def test_cancel_propagates(self, tmp_path: Path):
        cancel = threading.Event()
        cancel.set()
        store = make_store(tmp_path)

        with pytest.raises(DownloadCancelledError, match='cancel'):
            store.install(ASSET.id, cancel=cancel)


class HeldSession(FakeSession):
    """Blocks inside the download until released, so a test can act mid-download."""

    def __init__(self, body: bytes) -> None:
        super().__init__(body)
        self.entered = threading.Event()
        self.release = threading.Event()

    def get(self, url: str, **kwargs) -> FakeResponse:
        self.entered.set()
        assert self.release.wait(5)
        return super().get(url, **kwargs)


class TestTwoDownloadsOfOneAsset:
    """Two stores share one cache: the preflight and Settings, or two processes."""

    def test_the_second_waits_and_reuses_the_first_download(self, tmp_path: Path):
        first_session = HeldSession(BODY)
        second_session = FakeSession(BODY)
        first = make_store(tmp_path, first_session)
        second = make_store(tmp_path, second_session)
        results: list[str] = []
        worker = threading.Thread(target=lambda: results.append(str(first.install('tiny'))))
        worker.start()
        assert first_session.entered.wait(5)

        waiting = threading.Thread(target=lambda: results.append(str(second.install('tiny'))))
        waiting.start()
        waiting.join(0.5)
        assert waiting.is_alive()  # it waits for the first download instead of writing the same .part

        first_session.release.set()
        worker.join(5)
        waiting.join(5)
        assert second_session.calls == 0
        assert len(results) == 2 and results[0] == results[1]
        assert second.status('tiny').status is AssetStatus.READY

    def test_a_cancel_while_waiting_stops_the_wait(self, tmp_path: Path):
        first_session = HeldSession(BODY)
        first = make_store(tmp_path, first_session)
        second = make_store(tmp_path, FakeSession(BODY))
        worker = threading.Thread(target=lambda: first.install('tiny'))
        worker.start()
        assert first_session.entered.wait(5)

        cancel = threading.Event()
        errors: list[BaseException] = []

        def wait_then_cancel() -> None:
            try:
                second.install('tiny', cancel=cancel)
            except BaseException as error:  # noqa: BLE001 -- the test inspects it
                errors.append(error)

        waiting = threading.Thread(target=wait_then_cancel)
        waiting.start()
        cancel.set()
        waiting.join(5)
        assert not waiting.is_alive()
        assert isinstance(errors[0], DownloadCancelledError)
        first_session.release.set()
        worker.join(5)

    def test_status_answers_while_a_download_runs(self, tmp_path: Path):
        session = HeldSession(BODY)
        store = make_store(tmp_path, session)
        worker = threading.Thread(target=lambda: store.install('tiny'))
        worker.start()
        assert session.entered.wait(5)

        answer: list[AssetStatus] = []
        probe = threading.Thread(target=lambda: answer.append(store.status('tiny').status))
        probe.start()
        probe.join(1)
        session.release.set()
        worker.join(5)
        assert answer == [AssetStatus.MISSING]  # the UI thread did not wait for the download


class TestFirstDownloadOnANewAccount:
    def test_a_missing_cache_parent_still_checks_the_disk(self, tmp_path: Path):
        store = CachedModelStore(
            tmp_path / 'home' / '.cache' / 'lazy-take-notes', assets={ASSET.id: ASSET}, session=FakeSession(BODY)
        )
        path = store.install('tiny')
        assert path.read_bytes() == BODY


class TestSharedStateFile:
    def test_a_second_store_keeps_the_first_store_s_record(self, tmp_path: Path):
        other_body = b'other-model'
        other = replace(
            ASSET,
            id='other',
            url='https://example.test/other.bin',
            size=len(other_body),
            sha256=hashlib.sha256(other_body).hexdigest(),
            subdir='whisper/other',
            entry='other.bin',
        )
        assets = {ASSET.id: ASSET, other.id: other}
        first = make_store(tmp_path, FakeSession(BODY), assets=assets)
        second = make_store(tmp_path, FakeSession(other_body), assets=assets)  # loaded before the first install

        first.install('tiny')
        second.install('other')

        state = json.loads((tmp_path / 'cache' / 'state.json').read_text())
        assert set(state['assets']) == {'tiny', 'other'}
        assert not list((tmp_path / 'cache').glob('*.tmp'))
