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

    def test_absolute_path_exists(self, tmp_path: Path):
        model = tmp_path / 'model.bin'
        model.touch()
        store = make_store(tmp_path)
        assert store.resolve(str(model)) == str(model)

    def test_absolute_path_missing_raises(self, tmp_path: Path):
        store = make_store(tmp_path)
        with pytest.raises(ModelResolutionError, match='not found'):
            store.resolve(str(tmp_path / 'nope.bin'))

    def test_passthrough_name(self, tmp_path: Path):
        store = make_store(tmp_path)
        assert store.resolve('some-custom-model') == 'some-custom-model'

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

    def test_cancel_propagates(self, tmp_path: Path):
        cancel = threading.Event()
        cancel.set()
        store = make_store(tmp_path)

        with pytest.raises(DownloadCancelledError, match='cancel'):
            store.install(ASSET.id, cancel=cancel)
