"""Gateway: shared cache for downloaded models and runtimes."""

from __future__ import annotations

import json
import logging
import os
import shutil
import threading
from collections.abc import Callable, Generator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from lazy_take_notes.l1_entities.errors import ModelResolutionError
from lazy_take_notes.l1_entities.model_asset import (
    AssetProgress,
    AssetState,
    AssetStatus,
    ModelAsset,
)
from lazy_take_notes.l3_interface_adapters.gateways import file_lock, model_catalog
from lazy_take_notes.l3_interface_adapters.gateways.download_engine import (
    DownloadCancelledError,
    DownloadError,
    HttpSession,
    download_file,
    extract_archive,
    sha256_file,
)
from lazy_take_notes.l3_interface_adapters.gateways.paths import CACHE_DIR, LEGACY_NEMOTRON_STORE

log = logging.getLogger('ltn.assets')

_SCHEMA = 'ltn-assets/v1'
_DISK_MARGIN = 64 * 1024 * 1024


class CachedModelStore:
    """Download, verify, and reuse model assets in one cache root.

    The store satisfies both the ``AssetStore`` and ``ModelResolver`` ports.
    It never deletes legacy copies from other tools. It adopts them with a
    hard link or a copy, and the caller decides when to remove the original.
    """

    def __init__(
        self,
        cache_root: Path | None = None,
        *,
        assets: dict[str, ModelAsset] | None = None,
        session: HttpSession | None = None,
    ) -> None:
        self._root = Path(cache_root) if cache_root is not None else CACHE_DIR
        self._assets = dict(assets) if assets is not None else model_catalog.catalog()
        self._session = session
        self._lock = threading.RLock()
        self._state_path = self._root / 'state.json'
        self._state: dict = self._load_state()

    # --- AssetStore -----------------------------------------------------

    @property
    def cache_root(self) -> Path:
        """The directory that holds every asset and the state file."""
        return self._root

    def list_assets(self) -> list[AssetState]:
        return [self.status(asset_id) for asset_id in self._assets]

    def catalog(self) -> list[ModelAsset]:
        """Return the catalog without touching the disk.

        The UI builds its rows from this before a worker computes the states.
        """
        return list(self._assets.values())

    def status(self, asset_id: str) -> AssetState:
        asset = self._require(asset_id)
        with self._lock:
            if asset.platforms and model_catalog.platform_key() not in asset.platforms:
                return AssetState(asset, AssetStatus.UNSUPPORTED)

            entry = self._entry_path(asset)
            if entry.exists():
                if self._final_matches(asset, entry):
                    return AssetState(asset, AssetStatus.READY, _tree_size(entry), 0, str(entry))
                return AssetState(asset, AssetStatus.CORRUPT, _tree_size(entry), 0, str(entry))

            legacy = self._legacy_entry(asset)
            if legacy is None:
                return AssetState(asset, AssetStatus.MISSING)
            legacy_bytes = _legacy_size(asset, legacy)
            digest = self._legacy_digest(asset, legacy)
            if digest is not None:
                self._adopt(asset, legacy, digest)
                return AssetState(asset, AssetStatus.READY, _tree_size(entry), legacy_bytes, str(entry))
            return AssetState(asset, AssetStatus.MISSING, 0, legacy_bytes)

    def install(
        self,
        asset_id: str,
        on_progress: Callable[[AssetProgress], None] | None = None,
        cancel: threading.Event | None = None,
    ) -> Path:
        state = self.status(asset_id)
        if state.status is AssetStatus.READY and state.path:
            return Path(state.path)
        # Only the file lock spans the download, so status() still answers the UI meanwhile.
        with self._download_lock(state.asset, on_progress, cancel):
            state = self.status(asset_id)  # another store or process may have finished it
            if state.status is AssetStatus.READY and state.path:
                return Path(state.path)
            self._install_asset(state.asset, on_progress, cancel)
        return self._entry_path(state.asset)

    def remove(self, asset_id: str) -> int:
        asset = self._require(asset_id)
        with self._lock:
            target = self._entry_path(asset).parent if asset.archive else self._entry_path(asset)
            freed = _tree_size(target)
            if target.is_dir():
                shutil.rmtree(target)
            elif target.exists():
                target.unlink()
            self._state.pop(asset.id, None)
            self._save_state()
            return freed

    def resolve(self, name: str, cancel: threading.Event | None = None) -> str:
        asset = self._assets.get(name) or model_catalog.asset_for_model(name, self._assets)
        if asset is not None:
            return str(self.install(asset.id, cancel=cancel))

        parsed = model_catalog.parse_hf_uri(model_catalog.expand_model_alias(name))
        if parsed is not None:
            return str(self._install_custom(*parsed, cancel=cancel))
        if name.startswith('hf://'):
            raise ModelResolutionError(f'Malformed hf:// URI (expected hf://owner/repo/file): {name}')

        path = Path(name).expanduser()
        if path.is_absolute():
            if path.is_file():
                return str(path)
            raise ModelResolutionError(f'Model file not found: {name}')
        if path.is_file() or _is_pywhispercpp_name(name):
            return name
        # pywhispercpp loads any other name as no model; the child then dies on the first chunk.
        raise ModelResolutionError(f"Unknown speech model '{name}'")

    # --- Install --------------------------------------------------------

    def _install_asset(
        self,
        asset: ModelAsset,
        on_progress: Callable[[AssetProgress], None] | None,
        cancel: threading.Event | None,
    ) -> None:
        self._preflight(asset)
        entry = self._entry_path(asset)
        if asset.archive:
            self._install_archive(asset, on_progress, cancel)
        else:
            download_file(
                asset.url,
                entry,
                expected_size=asset.size,
                expected_sha256=asset.sha256,
                on_progress=on_progress,
                cancel=cancel,
                session=self._session,
            )
            self._record(asset, asset.sha256 or sha256_file(entry))
        log.info('installed model asset %s at %s', asset.id, entry)

    def _install_archive(
        self,
        asset: ModelAsset,
        on_progress: Callable[[AssetProgress], None] | None,
        cancel: threading.Event | None,
    ) -> None:
        directory = self._asset_dir(asset)
        if directory.exists():
            shutil.rmtree(directory)
        name = asset.url.rsplit('/', 1)[-1]
        archive = self._root / 'downloads' / name
        download_file(
            asset.url,
            archive,
            expected_size=asset.size,
            expected_sha256=asset.sha256,
            on_progress=on_progress,
            cancel=cancel,
            session=self._session,
        )
        entry = extract_archive(
            archive,
            directory,
            asset.entry,
            expected_entry_sha256=asset.entry_sha256,
            cancel=cancel,
        )
        archive.unlink(missing_ok=True)
        digest = asset.entry_sha256 or sha256_file(entry)
        self._record(asset, digest)

    def _install_custom(self, repo: str, filename: str, cancel: threading.Event | None = None) -> Path:
        asset = ModelAsset(
            id=f'custom:{repo}/{filename}',
            label=filename,
            purpose='Custom model',
            group='transcription',
            kind='hf',
            url=f'https://huggingface.co/{repo}/resolve/main/{filename}',
            size=0,
            sha256='',
            subdir=f'whisper/custom/{repo.replace("/", "__")}',
            entry=filename,
            hf_repo=repo,
        )
        with self._download_lock(asset, None, cancel):
            entry = self._entry_path(asset)
            if not entry.is_file():
                self._install_asset(asset, None, cancel)
        return self._entry_path(asset)

    @contextmanager
    def _download_lock(
        self,
        asset: ModelAsset,
        on_progress: Callable[[AssetProgress], None] | None,
        cancel: threading.Event | None,
    ) -> Generator[None]:
        part = self._part_path(asset)
        logged = False

        def waiting() -> None:
            nonlocal logged
            if not logged:
                log.info('waiting for another download of %s', asset.id)
                logged = True
            if on_progress is None:
                return
            try:
                size = part.stat().st_size
            except OSError:
                return  # the other download just finished and renamed it
            on_progress(AssetProgress(size, asset.size))

        name = ''.join(char if char.isalnum() or char in '-_.' else '_' for char in asset.id)
        try:
            with file_lock.exclusive(
                self._root / 'locks' / f'{name}.lock',
                cancelled=lambda: cancel is not None and cancel.is_set(),
                on_wait=waiting,
            ):
                yield
        except InterruptedError as error:
            raise DownloadCancelledError(str(error)) from error

    def _part_path(self, asset: ModelAsset) -> Path:
        if asset.archive:
            return self._root / 'downloads' / f'{asset.url.rsplit("/", 1)[-1]}.part'
        return Path(f'{self._entry_path(asset)}.part')

    def _preflight(self, asset: ModelAsset) -> None:
        if asset.size == 0:
            return
        probe = self._root
        while not probe.exists() and probe != probe.parent:
            probe = probe.parent  # a new Linux account may have no ~/.cache yet
        free = shutil.disk_usage(probe).free
        if free < asset.size + _DISK_MARGIN:
            raise DownloadError(
                f'not enough disk space for {asset.label}: need {_megabytes(asset.size)}, have {_megabytes(free)}'
            )

    # --- Verification and state ----------------------------------------

    def _final_matches(self, asset: ModelAsset, entry: Path) -> bool:
        record = self._state.get(asset.id, {})
        stat = entry.stat()
        if (
            record.get('path') == self._relative(entry)
            and record.get('size') == stat.st_size
            and record.get('mtime_ns') == stat.st_mtime_ns
        ):
            return True
        digest = sha256_file(entry)
        expected = asset.entry_sha256 if asset.archive else asset.sha256
        if expected and digest != expected:
            return False
        self._record(asset, digest)
        return True

    def _legacy_digest(self, asset: ModelAsset, legacy: Path) -> str | None:
        digest = sha256_file(legacy)
        expected = asset.entry_sha256 if asset.archive else asset.sha256
        if expected and digest != expected:
            log.warning('ignoring legacy copy with a bad digest: %s', legacy)
            return None
        return digest

    def _adopt(self, asset: ModelAsset, legacy: Path, digest: str) -> None:
        entry = self._entry_path(asset)
        if asset.archive:
            tree = entry.parent.parent
            # _adopt runs only when the entry is missing, so an existing tree is
            # incomplete from an earlier attempt.
            if tree.exists():
                shutil.rmtree(tree)
            tree.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(legacy.parent.parent, tree)
        else:
            entry.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(legacy, entry)
            except OSError:
                shutil.copy2(legacy, entry)
        log.info('adopted legacy asset %s from %s', asset.id, legacy)
        self._record(asset, digest)

    def _record(self, asset: ModelAsset, digest: str) -> None:
        entry = self._entry_path(asset)
        stat = entry.stat()
        with self._lock:
            # Another store or process may have recorded assets since this one loaded the file.
            self._state = {**self._load_state(), **self._state}
            self._state[asset.id] = {
                'path': self._relative(entry),
                'size': stat.st_size,
                'mtime_ns': stat.st_mtime_ns,
                'sha256': digest,
                'source': asset.url,
                'version': asset.version,
            }
            self._save_state()

    def _load_state(self) -> dict:
        try:
            payload = json.loads(self._state_path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            return {}
        if not isinstance(payload, dict) or payload.get('schema') != _SCHEMA:
            return {}
        assets = payload.get('assets')
        return assets if isinstance(assets, dict) else {}

    def _save_state(self) -> None:
        self._root.mkdir(parents=True, exist_ok=True)
        payload = {
            'schema': _SCHEMA,
            'updated': datetime.now(UTC).isoformat(timespec='seconds'),
            'assets': self._state,
        }
        # One temporary file per writer, so two processes never replace each other's half-written file.
        temporary = self._state_path.with_name(f'state.json.{os.getpid()}.{threading.get_ident()}.tmp')
        temporary.write_text(json.dumps(payload, indent=2), encoding='utf-8')
        os.replace(temporary, self._state_path)

    # --- Paths ----------------------------------------------------------

    def _asset_dir(self, asset: ModelAsset) -> Path:
        return self._root / 'assets' / asset.subdir

    def _entry_path(self, asset: ModelAsset) -> Path:
        return self._asset_dir(asset) / asset.entry

    def _relative(self, path: Path) -> str:
        return str(path.relative_to(self._root))

    def _require(self, asset_id: str) -> ModelAsset:
        asset = self._assets.get(asset_id)
        if asset is None:
            raise ValueError(f'unknown model asset: {asset_id!r}')
        return asset

    def _legacy_entry(self, asset: ModelAsset) -> Path | None:
        for candidate in self._legacy_candidates(asset):
            if candidate.is_file():
                return candidate
        return None

    def _legacy_candidates(self, asset: ModelAsset) -> list[Path]:
        candidates: list[Path] = []
        if asset.hf_repo:
            from pywhispercpp.constants import MODELS_DIR  # noqa: PLC0415 -- deferred: only needed for legacy adoption

            root = Path(MODELS_DIR)
            candidates.extend(root / legacy / asset.entry for legacy in asset.legacy_dirs)
            owner, repo = asset.hf_repo.split('/', 1)
            candidates.append(root / 'hf' / f'{owner}__{repo}' / asset.entry)

        record = _read_runtime_record()
        if record is not None:
            key = 'binary' if asset.id == model_catalog.RUNTIME_ID else 'model'
            value = record.get(key)
            if isinstance(value, str) and value:
                candidates.append(Path(value).expanduser())
        return candidates


def _is_pywhispercpp_name(name: str) -> bool:
    """pywhispercpp downloads these names itself."""
    from pywhispercpp.constants import AVAILABLE_MODELS  # noqa: PLC0415 -- deferred: rare path

    return name in AVAILABLE_MODELS


def _read_runtime_record() -> dict | None:
    path = LEGACY_NEMOTRON_STORE / 'runtime.json'
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _legacy_size(asset: ModelAsset, legacy: Path) -> int:
    """Size of the legacy artifact. A runtime entry lives inside its tree."""
    return _tree_size(legacy.parent.parent) if asset.archive else _tree_size(legacy)


def _tree_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    if not path.is_dir():
        return 0
    return sum(child.stat().st_size for child in path.rglob('*') if child.is_file())


def _megabytes(size: int) -> str:
    return f'{size / (1024 * 1024):.0f} MB'
