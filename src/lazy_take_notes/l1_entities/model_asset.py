"""Model asset entities for the shared download cache."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal


class AssetStatus(StrEnum):
    """Lifecycle state of one downloadable asset."""

    READY = 'ready'
    MISSING = 'missing'
    DOWNLOADING = 'downloading'
    CORRUPT = 'corrupt'
    UNSUPPORTED = 'unsupported'


@dataclass(frozen=True)
class ModelAsset:
    """One file or runtime tree in the shared cache.

    ``subdir`` and ``entry`` locate the asset under the cache root. A file asset
    uses its file name as ``entry``. A runtime archive uses the executable path
    inside the extracted tree, for example ``bin/nemo-speech``.
    """

    id: str
    label: str
    purpose: str
    group: Literal['transcription', 'speakers']
    kind: Literal['hf', 'release']
    url: str
    size: int
    sha256: str
    subdir: str
    entry: str
    hf_repo: str = ''
    legacy_dirs: tuple[str, ...] = ()
    archive: bool = False
    version: str = ''
    platforms: tuple[str, ...] = ()
    entry_sha256: str = ''


@dataclass(frozen=True)
class AssetProgress:
    """Progress of one install, in bytes."""

    downloaded: int
    total: int
    bytes_per_second: float = 0.0


@dataclass(frozen=True)
class AssetState:
    """Resolved state of one asset. ``path`` points to the entry on disk."""

    asset: ModelAsset
    status: AssetStatus
    installed_bytes: int = 0
    legacy_bytes: int = 0
    path: str | None = None
