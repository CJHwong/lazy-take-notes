"""HTTP download and archive extraction for the model cache.

The engine resumes partial downloads, reports progress, and verifies every
asset before it lands in the cache.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tarfile
import tempfile
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Protocol, cast

import requests

from lazy_take_notes.l1_entities.model_asset import AssetProgress

_CHUNK = 1024 * 1024
_REPORT_INTERVAL = 0.25
_TIMEOUT = (15, 60)


class HttpResponse(Protocol):
    """The response surface the downloader uses."""

    status_code: int
    headers: Mapping[str, str]

    def raise_for_status(self) -> None: ...
    def iter_content(self, chunk_size: int) -> Iterator[bytes]: ...
    def close(self) -> None: ...


class HttpSession(Protocol):
    """The session surface the downloader uses."""

    def get(self, url: str, **kwargs) -> HttpResponse: ...


class DownloadError(RuntimeError):
    """A download, verification, or extraction failed."""


class DownloadCancelledError(DownloadError):
    """The caller cancelled the transfer. The partial file is kept."""


def sha256_file(path: Path, cancel: threading.Event | None = None) -> str:
    """Return the SHA-256 hex digest of *path*."""
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while True:
            if cancel is not None and cancel.is_set():
                raise DownloadCancelledError('cancelled while verifying')
            chunk = handle.read(_CHUNK)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def download_file(
    url: str,
    dest: Path,
    *,
    expected_size: int = 0,
    expected_sha256: str = '',
    on_progress: Callable[[AssetProgress], None] | None = None,
    cancel: threading.Event | None = None,
    session: HttpSession | None = None,
) -> Path:
    """Download *url* to *dest*, resuming a ``.part`` file when possible."""
    dest = Path(dest)
    if dest.is_file() and _matches(dest, expected_size, expected_sha256, cancel):
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    part = Path(f'{dest}.part')
    existing = part.stat().st_size if part.is_file() else 0
    headers = {'Range': f'bytes={existing}-'} if existing else {}

    client = session or requests
    response = cast(HttpResponse, client.get(url, stream=True, timeout=_TIMEOUT, headers=headers, allow_redirects=True))
    if response.status_code == 416:  # noqa: PLR2004 -- HTTP range not satisfiable
        response.close()
        if expected_size and existing == expected_size:
            _verify(part, expected_size, expected_sha256, cancel)
            os.replace(part, dest)
            return dest
        part.unlink(missing_ok=True)
        raise DownloadError(f'stale partial file for {url}')
    try:
        response.raise_for_status()
        mode, existing = _resume_mode(response.status_code, existing, url)
        total = _content_total(response, existing)
        downloaded, started = _write_body(response, part, mode, existing, total, on_progress, cancel)
    finally:
        response.close()

    _verify(part, expected_size or total, expected_sha256, cancel)
    os.replace(part, dest)
    if on_progress is not None:
        elapsed = max(time.monotonic() - started, 0.001)
        on_progress(AssetProgress(downloaded, total or downloaded, (downloaded - existing) / elapsed))
    return dest


def extract_archive(
    archive: Path,
    target: Path,
    entry: str,
    *,
    expected_entry_sha256: str = '',
    cancel: threading.Event | None = None,
) -> Path:
    """Extract *archive* into *target* and return ``target / entry``.

    The archive must contain one top-level directory. Its name must match the
    archive name without ``.tar.gz``.
    """
    target = Path(target)
    entry_path = target / entry
    if entry_path.is_file():
        if expected_entry_sha256 and sha256_file(entry_path, cancel) != expected_entry_sha256:
            raise DownloadError(f'entry digest mismatch in {target}')
        return entry_path

    target.parent.mkdir(parents=True, exist_ok=True)
    workdir = Path(tempfile.mkdtemp(prefix='.extract-', dir=target.parent))
    try:
        source = _extract_tree(archive, workdir, cancel)
        candidate = source / entry
        if not candidate.is_file():
            raise DownloadError(f'archive {archive.name} has no {entry}')
        if expected_entry_sha256 and sha256_file(candidate, cancel) != expected_entry_sha256:
            raise DownloadError(f'entry digest mismatch in {archive.name}')
        os.replace(source, target)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    return entry_path


def _matches(path: Path, size: int, digest: str, cancel: threading.Event | None) -> bool:
    if size and path.stat().st_size != size:
        return False
    return not digest or sha256_file(path, cancel) == digest


def _resume_mode(status: int, existing: int, url: str) -> tuple[str, int]:
    if status == 206:  # noqa: PLR2004 -- HTTP partial content
        return 'ab', existing
    if status != 200:  # noqa: PLR2004 -- HTTP OK
        raise DownloadError(f'unexpected HTTP {status} from {url}')
    return 'wb', 0


def _content_total(response: HttpResponse, existing: int) -> int:
    raw = response.headers.get('Content-Length', '')
    if not raw.isdigit():
        return 0
    return existing + int(raw)


def _write_body(
    response: HttpResponse,
    part: Path,
    mode: str,
    existing: int,
    total: int,
    on_progress: Callable[[AssetProgress], None] | None,
    cancel: threading.Event | None,
) -> tuple[int, float]:
    downloaded = existing
    started = time.monotonic()
    last_report = 0.0
    with part.open(mode) as handle:
        for chunk in response.iter_content(chunk_size=_CHUNK):
            if cancel is not None and cancel.is_set():
                raise DownloadCancelledError('download cancelled')
            if not chunk:
                continue
            handle.write(chunk)
            downloaded += len(chunk)
            if total and downloaded > total:
                raise DownloadError(f'download exceeded {total} bytes')
            now = time.monotonic()
            if on_progress is not None and now - last_report >= _REPORT_INTERVAL:
                speed = (downloaded - existing) / max(now - started, 0.001)
                on_progress(AssetProgress(downloaded, total, speed))
                last_report = now
    return downloaded, started


def _verify(part: Path, size: int, digest: str, cancel: threading.Event | None) -> None:
    actual_size = part.stat().st_size
    if size and actual_size != size:
        part.unlink(missing_ok=True)
        raise DownloadError(f'size mismatch: expected {size}, got {actual_size}')
    if digest:
        actual = sha256_file(part, cancel)
        if actual != digest:
            part.unlink(missing_ok=True)
            raise DownloadError(f'digest mismatch: expected {digest}, got {actual}')


def _extract_tree(archive: Path, workdir: Path, cancel: threading.Event | None) -> Path:
    with tarfile.open(archive, 'r:gz') as tar:
        members = _safe_members(tar, workdir)
        try:
            tar.extractall(workdir, members=members, filter='data')  # noqa: S202 -- members validated by _safe_members
        except TypeError:  # noqa: PLW0714 -- Python without the tar filter argument
            tar.extractall(workdir, members=members)  # noqa: S202 -- members validated by _safe_members
    if cancel is not None and cancel.is_set():
        raise DownloadCancelledError('cancelled while extracting')

    expected = archive.name.removesuffix('.tar.gz')
    source = workdir / expected
    if source.is_dir():
        return source
    children = [child for child in workdir.iterdir() if child.is_dir()]
    if len(children) == 1:
        return children[0]
    raise DownloadError(f'archive {archive.name} does not contain {expected}')


def _safe_members(tar: tarfile.TarFile, root: Path) -> list[tarfile.TarInfo]:
    resolved_root = root.resolve()
    members: list[tarfile.TarInfo] = []
    for member in tar.getmembers():
        destination = (resolved_root / member.name).resolve()
        if not destination.is_relative_to(resolved_root):
            raise DownloadError(f'archive member escapes the target: {member.name}')
        if member.issym() or member.islnk():
            link_target = (destination.parent / member.linkname).resolve()
            if not link_target.is_relative_to(resolved_root):
                raise DownloadError(f'archive link escapes the target: {member.name}')
        members.append(member)
    return members
