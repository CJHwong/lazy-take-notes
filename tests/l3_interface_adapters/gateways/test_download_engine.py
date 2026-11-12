"""Tests for the HTTP download and archive extraction engine."""

from __future__ import annotations

import hashlib
import io
import tarfile
import threading
from collections.abc import Iterator, Mapping
from pathlib import Path

import pytest

from lazy_take_notes.l1_entities.model_asset import AssetProgress
from lazy_take_notes.l3_interface_adapters.gateways.download_engine import (
    DownloadCancelledError,
    DownloadError,
    HttpSession,
    download_file,
    extract_archive,
    sha256_file,
)


class FakeResponse:
    status_code: int
    headers: Mapping[str, str]

    def __init__(self, body: bytes, status: int = 200, headers: dict | None = None) -> None:
        self._body = body
        self.status_code = status
        self.headers = headers or {'Content-Length': str(len(body))}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:  # noqa: PLR2004 -- HTTP error range
            raise DownloadError(f'HTTP {self.status_code}')

    def iter_content(self, chunk_size: int) -> Iterator[bytes]:
        for offset in range(0, len(self._body), chunk_size):
            yield self._body[offset : offset + chunk_size]

    def close(self) -> None:
        pass


class FakeSession(HttpSession):
    def __init__(self, body: bytes, status: int = 200, headers: dict | None = None) -> None:
        self.body = body
        self.status = status
        self.headers = headers
        self.headers_seen: dict = {}

    def get(self, url: str, **kwargs) -> FakeResponse:
        self.headers_seen = kwargs.get('headers') or {}
        headers = self.headers or {'Content-Length': str(len(self.body))}
        return FakeResponse(self.body, self.status, headers)


def _digest(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


class TestDownloadFile:
    def test_downloads_and_verifies(self, tmp_path: Path):
        body = b'hello world'
        dest = tmp_path / 'model.bin'
        progress: list[AssetProgress] = []

        download_file(
            'https://example.test/model.bin',
            dest,
            expected_size=len(body),
            expected_sha256=_digest(body),
            on_progress=progress.append,
            session=FakeSession(body),
        )

        assert dest.read_bytes() == body
        assert progress[-1].downloaded == len(body)

    def test_existing_valid_file_skips_network(self, tmp_path: Path):
        body = b'hello'
        dest = tmp_path / 'model.bin'
        dest.write_bytes(body)
        session = FakeSession(b'other')

        download_file(
            'https://example.test/model.bin',
            dest,
            expected_size=len(body),
            expected_sha256=_digest(body),
            session=session,
        )

        assert dest.read_bytes() == body
        assert session.headers_seen == {}

    def test_resumes_a_partial_file(self, tmp_path: Path):
        body = b'0123456789'
        dest = tmp_path / 'model.bin'
        (tmp_path / 'model.bin.part').write_bytes(body[:5])
        session = FakeSession(body[5:], status=206, headers={'Content-Length': '5'})

        download_file(
            'https://example.test/model.bin',
            dest,
            expected_size=len(body),
            expected_sha256=_digest(body),
            session=session,
        )

        assert session.headers_seen == {'Range': 'bytes=5-'}
        assert dest.read_bytes() == body

    def test_digest_mismatch_deletes_the_part(self, tmp_path: Path):
        body = b'hello'
        dest = tmp_path / 'model.bin'

        with pytest.raises(DownloadError, match='digest mismatch'):
            download_file(
                'https://example.test/model.bin',
                dest,
                expected_sha256='0' * 64,
                session=FakeSession(body),
            )

        assert not dest.exists()
        assert not (tmp_path / 'model.bin.part').exists()

    def test_cancel_keeps_the_part(self, tmp_path: Path):
        body = b'x' * (3 * 1024 * 1024)
        dest = tmp_path / 'model.bin'
        cancel = threading.Event()

        def stop_after_first(_progress: AssetProgress) -> None:
            cancel.set()

        with pytest.raises(DownloadCancelledError):
            download_file(
                'https://example.test/model.bin',
                dest,
                on_progress=stop_after_first,
                cancel=cancel,
                session=FakeSession(body),
            )

        assert not dest.exists()
        assert (tmp_path / 'model.bin.part').exists()


class TestExtractArchive:
    def _archive(self, tmp_path: Path, members: dict[str, bytes]) -> Path:
        archive = tmp_path / 'runtime.tar.gz'
        source = tmp_path / 'runtime'
        source.mkdir()
        for name, body in members.items():
            path = source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(body)
        with tarfile.open(archive, 'w:gz') as tar:
            tar.add(source, arcname='runtime')
        return archive

    def test_extracts_and_returns_entry(self, tmp_path: Path):
        archive = self._archive(tmp_path, {'bin/tool': b'#!/bin/sh\n'})
        target = tmp_path / 'out' / 'runtime'

        entry = extract_archive(archive, target, 'bin/tool')

        assert entry == target / 'bin' / 'tool'
        assert entry.read_bytes() == b'#!/bin/sh\n'

    def test_entry_digest_mismatch_raises(self, tmp_path: Path):
        archive = self._archive(tmp_path, {'bin/tool': b'x'})
        target = tmp_path / 'out' / 'runtime'

        with pytest.raises(DownloadError, match='entry digest mismatch'):
            extract_archive(archive, target, 'bin/tool', expected_entry_sha256='0' * 64)

    def test_rejects_path_traversal(self, tmp_path: Path):
        archive = tmp_path / 'bad.tar.gz'
        with tarfile.open(archive, 'w:gz') as tar:
            info = tarfile.TarInfo('../evil')
            info.size = 1
            tar.addfile(info, io.BytesIO(b'x'))

        with pytest.raises(DownloadError, match='escapes the target'):
            extract_archive(archive, tmp_path / 'out', 'bin/tool')


class TestSha256File:
    def test_hashes_file(self, tmp_path: Path):
        path = tmp_path / 'file'
        path.write_bytes(b'abc')
        assert sha256_file(path) == hashlib.sha256(b'abc').hexdigest()
