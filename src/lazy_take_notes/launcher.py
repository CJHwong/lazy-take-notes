"""The `lazy-take-notes` and `take-note` commands: run the Rust app, else the Python one.

Every install (setup.sh's `take-note` wrapper, `uvx --from git+...`, `uv tool
install`, `pip install`) runs this. It downloads the Rust build pinned in
`rust_release`, checks its SHA-256, caches it per tag, and replaces this process
with it. Anything that goes wrong runs the Python app instead, with one note on
stderr. `LTN_PYTHON=1` always runs the Python app.
"""

from __future__ import annotations

import hashlib
import os
import platform
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

from lazy_take_notes import rust_release

RELEASES = 'https://github.com/CJHwong/lazy-take-notes/releases/download'
BINARY = 'lazy-take-notes'
DOWNLOAD_TIMEOUT_S = 60


class SetupError(Exception):
    """The Rust build cannot be used; the message says why."""


def cache_root() -> Path:
    from platformdirs import user_cache_path  # noqa: PLC0415 -- deferred: not needed when cached

    return user_cache_path('lazy-take-notes') / 'bin'


def pinned_asset() -> tuple[str, str, str] | None:
    """(tag, asset name, SHA-256) for this machine, or None when no build fits it."""
    if rust_release.TAG is None:
        return None
    asset = rust_release.ASSETS.get((platform.system(), platform.machine()))
    if asset is None:
        return None
    return rust_release.TAG, asset[0], asset[1]


def download(url: str, target: Path) -> str:
    """Stream `url` into `target`. Returns the SHA-256 of what arrived."""
    digest = hashlib.sha256()
    try:
        with urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT_S) as response, target.open('wb') as out:  # noqa: S310 -- fixed https URL
            while chunk := response.read(1 << 16):
                digest.update(chunk)
                out.write(chunk)
    except OSError as exc:  # URLError and timeouts are OSErrors
        raise SetupError(f'cannot download {url}: {exc}') from exc
    return digest.hexdigest()


def extract(archive: Path, into: Path) -> None:
    """Unpack the tarball's regular files, refusing any path that leaves `into`."""
    try:
        with tarfile.open(archive, 'r:gz') as tar:
            for member in tar.getmembers():
                target = (into / member.name).resolve()
                if not target.is_relative_to(into.resolve()):
                    raise SetupError(f'unsafe path in the download: {member.name}')
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                if not member.isfile():
                    raise SetupError(f'unexpected entry in the download: {member.name}')
                source = tar.extractfile(member)
                if source is None:  # pragma: no cover -- isfile() members always have data
                    raise SetupError(f'empty entry in the download: {member.name}')
                target.parent.mkdir(parents=True, exist_ok=True)
                with source, target.open('wb') as out:
                    shutil.copyfileobj(source, out)
                target.chmod(0o755)
    except (OSError, tarfile.TarError) as exc:
        raise SetupError(f'cannot unpack the download: {exc}') from exc


def install(tag: str, asset: str, sha256: str, root: Path) -> Path:
    """Download, check and unpack one build into `root/<tag>`. Returns the executable."""
    final = root / tag
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root, prefix='.download-') as scratch:
        work = Path(scratch)
        archive = work / asset
        got = download(f'{RELEASES}/{tag}/{asset}', archive)
        if got != sha256:
            raise SetupError(f'the download of {asset} does not match its pinned SHA-256')
        unpacked = work / 'unpacked'
        extract(archive, unpacked)
        folders = [entry for entry in unpacked.iterdir() if entry.is_dir()]
        if len(folders) != 1 or not (folders[0] / BINARY).is_file():
            raise SetupError(f'{asset} does not hold {BINARY}')
        try:
            folders[0].rename(final)
        except OSError:
            # Another start installed it first.
            if not (final / BINARY).is_file():
                raise
    return final / BINARY


def locked_install(tag: str, asset: str, sha256: str, root: Path) -> Path:
    """`install`, once across processes: a second start waits, then finds it installed."""
    import fcntl  # noqa: PLC0415 -- POSIX only; every pinned build is POSIX

    root.mkdir(parents=True, exist_ok=True)
    with (root / '.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        executable = root / tag / BINARY
        if executable.is_file():
            return executable
        print(f'Downloading lazy-take-notes {tag} (once)...', file=sys.stderr)
        return install(tag, asset, sha256, root)


def rust_executable() -> Path | None:
    """The cached Rust build, installed now if needed. None: run the Python app."""
    if os.environ.get('LTN_PYTHON') == '1':
        return None
    pinned = pinned_asset()
    if pinned is None:
        return None
    tag, asset, sha256 = pinned
    root = cache_root()
    executable = root / tag / BINARY
    if executable.is_file():
        return executable
    try:
        return locked_install(tag, asset, sha256, root)
    except (SetupError, OSError) as exc:
        print(f'Note: running the Python version; the Rust version is unavailable ({exc}).', file=sys.stderr)
        return None


def run_python() -> None:
    from lazy_take_notes.l4_frameworks_and_drivers.cli import (  # noqa: PLC0415 -- deferred: not loaded when Rust runs
        cli,
    )

    cli()


def main() -> None:
    executable = rust_executable()
    if executable is None:
        run_python()
        return
    try:
        # Replace this process, so signals and the terminal go straight to the app.
        os.execv(executable, [str(executable), *sys.argv[1:]])  # noqa: S606 -- verified, pinned build
    except OSError as exc:
        print(f'Note: running the Python version; cannot start {executable} ({exc}).', file=sys.stderr)
        run_python()
