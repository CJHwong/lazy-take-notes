"""The launcher: the pinned Rust build when it can be had, else the Python app."""

from __future__ import annotations

import hashlib
import io
import sys
import tarfile
import types
import urllib.error
import zipfile
from pathlib import Path
from typing import Any

import pytest

from lazy_take_notes import launcher, rust_release

ASSET = 'lazy-take-notes-arm64-macos.tar.gz'


def tarball(entries: dict[str, bytes], folder: str = 'lazy-take-notes-arm64-macos') -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w:gz') as tar:
        top = tarfile.TarInfo(folder)
        top.type = tarfile.DIRTYPE
        tar.addfile(top)
        for name, data in entries.items():
            info = tarfile.TarInfo(f'{folder}/{name}' if not name.startswith('/') and '..' not in name else name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


GOOD = tarball({'lazy-take-notes': b'#!/bin/sh\n', 'coreaudio-tap': b'tap'})


class Served:
    """Stands in for urlopen: serves `body`, or raises `error`, and counts the calls."""

    def __init__(self, body: bytes = GOOD, error: Exception | None = None):
        self.body, self.error, self.urls = body, error, []

    def __call__(self, url, timeout):
        self.urls.append(url)
        if self.error:
            raise self.error
        return io.BytesIO(self.body)


@pytest.fixture
def pinned(monkeypatch, tmp_path):
    """A pinned v9 build for this machine, a temp cache, and a fake network."""

    def pin(body: bytes = GOOD, sha: str | None = None, error: Exception | None = None) -> Served:
        served = Served(body, error)
        monkeypatch.setattr(rust_release, 'TAG', 'v9')
        monkeypatch.setattr(
            rust_release, 'ASSETS', {('Darwin', 'arm64'): (ASSET, sha or hashlib.sha256(body).hexdigest())}
        )
        monkeypatch.setattr(launcher.platform, 'system', lambda: 'Darwin')
        monkeypatch.setattr(launcher.platform, 'machine', lambda: 'arm64')
        monkeypatch.setattr(launcher, 'cache_root', lambda: tmp_path / 'bin')
        monkeypatch.setattr(launcher.urllib.request, 'urlopen', served)
        monkeypatch.delenv('LTN_PYTHON', raising=False)
        return served

    return pin


def test_the_first_start_downloads_checks_and_unpacks_the_build(pinned, tmp_path, capsys):
    served = pinned()
    executable = launcher.rust_executable()
    assert executable is not None
    assert executable == tmp_path / 'bin' / 'v9' / 'lazy-take-notes'
    assert executable.read_bytes() == b'#!/bin/sh\n'
    assert (executable.parent / 'coreaudio-tap').is_file()
    assert executable.stat().st_mode & 0o111
    assert served.urls == [f'{launcher.RELEASES}/v9/{ASSET}']
    assert 'Downloading lazy-take-notes v9 (once)...' in capsys.readouterr().err
    # The next start uses the cache.
    assert launcher.rust_executable() == executable
    assert len(served.urls) == 1
    assert not [p for p in (tmp_path / 'bin').iterdir() if p.name.startswith('.download-')]


@pytest.mark.parametrize(
    ('setup', 'reason'),
    [
        (lambda pin: pin(sha='0' * 64), 'does not match its pinned SHA-256'),
        (lambda pin: pin(error=urllib.error.URLError('offline')), 'cannot download'),
        (lambda pin: pin(error=TimeoutError('timed out')), 'cannot download'),
        (lambda pin: pin(body=b'not a tarball'), 'cannot unpack'),
        (lambda pin: pin(body=tarball({'../evil': b'x'}, folder='x')), 'unsafe path'),
        (lambda pin: pin(body=tarball({'other': b'x'})), 'does not hold lazy-take-notes'),
    ],
)
def test_any_setup_failure_runs_the_python_app_with_one_note(pinned, tmp_path, capsys, setup, reason):
    setup(pinned)
    assert launcher.rust_executable() is None
    err = capsys.readouterr().err
    assert 'Note: running the Python version; the Rust version is unavailable' in err
    assert reason in err
    assert not (tmp_path / 'bin' / 'v9').exists()


def test_a_link_in_the_download_is_refused(pinned, capsys):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w:gz') as tar:
        link = tarfile.TarInfo('lazy-take-notes-arm64-macos/lazy-take-notes')
        link.type = tarfile.SYMTYPE
        link.linkname = '/bin/sh'
        tar.addfile(link)
    pinned(body=buffer.getvalue())
    assert launcher.rust_executable() is None
    assert 'unexpected entry' in capsys.readouterr().err


def test_no_pin_or_no_build_for_this_machine_or_ltn_python_runs_python(pinned, monkeypatch):
    served = pinned()
    monkeypatch.setenv('LTN_PYTHON', '1')
    assert launcher.rust_executable() is None
    monkeypatch.delenv('LTN_PYTHON')
    monkeypatch.setattr(launcher.platform, 'system', lambda: 'Linux')
    assert launcher.rust_executable() is None
    monkeypatch.setattr(rust_release, 'TAG', None)
    assert launcher.pinned_asset() is None
    assert served.urls == []


def test_a_start_that_lost_the_race_uses_the_winners_install(pinned, tmp_path, monkeypatch):
    pinned()
    real_install = launcher.install

    def raced(tag, asset, sha256, root):
        (root / tag).mkdir(parents=True)
        (root / tag / 'lazy-take-notes').write_bytes(b'winner')
        return real_install(tag, asset, sha256, root)

    monkeypatch.setattr(launcher, 'install', raced)
    executable = launcher.rust_executable()
    assert executable is not None
    assert executable.read_bytes() == b'winner'


def test_a_failed_rename_with_nothing_installed_is_reported(pinned, monkeypatch, capsys):
    pinned()
    monkeypatch.setattr(Path, 'rename', lambda self, target: (_ for _ in ()).throw(PermissionError('denied')))
    assert launcher.rust_executable() is None
    assert 'denied' in capsys.readouterr().err


def test_a_waiting_start_finds_the_build_installed_under_the_lock(pinned, tmp_path):
    served = pinned()
    root = tmp_path / 'bin'
    (root / 'v9').mkdir(parents=True)
    (root / 'v9' / 'lazy-take-notes').write_bytes(b'done')
    assert launcher.locked_install('v9', ASSET, 'x', root) == root / 'v9' / 'lazy-take-notes'
    assert served.urls == []


def test_main_replaces_the_process_with_the_rust_build(pinned, monkeypatch):
    pinned()
    calls = []
    monkeypatch.setattr(launcher.sys, 'argv', ['take-note', 'record', '-l', 'x'])
    monkeypatch.setattr(launcher.os, 'execv', lambda path, argv: calls.append((path, argv)))
    monkeypatch.setattr(launcher, 'run_python', lambda: calls.append('python'))
    launcher.main()
    executable = launcher.rust_executable()
    assert calls == [(executable, [str(executable), 'record', '-l', 'x'])]


def test_main_runs_python_when_the_build_cannot_start(pinned, monkeypatch, capsys):
    pinned()
    calls = []

    def broken(path, argv):
        raise OSError(86, 'Bad CPU type in executable')

    monkeypatch.setattr(launcher.os, 'execv', broken)
    monkeypatch.setattr(launcher, 'run_python', lambda: calls.append('python'))
    launcher.main()
    assert calls == ['python']
    assert 'Bad CPU type' in capsys.readouterr().err


def test_main_without_a_build_runs_python(monkeypatch):
    calls = []
    monkeypatch.setattr(launcher, 'rust_executable', lambda: None)
    monkeypatch.setattr(launcher, 'run_python', lambda: calls.append('python'))
    launcher.main()
    assert calls == ['python']


def test_run_python_is_the_click_cli(monkeypatch):
    from lazy_take_notes.l4_frameworks_and_drivers import cli as cli_module

    calls = []
    monkeypatch.setattr(cli_module, 'cli', lambda: calls.append('cli'))
    launcher.run_python()
    assert calls == ['cli']


def test_the_cache_is_the_apps_cache_folder():
    assert launcher.cache_root().parts[-2:] == ('lazy-take-notes', 'bin')


def zip_asset(entries: dict[str, bytes], folder: str = 'lazy-take-notes-x86_64-windows') -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as zipped:
        for name, data in entries.items():
            zipped.writestr(f'{folder}/{name}', data)
    return buffer.getvalue()


def test_a_zip_asset_is_unpacked(pinned, tmp_path, monkeypatch):
    body = zip_asset({'lazy-take-notes.exe': b'MZ'})
    pinned(body=body)
    monkeypatch.setattr(launcher, 'BINARY', 'lazy-take-notes.exe')
    executable = launcher.install(
        'v9',
        'lazy-take-notes-x86_64-windows.zip',
        hashlib.sha256(body).hexdigest(),
        tmp_path / 'bin',
    )
    assert executable == tmp_path / 'bin' / 'v9' / 'lazy-take-notes.exe'
    assert executable.read_bytes() == b'MZ'


def test_a_zip_path_that_escapes_is_refused(tmp_path):
    archive = tmp_path / 'a.zip'
    with zipfile.ZipFile(archive, 'w') as zipped:
        zipped.writestr('../../evil', b'x')
    with pytest.raises(launcher.SetupError, match='unsafe path'):
        launcher.extract(archive, tmp_path / 'out')


def test_main_runs_the_windows_build_with_subprocess(pinned, tmp_path, monkeypatch):
    pinned()
    executable = tmp_path / 'bin' / 'v9' / 'lazy-take-notes.exe'
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b'MZ')
    monkeypatch.setattr(launcher, 'BINARY', 'lazy-take-notes.exe')
    monkeypatch.setattr(launcher, 'WINDOWS', True)
    monkeypatch.setattr(launcher.sys, 'argv', ['lazy-take-notes', 'status'])
    calls = []

    class Completed:
        returncode = 7

    monkeypatch.setattr(launcher.subprocess, 'run', lambda args, check: calls.append(args) or Completed())
    with pytest.raises(SystemExit) as stopped:
        launcher.main()
    assert calls == [[str(executable), 'status']]
    assert stopped.value.code == 7


def test_the_windows_lock_waits_then_installs(pinned, tmp_path, monkeypatch):
    served = pinned()
    monkeypatch.setattr(launcher, 'WINDOWS', True)
    fake: Any = types.ModuleType('msvcrt')
    fake.LK_NBLCK = 1
    fake.locking = lambda handle, mode, size: None
    monkeypatch.setitem(sys.modules, 'msvcrt', fake)
    executable = launcher.locked_install('v9', ASSET, hashlib.sha256(served.body).hexdigest(), tmp_path / 'bin')
    assert executable.read_bytes() == b'#!/bin/sh\n'
