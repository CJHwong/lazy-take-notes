"""Behavioural tests for the Trolley launcher.

Trolley execs this script and nothing else, so the tests run the script for
real against a fake ``uv`` and a fake installed console script. No terminal,
no Trolley, no network.
"""

from __future__ import annotations

import os
import stat
import subprocess  # noqa: S404 -- launches the launcher under test
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parents[2]
LAUNCHER = REPO_ROOT / 'scripts' / 'trolley' / 'ltn-launch'

pytestmark = pytest.mark.skipif(
    os.name != 'posix',
    reason='the launcher is a POSIX shell script shipped only in the macOS bundle',
)

FAKE_UV = """#!/bin/sh
set -eu
case "$1 ${2:-}" in
    "tool dir")
        printf '%s\\n' "$FAKE_TOOLS_DIR"
        ;;
    "tool install")
        printf '%s\\n' "$*" >> "$FAKE_LOG"
        if [ "${FAKE_SKIP_INSTALL:-0}" = "1" ]; then
            exit 0
        fi
        mkdir -p "$FAKE_TOOLS_DIR/lazy-take-notes/bin"
        cp "$FAKE_APP_BIN" "$FAKE_TOOLS_DIR/lazy-take-notes/bin/lazy-take-notes"
        chmod +x "$FAKE_TOOLS_DIR/lazy-take-notes/bin/lazy-take-notes"
        ;;
    *)
        printf 'unexpected uv call: %s\\n' "$*" >&2
        exit 2
        ;;
esac
"""

FAKE_APP = """#!/bin/sh
printf 'APP_RAN\\n'
"""


def write_executable(path: Path, body: str) -> Path:
    """Write *body* to *path* and mark it executable."""
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


class Sandbox:
    """A throwaway HOME, a fake uv, and the paths the launcher will touch."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.home = root / 'home'
        self.tools_dir = root / 'uv-tools'
        self.log = root / 'uv-calls.log'
        self.app_bin = write_executable(root / 'fake-app', FAKE_APP)
        self.uv = write_executable(root / 'fake-uv', FAKE_UV)
        self.home.mkdir()

    def installed_app(self, body: str = FAKE_APP) -> Path:
        """Pre-create an installed console script, as a previous launch would."""
        app_bin = self.tools_dir / 'lazy-take-notes' / 'bin' / 'lazy-take-notes'
        app_bin.parent.mkdir(parents=True)
        return write_executable(app_bin, body)

    def run(
        self, *, uv: Path | None = None, extra_env: dict[str, str] | None = None
    ) -> subprocess.CompletedProcess[str]:
        """Run the launcher with *uv* discoverable only through LTN_UV."""
        env = {
            'HOME': str(self.home),
            'PATH': '/usr/bin:/bin',
            'LTN_UV': str(uv if uv is not None else self.uv),
            'FAKE_TOOLS_DIR': str(self.tools_dir),
            'FAKE_APP_BIN': str(self.app_bin),
            'FAKE_LOG': str(self.log),
            **(extra_env or {}),
        }
        return subprocess.run(  # noqa: S603 -- fixed arg list, no shell
            ['/bin/sh', str(LAUNCHER)],
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )

    @property
    def install_calls(self) -> list[str]:
        """Every `uv tool install` invocation the fake uv recorded."""
        if not self.log.exists():
            return []
        return self.log.read_text().splitlines()


@pytest.fixture
def sandbox(tmp_path: Path) -> Sandbox:
    return Sandbox(tmp_path)


def test_launcher_has_valid_shell_syntax() -> None:
    result = subprocess.run(  # noqa: S603 -- fixed arg list, no shell
        ['/bin/sh', '-n', str(LAUNCHER)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_launcher_is_executable() -> None:
    assert LAUNCHER.stat().st_mode & stat.S_IXUSR


def test_installs_on_first_run_then_execs_the_app(sandbox: Sandbox) -> None:
    result = sandbox.run()

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == 'APP_RAN'
    assert len(sandbox.install_calls) == 1


def test_install_uses_the_configured_source(sandbox: Sandbox) -> None:
    sandbox.run(extra_env={'LTN_DESKTOP_SOURCE': 'git+https://example.test/fork.git'})

    assert 'git+https://example.test/fork.git' in sandbox.install_calls[0]


def test_existing_install_is_reused_without_network(sandbox: Sandbox) -> None:
    sandbox.installed_app()

    result = sandbox.run()

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == 'APP_RAN'
    assert sandbox.install_calls == []


def test_missing_uv_explains_how_to_install(sandbox: Sandbox) -> None:
    """A broken LTN_UV must not surface as a raw shell error."""
    missing = sandbox.root / 'does-not-exist'

    result = sandbox.run(uv=missing)

    assert result.returncode == 1
    assert str(missing) in result.stderr
    assert 'astral.sh/uv/install.sh' in result.stderr


def test_missing_uv_does_not_try_to_install_anything(sandbox: Sandbox) -> None:
    sandbox.run(uv=sandbox.root / 'does-not-exist')

    assert sandbox.install_calls == []


def test_install_that_produces_no_binary_is_reported(sandbox: Sandbox) -> None:
    """A uv that claims success but leaves no console script must not exec nothing."""
    result = sandbox.run(extra_env={'FAKE_SKIP_INSTALL': '1'})

    assert result.returncode == 1
    assert 'is missing' in result.stderr
