"""
Validate that build_app_bundle.sh produces a launcher with valid bash syntax.

These tests are macOS-only: the script itself uses macOS-specific tools
(osascript, CoreAudio) and would fail on Linux regardless.
"""

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent.parent
BUILD_SCRIPT = REPO_ROOT / 'scripts' / 'build_app_bundle.sh'
LAUNCHER = REPO_ROOT / 'build' / 'LazyTakeNotes.app' / 'Contents' / 'MacOS' / 'launcher'

pytestmark = pytest.mark.skipif(
    sys.platform != 'darwin',
    reason='app bundle is macOS-only',
)


@pytest.fixture(scope='module')
def built_launcher():
    """Run the build script once and yield the launcher path."""
    result = subprocess.run(  # noqa: S603 -- fixed arg list, not shell=True
        ['bash', str(BUILD_SCRIPT)],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, f'build_app_bundle.sh failed:\n{result.stderr}'
    assert LAUNCHER.exists(), 'launcher was not created'
    return LAUNCHER


def test_launcher_bash_syntax(built_launcher):
    """bash -n must pass — catches missing fi/done/esac in heredoc-generated scripts."""
    result = subprocess.run(  # noqa: S603 -- fixed arg list, not shell=True
        ['bash', '-n', str(built_launcher)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f'launcher has bash syntax errors:\n{result.stderr}'


def test_launcher_is_executable(built_launcher):
    """The launcher must have its executable bit set so macOS can run it."""
    assert built_launcher.stat().st_mode & 0o111, 'launcher is not executable'


def test_launcher_contains_path_helper(built_launcher):
    """path_helper invocation must be present in the launcher."""
    source = built_launcher.read_text()
    assert 'path_helper' in source, 'path_helper invocation missing from launcher'
