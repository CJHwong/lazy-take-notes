"""Behavioural tests for the macOS release script.

The script is a release tool, so these tests run it for real against a throwaway
tree and stub the macOS tooling it shells out to. The real ``plutil`` is used,
because the Info.plist patch is the part worth proving.
"""

from __future__ import annotations

import os
import stat
import subprocess  # noqa: S404 -- runs the script under test
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parents[2]
RELEASE_SCRIPT = REPO_ROOT / 'scripts' / 'release.sh'

pytestmark = pytest.mark.skipif(
    os.name != 'posix' or os.uname().sysname != 'Darwin',
    reason='the release script targets macOS and needs the system plutil',
)

APP_RELATIVE = 'trolley/build/com.cjhwong.lazy-take-notes/aarch64-macos/dist/Lazy Take Notes.app'
PLIST_XML = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleIdentifier</key>
    <string>com.cjhwong.lazy-take-notes</string>
</dict>
</plist>
"""


def _executable(path: Path, body: str) -> Path:
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


class Tree:
    """A throwaway repo with stub macOS tools on PATH."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.bin = root / 'bin'
        self.log = root / 'calls.log'
        self.home = root / 'home'
        self.bin.mkdir()
        self.home.mkdir()

        (root / 'scripts').mkdir()
        (root / 'scripts' / 'release.sh').write_text(RELEASE_SCRIPT.read_text())
        (root / 'scripts' / 'release.sh').chmod(0o755)

        self._write_project_files('0.1.0', trolley_version='0.1.0')
        self._write_tools()

    def _write_project_files(self, version: str, *, trolley_version: str) -> None:
        self.project_version = version
        (self.root / 'pyproject.toml').write_text(f'[project]\nname = "x"\nversion = "{version}"\n')
        (self.root / 'trolley.toml').write_text(f'[app]\nversion = "{trolley_version}"\n')

    def _write_tools(self) -> None:
        _executable(
            self.bin / 'trolley',
            '#!/bin/sh\n'
            '[ "${FAKE_SKIP_BUILD:-0}" = "1" ] && exit 0\n'
            f'mkdir -p "{APP_RELATIVE}/Contents/MacOS" "{APP_RELATIVE}/Contents/Resources"\n'
            f'printf "#!/bin/sh\\n" > "{APP_RELATIVE}/Contents/MacOS/lazy-take-notes_runtime"\n'
            f'printf "#!/bin/sh\\n" > "{APP_RELATIVE}/Contents/MacOS/lazy-take-notes_core"\n'
            f'cat > "{APP_RELATIVE}/Contents/Info.plist" <<\'PLIST\'\n{PLIST_XML}PLIST\n'
            'exit 0\n',
        )
        _executable(
            self.bin / 'security',
            '#!/bin/sh\n[ -n "${FAKE_IDENTITY:-}" ] && printf \'  1) ABC123 "%s"\\n\' "$FAKE_IDENTITY"\nexit 0\n',
        )
        for tool in ('codesign', 'xcrun', 'spctl'):
            _executable(self.bin / tool, f'#!/bin/sh\nprintf "{tool} $*\\n" >> "$FAKE_LOG"\nexit 0\n')
        _executable(
            self.bin / 'ditto',
            '#!/bin/sh\nprintf "ditto $*\\n" >> "$FAKE_LOG"\nfor last; do :; done\nprintf "zip" > "$last"\nexit 0\n',
        )
        _executable(
            self.bin / 'hdiutil',
            '#!/bin/sh\nprintf "hdiutil $*\\n" >> "$FAKE_LOG"\nfor last; do :; done\nprintf "dmg" > "$last"\nexit 0\n',
        )

    def run(self, *args: str, **env: str) -> subprocess.CompletedProcess[str]:
        merged = {
            'HOME': str(self.home),
            'PATH': f'{self.bin}:/usr/bin:/bin',
            'FAKE_LOG': str(self.log),
            'LTN_SIGN_IDENTITY': '',
            'LTN_NOTARY_PROFILE': '',
            **env,
        }
        return subprocess.run(  # noqa: S603 -- fixed arg list, no shell
            ['/bin/bash', str(self.root / 'scripts' / 'release.sh'), *args],
            capture_output=True,
            text=True,
            env=merged,
            cwd=self.root,
            check=False,
        )

    @property
    def calls(self) -> str:
        return self.log.read_text() if self.log.exists() else ''

    @property
    def app(self) -> Path:
        return self.root / APP_RELATIVE

    @property
    def zip_path(self) -> Path:
        return self.root / 'trolley' / 'release' / f'LazyTakeNotes-{self.project_version}.zip'


@pytest.fixture
def tree(tmp_path: Path) -> Tree:
    return Tree(tmp_path)


def test_script_has_valid_shell_syntax() -> None:
    result = subprocess.run(  # noqa: S603 -- fixed arg list, no shell
        ['/bin/bash', '-n', str(RELEASE_SCRIPT)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_plist_gets_the_microphone_usage_description(tree: Tree) -> None:
    result = tree.run()

    assert result.returncode == 0, result.stderr
    extracted = subprocess.run(  # noqa: S603 -- fixed arg list, no shell
        ['plutil', '-extract', 'NSMicrophoneUsageDescription', 'raw', str(tree.app / 'Contents' / 'Info.plist')],
        capture_output=True,
        text=True,
        check=False,
    )
    assert extracted.stdout.strip()


def test_unsigned_build_still_produces_an_archive(tree: Tree) -> None:
    result = tree.run()

    assert result.returncode == 0, result.stderr
    assert tree.zip_path.is_file()


def test_unsigned_build_says_so(tree: Tree) -> None:
    result = tree.run()

    assert 'UNSIGNED' in result.stdout


def test_no_signing_without_an_identity(tree: Tree) -> None:
    tree.run()

    assert 'codesign' not in tree.calls


def test_signing_covers_the_core_runtime_then_the_bundle(tree: Tree) -> None:
    result = tree.run(LTN_SIGN_IDENTITY='Developer ID Application: Test (TEAMID)')

    assert result.returncode == 0, result.stderr
    signed = [line for line in tree.calls.splitlines() if line.startswith('codesign --force')]
    assert 'lazy-take-notes_core' in signed[0]
    assert 'lazy-take-notes_runtime' in signed[1]
    assert signed[2].rstrip().endswith('Lazy Take Notes.app')


def test_identity_is_detected_from_the_keychain(tree: Tree) -> None:
    result = tree.run(FAKE_IDENTITY='Developer ID Application: Detected (TEAMID)')

    assert result.returncode == 0, result.stderr
    assert 'Developer ID Application: Detected (TEAMID)' in tree.calls


def test_notarization_runs_when_a_profile_is_configured(tree: Tree) -> None:
    result = tree.run(
        LTN_SIGN_IDENTITY='Developer ID Application: Test (TEAMID)',
        LTN_NOTARY_PROFILE='ltn',
    )

    assert result.returncode == 0, result.stderr
    assert 'notarytool submit' in tree.calls
    assert 'stapler staple' in tree.calls
    assert 'signed and notarized' in result.stdout


def test_notarization_is_skipped_without_an_identity(tree: Tree) -> None:
    result = tree.run(LTN_NOTARY_PROFILE='ltn')

    assert result.returncode == 0, result.stderr
    assert 'notarytool' not in tree.calls
    assert 'Skipping notarization' in result.stderr


def test_dmg_flag_builds_a_disk_image(tree: Tree) -> None:
    result = tree.run('--dmg')

    assert result.returncode == 0, result.stderr
    assert 'hdiutil' in tree.calls
    assert (tree.root / 'trolley' / 'release' / f'LazyTakeNotes-{tree.project_version}.dmg').is_file()


def test_missing_bundle_aborts(tree: Tree) -> None:
    result = tree.run(FAKE_SKIP_BUILD='1')

    assert result.returncode == 1
    assert 'is missing' in result.stderr


def test_version_mismatch_aborts(tree: Tree) -> None:
    tree._write_project_files('0.2.0', trolley_version='0.1.0')

    result = tree.run()

    assert result.returncode == 1
    assert 'does not match' in result.stderr or 'but trolley.toml says' in result.stderr


def test_unknown_argument_is_rejected(tree: Tree) -> None:
    result = tree.run('--wat')

    assert result.returncode == 2
