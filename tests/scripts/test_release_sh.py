"""Behavioural tests for the macOS release script.

The script is a release tool, so these tests run it for real against a throwaway
tree and stub the macOS tooling it shells out to. The real ``plutil`` is used,
because the check of the Info.plist usage strings is the part worth proving.
"""

from __future__ import annotations

import os
import stat
import subprocess  # noqa: S404 -- runs the script under test
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parents[2]
RELEASE_SCRIPT = REPO_ROOT / 'scripts' / 'release.sh'
ENTITLEMENTS = REPO_ROOT / 'rust' / 'scripts' / 'app.entitlements'

pytestmark = pytest.mark.skipif(
    os.name != 'posix' or os.uname().sysname != 'Darwin',
    reason='the release script targets macOS and needs the system plutil',
)

APP_RELATIVE = 'rust/target/app/Lazy Take Notes.app'
USAGE_KEYS = """    <key>NSMicrophoneUsageDescription</key>
    <string>mic</string>
    <key>NSAudioCaptureUsageDescription</key>
    <string>audio</string>
"""
PLIST_XML = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleIdentifier</key>
    <string>com.cjhwong.lazy-take-notes</string>
{usage}</dict>
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

        (root / 'rust' / 'scripts').mkdir(parents=True)
        (root / 'rust' / 'scripts' / 'app.entitlements').write_text(ENTITLEMENTS.read_text())
        # The real package_app.sh runs cargo; this stand-in makes the bundle it would.
        _executable(
            root / 'rust' / 'scripts' / 'package_app.sh',
            '#!/bin/sh\n'
            '[ "${FAKE_SKIP_BUILD:-0}" = "1" ] && exit 0\n'
            f'cd "{root}"\n'
            f'mkdir -p "{APP_RELATIVE}/Contents/MacOS" "{APP_RELATIVE}/Contents/Resources"\n'
            f'printf "#!/bin/sh\\n" > "{APP_RELATIVE}/Contents/MacOS/lazy-take-notes"\n'
            f'printf "#!/bin/sh\\n" > "{APP_RELATIVE}/Contents/MacOS/coreaudio-tap"\n'
            'if [ "${FAKE_NO_USAGE:-0}" = "1" ]; then\n'
            f'cat > "{APP_RELATIVE}/Contents/Info.plist" <<\'PLIST\'\n{PLIST_XML.format(usage="")}PLIST\n'
            'else\n'
            f'cat > "{APP_RELATIVE}/Contents/Info.plist" <<\'PLIST\'\n{PLIST_XML.format(usage=USAGE_KEYS)}PLIST\n'
            'fi\n'
            'exit 0\n',
        )

        self._write_project_files('0.1.0', rust_version='0.1.0')
        self._write_tools()

    def _write_project_files(self, version: str, *, rust_version: str) -> None:
        self.project_version = version
        (self.root / 'pyproject.toml').write_text(f'[project]\nname = "x"\nversion = "{version}"\n')
        (self.root / 'rust' / 'Cargo.toml').write_text(f'[package]\nname = "x"\nversion = "{rust_version}"\n')

    def _write_tools(self) -> None:
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
        return self.root / 'rust' / 'target' / 'release-app' / f'LazyTakeNotes-{self.project_version}.zip'


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


def test_a_bundle_without_the_usage_strings_aborts(tree: Tree) -> None:
    result = tree.run(FAKE_NO_USAGE='1')

    assert result.returncode == 1
    assert 'NSMicrophoneUsageDescription' in result.stderr


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


def test_signing_covers_the_audio_helper_then_the_bundle(tree: Tree) -> None:
    result = tree.run(LTN_SIGN_IDENTITY='Developer ID Application: Test (TEAMID)')

    assert result.returncode == 0, result.stderr
    signed = [line for line in tree.calls.splitlines() if line.startswith('codesign --force')]
    assert len(signed) == 2
    assert 'coreaudio-tap' in signed[0]
    assert signed[1].rstrip().endswith('Lazy Take Notes.app')
    # The hardened runtime blocks the microphone without the entitlement.
    assert all('--options runtime' in line and 'app.entitlements' in line for line in signed)


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
    assert (tree.root / 'rust' / 'target' / 'release-app' / f'LazyTakeNotes-{tree.project_version}.dmg').is_file()


def test_missing_bundle_aborts(tree: Tree) -> None:
    result = tree.run(FAKE_SKIP_BUILD='1')

    assert result.returncode == 1
    assert 'is missing' in result.stderr


def test_version_mismatch_aborts(tree: Tree) -> None:
    tree._write_project_files('0.2.0', rust_version='0.1.0')

    result = tree.run()

    assert result.returncode == 1
    assert 'but rust/Cargo.toml says' in result.stderr


def test_unknown_argument_is_rejected(tree: Tree) -> None:
    result = tree.run('--wat')

    assert result.returncode == 2
