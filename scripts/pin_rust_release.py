# /// script
# requires-python = ">=3.11"
# ///
"""Pin a published Rust release in src/lazy_take_notes/rust_release.py.

    uv run scripts/pin_rust_release.py v0.2.0

Downloads the release's tarballs with `gh`, hashes them, and writes each one
this script knows a platform for. The launcher runs only what is pinned here.
"""

import hashlib
import subprocess  # noqa: S404 -- fixed gh argument list
import sys
import tempfile
from pathlib import Path

# Release asset -> (platform.system(), platform.machine()) on the machines it runs on.
PLATFORMS = {
    'lazy-take-notes-arm64-macos.tar.gz': ('Darwin', 'arm64'),
}
TARGET = Path(__file__).resolve().parent.parent / 'src' / 'lazy_take_notes' / 'rust_release.py'


def main(tag: str) -> None:
    with tempfile.TemporaryDirectory() as folder:
        subprocess.run(  # noqa: S603 -- fixed arg list, not shell=True
            ['gh', 'release', 'download', tag, '--pattern', '*.tar.gz', '--dir', folder],  # noqa: S607 -- gh on PATH
            check=True,
        )
        assets = {}
        for archive in sorted(Path(folder).glob('*.tar.gz')):
            key = PLATFORMS.get(archive.name)
            if key is None:
                print(f'skipping {archive.name}: no platform known for it', file=sys.stderr)
                continue
            assets[key] = (archive.name, hashlib.sha256(archive.read_bytes()).hexdigest())
    if not assets:
        sys.exit(f'{tag} has no asset this script knows')
    lines = [f'    {key!r}: {value!r},' for key, value in sorted(assets.items())]
    text = TARGET.read_text(encoding='utf-8')
    head = text.split('TAG:', 1)[0]
    TARGET.write_text(
        head
        + f'TAG: str | None = {tag!r}\n\n'
        + '# (platform.system(), platform.machine()) -> (release asset name, SHA-256 of the asset)\n'
        + 'ASSETS: dict[tuple[str, str], tuple[str, str]] = {\n'
        + '\n'.join(lines)
        + '\n}\n',
        encoding='utf-8',
    )
    print(f'pinned {tag}: {", ".join(name for name, _ in assets.values())}')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit('usage: pin_rust_release.py <tag>')
    main(sys.argv[1])
