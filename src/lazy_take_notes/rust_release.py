"""The Rust build the launcher runs. `scripts/pin_rust_release.py <tag>` writes this file.

TAG None means no release is pinned: the launcher runs the Python app.
"""

TAG: str | None = 'v0.5.0'

# (platform.system(), platform.machine()) -> (release asset name, SHA-256 of the asset)
ASSETS: dict[tuple[str, str], tuple[str, str]] = {
    ('Darwin', 'arm64'): (
        'lazy-take-notes-arm64-macos.tar.gz',
        '0b4dc4413a3a95025bb392c85b779a3a4bb90593bdc910242c793936ffc45505',
    ),
    ('Linux', 'aarch64'): (
        'lazy-take-notes-arm64-linux.tar.gz',
        '50f7ad08cd8a658b3126dfde469d59cf1bfbc7966bd2bd739ca585a3095faeb3',
    ),
    ('Linux', 'x86_64'): (
        'lazy-take-notes-x86_64-linux.tar.gz',
        'de7ff2c6d0f041c4b87f74f9d40096cc971ac04a382ab95321e996d49f480a80',
    ),
    ('Windows', 'AMD64'): (
        'lazy-take-notes-x86_64-windows.zip',
        'f9b53f5d8e23a97507a2f47fed3d9ce9006cce573c81e75840b219e59d0acb17',
    ),
}
