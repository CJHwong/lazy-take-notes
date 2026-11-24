"""The Rust build the launcher runs. `scripts/pin_rust_release.py <tag>` writes this file.

TAG None means no release is pinned: the launcher runs the Python app.
"""

TAG: str | None = 'v0.2.0-rc.1'

# (platform.system(), platform.machine()) -> (release asset name, SHA-256 of the asset)
ASSETS: dict[tuple[str, str], tuple[str, str]] = {
    ('Darwin', 'arm64'): (
        'lazy-take-notes-arm64-macos.tar.gz',
        'cfbdee0dcbef3f53b3a698382510e1eb3853327e45a9305788fee20b4fb8935f',
    ),
}
