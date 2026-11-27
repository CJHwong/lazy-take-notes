"""The Rust build the launcher runs. `scripts/pin_rust_release.py <tag>` writes this file.

TAG None means no release is pinned: the launcher runs the Python app.
"""

TAG: str | None = 'v0.3.5'

# (platform.system(), platform.machine()) -> (release asset name, SHA-256 of the asset)
ASSETS: dict[tuple[str, str], tuple[str, str]] = {
    ('Darwin', 'arm64'): (
        'lazy-take-notes-arm64-macos.tar.gz',
        'a18e770e5ca413e20a53f083f576d436036dc6d25d7cc99b06b4ba2dd39f9292',
    ),
}
