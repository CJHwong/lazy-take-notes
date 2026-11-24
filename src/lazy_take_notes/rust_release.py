"""The Rust build the launcher runs. `scripts/pin_rust_release.py <tag>` writes this file.

TAG None means no release is pinned: the launcher runs the Python app.
"""

TAG: str | None = 'v0.2.0'

# (platform.system(), platform.machine()) -> (release asset name, SHA-256 of the asset)
ASSETS: dict[tuple[str, str], tuple[str, str]] = {
    ('Darwin', 'arm64'): (
        'lazy-take-notes-arm64-macos.tar.gz',
        'd839eb744a8f5352df66ed23927dde392ea55e9847e9cedf7a9192ea81c4bd82',
    ),
}
