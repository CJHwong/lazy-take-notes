"""The Rust build the launcher runs. `scripts/pin_rust_release.py <tag>` writes this file.

TAG None means no release is pinned: the launcher runs the Python app.
"""

TAG: str | None = 'v0.5.1'

# (platform.system(), platform.machine()) -> (release asset name, SHA-256 of the asset)
ASSETS: dict[tuple[str, str], tuple[str, str]] = {
    ('Darwin', 'arm64'): (
        'lazy-take-notes-arm64-macos.tar.gz',
        '5fcdfe05bd62e99914795b0180daa8e7e8491dd3d40d0d3aba98d6d8066898c2',
    ),
    ('Linux', 'aarch64'): (
        'lazy-take-notes-arm64-linux.tar.gz',
        '9dceeea45ee1d7b6d85d11b1f4a7b8503cae68cf6c148ef84acfb75d50b97b4b',
    ),
    ('Linux', 'x86_64'): (
        'lazy-take-notes-x86_64-linux.tar.gz',
        '8b830a754fc41eb175684701b90dfc0ed73df543b63a526a8b77625080cc063f',
    ),
    ('Windows', 'AMD64'): (
        'lazy-take-notes-x86_64-windows.zip',
        '4c061d6ea79cd09d3e5c3029865c222889d5d0232cff11ed46d85542dbe20b68',
    ),
}
