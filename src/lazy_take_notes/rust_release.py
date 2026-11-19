"""The Rust build the launcher runs. `scripts/pin_rust_release.py <tag>` writes this file.

TAG None means no release is pinned: the launcher runs the Python app.
"""

TAG: str | None = None

# (platform.system(), platform.machine()) -> (release asset name, SHA-256 of the asset)
ASSETS: dict[tuple[str, str], tuple[str, str]] = {}
