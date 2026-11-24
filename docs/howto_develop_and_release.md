# Develop and release

The repository holds two apps. The Python app is in `src/` and `tests/`. The native app is in `rust/`. [Rust and Python versions](explain_rust_and_python.md) explains which one runs where.

## Work on the Python app

```bash
uv sync                    # install the dependencies
uv run pytest tests/ -v    # run the tests
uv run lint-imports        # check the layer rules
uv run ty check src tests  # check the types
```

[AGENTS.md](../AGENTS.md) describes the architecture.

## Work on the native app

The Rust toolchain is pinned in `rust/rust-toolchain.toml`. rustup installs it when you first build in `rust/`.

```bash
cd rust
cargo test                                  # the transcribe tests need ffmpeg
cargo clippy --all-targets -- -D warnings
cargo fmt --check
scripts/package.sh                          # build dist/lazy-take-notes-<arch>-macos.tar.gz
```

[rust/PLAN.md](../rust/PLAN.md) lists each way the native app differs from the Python app. Add a line there for every new difference.

## Release the native app

1. Run the Release workflow by hand: **Actions > Release > Run workflow**. This is a dry run: it builds and checks the package, keeps it as a workflow artifact, and publishes nothing.
2. Tag the commit and push the tag:

   ```bash
   git tag -a v0.3.0 -m "lazy-take-notes v0.3.0"
   git push origin v0.3.0
   ```

   The workflow publishes a GitHub Release with the package and its SHA-256. A tag with a `-`, such as `v0.3.0-rc.1`, is a prerelease.
3. Pin the release on a branch and test it:

   ```bash
   uv run scripts/pin_rust_release.py v0.3.0
   ```

   Commit `src/lazy_take_notes/rust_release.py`, push the branch, and run `uvx --from git+https://github.com/CJHwong/lazy-take-notes.git@<branch> take-note --version`.
4. Merge the branch. Users get the release when the pin reaches `main`.

Never move a tag that a pin uses. The pin holds the package's SHA-256, so a rebuilt package fails the check. Release a new tag instead.
