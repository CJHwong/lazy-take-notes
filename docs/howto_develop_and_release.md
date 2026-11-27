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
scripts/package.sh                          # macOS and Linux: dist/lazy-take-notes-<arch>-<os>.tar.gz
powershell scripts/package.ps1              # Windows: dist/lazy-take-notes-x86_64-windows.zip
```

[rust/PLAN.md](../rust/PLAN.md) lists each way the native app differs from the Python app. Add a line there for every new difference.

## Release the native app

1. Set `version` in `rust/Cargo.toml` to the new version, and commit it with `rust/Cargo.lock`. The `.app` takes its version from this file, and the Homebrew cask takes its version from the tag, so the two must match.
2. Run the Release workflow by hand: **Actions > Release > Run workflow**. This is a dry run: it builds and checks the packages, keeps them as a workflow artifact, and publishes nothing.
3. Tag the commit and push the tag:

   ```bash
   git tag -a v0.3.0 -m "lazy-take-notes v0.3.0"
   git push origin v0.3.0
   ```

   The workflow publishes a GitHub Release with the terminal packages for macOS, Linux and Windows, `Lazy-Take-Notes-arm64-macos.zip` (the window app), and their SHA-256 files. The Windows package is built on `windows-2022` by `rust/scripts/package.ps1`; the dry run keeps its zip as an artifact too. The workflow then writes `Casks/lazy-take-notes.rb` to `CJHwong/homebrew-tap` with the `HOMEBREW_TAP_KEY` deploy key. A tag with a `-`, such as `v0.3.0-rc.1`, is a prerelease and leaves the cask alone.
4. Pin the release on a branch and test it:

   ```bash
   uv run scripts/pin_rust_release.py v0.3.0
   ```

   Commit `src/lazy_take_notes/rust_release.py`, push the branch, and run `uvx --from git+https://github.com/CJHwong/lazy-take-notes.git@<branch> take-note --version`. On Windows, run the same command with the pinned launcher and check that `--version` prints the version.
5. Merge the branch. Users of the Python launcher get the release when the pin reaches `main`.

The release job signs the `.app` with a self-signed certificate from the `MACOS_SIGN_P12` and `MACOS_SIGN_P12_PASSWORD` secrets, and a tag fails without them. macOS keeps an app's privacy permissions (microphone, system audio, Documents) for its bundle id and certificate. Keep the same certificate: a new one makes every user grant the permissions again. A local build of `rust/scripts/package_app.sh` is signed ad hoc, unless `LTN_SIGN_IDENTITY` (and `LTN_SIGN_KEYCHAIN`) name an identity.

Never move a tag that a pin uses. The pin holds the package's SHA-256, so a rebuilt package fails the check. Release a new tag instead.
