#!/usr/bin/env bash
# Build a release and put coreaudio-tap next to the executable.
# A release build looks for the helper only beside itself.
#
#   rust/scripts/package.sh     -> rust/dist/lazy-take-notes-<arch>-macos.tar.gz
#
# The helper is the one the Python package ships. If it is missing, this runs
# scripts/build_native.sh, which needs the Xcode Command Line Tools.

set -euo pipefail

RUST_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(cd "$RUST_DIR/.." && pwd)"
TAP="$REPO_ROOT/src/lazy_take_notes/_native/bin/coreaudio-tap"
NAME="lazy-take-notes-$(uname -m)-macos"
STAGE="$RUST_DIR/dist/$NAME"

if [ "$(uname -s)" != "Darwin" ]; then
    echo "ERROR: coreaudio-tap is macOS only" >&2
    exit 1
fi

if [ ! -x "$TAP" ]; then
    bash "$REPO_ROOT/scripts/build_native.sh"
fi

# rustup picks the toolchain from the working folder, so build from rust/:
# rust-toolchain.toml pins it.
(cd "$RUST_DIR" && cargo build --release)

mkdir -p "$STAGE"
cp "$RUST_DIR/target/release/lazy-take-notes" "$STAGE/"
cp "$TAP" "$STAGE/"
# COPYFILE_DISABLE: no ._ AppleDouble files for the extended attributes.
COPYFILE_DISABLE=1 tar -czf "$RUST_DIR/dist/$NAME.tar.gz" -C "$RUST_DIR/dist" "$NAME"
echo "$RUST_DIR/dist/$NAME.tar.gz"
