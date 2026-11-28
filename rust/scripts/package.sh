#!/usr/bin/env bash
# Build a release and, on macOS, put coreaudio-tap next to the executable.
# A release build looks for the helper only beside itself.
#
#   rust/scripts/package.sh     -> rust/dist/lazy-take-notes-<arch>-<os>.tar.gz
#
# The helper is the one the Python package ships. If it is missing on macOS,
# this runs scripts/build_native.sh, which needs the Xcode Command Line Tools.

set -euo pipefail

RUST_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(cd "$RUST_DIR/.." && pwd)"
TAP="$REPO_ROOT/src/lazy_take_notes/_native/bin/coreaudio-tap"

case "$(uname -s)" in
    Darwin) OS=macos ;;
    Linux) OS=linux ;;
    *)
        echo "ERROR: unsupported system: $(uname -s)" >&2
        exit 1
        ;;
esac
case "$(uname -m)" in
    x86_64) ARCH=x86_64 ;;
    aarch64 | arm64) ARCH=arm64 ;;
    *)
        echo "ERROR: unsupported machine: $(uname -m)" >&2
        exit 1
        ;;
esac

NAME="lazy-take-notes-$ARCH-$OS"
STAGE="$RUST_DIR/dist/$NAME"

if [ "$OS" = macos ] && [ ! -x "$TAP" ]; then
    bash "$REPO_ROOT/scripts/build_native.sh"
fi

# whisper.cpp builds with -march=native by default. A release built for the
# runner's CPU can crash on an older one, so take the baseline and add the
# instruction sets every x86_64 machine of that release needs.
export GGML_NATIVE=OFF
if [ "$ARCH" = x86_64 ]; then
    export GGML_AVX=ON GGML_AVX2=ON GGML_FMA=ON GGML_F16C=ON
fi

# rustup picks the toolchain from the working folder, so build from rust/:
# rust-toolchain.toml pins it.
(cd "$RUST_DIR" && cargo build --release)

mkdir -p "$STAGE"
cp "$RUST_DIR/target/release/lazy-take-notes" "$STAGE/"
if [ "$OS" = macos ]; then
    cp "$TAP" "$STAGE/"
fi
# COPYFILE_DISABLE: no ._ AppleDouble files for the extended attributes.
COPYFILE_DISABLE=1 tar -czf "$RUST_DIR/dist/$NAME.tar.gz" -C "$RUST_DIR/dist" "$NAME"
if command -v shasum >/dev/null 2>&1; then
    (cd "$RUST_DIR/dist" && shasum -a 256 "$NAME.tar.gz" >"$NAME.tar.gz.sha256")
else
    (cd "$RUST_DIR/dist" && sha256sum "$NAME.tar.gz" >"$NAME.tar.gz.sha256")
fi
echo "$RUST_DIR/dist/$NAME.tar.gz"
