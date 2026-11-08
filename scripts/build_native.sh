#!/usr/bin/env bash
# Build the coreaudio-tap Swift binary as a universal (arm64 + x86_64) macOS binary.
# Requires: Xcode Command Line Tools (xcode-select --install)
# Output: src/lazy_take_notes/_native/bin/coreaudio-tap

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SOURCE_DIR="$REPO_ROOT/native/coreaudio_tap"
BINARY="$REPO_ROOT/src/lazy_take_notes/_native/bin/coreaudio-tap"

cd "$SOURCE_DIR"

# SwiftPM resolves its product directory itself. Newer toolchains share one
# output directory across --arch builds, so the old hardcoded
# .build/<arch>-apple-macosx/release paths either do not exist or hand lipo a
# stale binary twice (once per arch) and still exit 0.
STAGE=$(mktemp -d)
trap 'rm -r "$STAGE"' EXIT

for arch in arm64 x86_64; do
    echo "Building $arch..."
    swift build -c release --arch "$arch"
    product_dir=$(swift build -c release --arch "$arch" --show-bin-path)
    if [ ! -f "$product_dir/coreaudio-tap" ]; then
        echo "ERROR: $product_dir/coreaudio-tap was not produced" >&2
        exit 1
    fi
    # Copy now: the next arch build overwrites the shared product directory.
    cp "$product_dir/coreaudio-tap" "$STAGE/$arch"
done

echo "Creating universal binary..."
lipo -create -output "$BINARY" "$STAGE/arm64" "$STAGE/x86_64"
# This lipo takes one arch per call, unlike older versions.
lipo -verify_arch arm64 "$BINARY"
lipo -verify_arch x86_64 "$BINARY"

chmod +x "$BINARY"
echo "Built: $BINARY"
echo "Set executable bit in git with:"
echo "  git update-index --chmod=+x src/lazy_take_notes/_native/bin/coreaudio-tap"
