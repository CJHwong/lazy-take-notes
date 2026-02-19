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

echo "Building arm64..."
swift build -c release --arch arm64 2>&1

echo "Building x86_64..."
swift build -c release --arch x86_64 2>&1

echo "Creating universal binary..."
lipo -create \
  -output "$BINARY" \
  ".build/arm64-apple-macosx/release/coreaudio-tap" \
  ".build/x86_64-apple-macosx/release/coreaudio-tap"

chmod +x "$BINARY"
echo "Built: $BINARY"
echo "Set executable bit in git with:"
echo "  git update-index --chmod=+x src/lazy_take_notes/_native/bin/coreaudio-tap"
