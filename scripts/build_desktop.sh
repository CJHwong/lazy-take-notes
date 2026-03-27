#!/usr/bin/env bash
# Build lazy-take-notes as a native desktop app via PyInstaller + Trolley.
#
# Prerequisites:
#   brew install weedonandscott/tap/trolley
#   uv sync
#   uv pip install pyinstaller
#
# Usage:
#   ./scripts/build_desktop.sh          # build + run locally
#   ./scripts/build_desktop.sh package  # build + produce .app/.dmg

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

step() { printf '\n\033[1;34m==> %s\033[0m\n' "$1"; }

# ── Step 1: Freeze Python app with PyInstaller ──────────────────────
step "Freezing with PyInstaller"
uv run pyinstaller lazy-take-notes.spec --noconfirm --clean

# Verify the binary exists
BINARY="dist/lazy-take-notes"
if [[ ! -x "$BINARY" ]]; then
    echo "ERROR: PyInstaller output not found at $BINARY" >&2
    exit 1
fi

# Quick smoke test: --help should exit 0
step "Smoke testing frozen binary"
"$BINARY" --help > /dev/null

# ── Step 2: Generate .icns if missing ────────────────────────────────
LOGO_PNG="assets/logo.png"
LOGO_ICNS="assets/logo.icns"
if [[ -f "$LOGO_PNG" && ! -f "$LOGO_ICNS" ]]; then
    step "Generating .icns from logo.png"
    ICONSET=$(mktemp -d)/icon.iconset
    mkdir -p "$ICONSET"
    for size in 16 32 128 256 512; do
        sips -z $size $size "$LOGO_PNG" --out "$ICONSET/icon_${size}x${size}.png" > /dev/null
        double=$((size * 2))
        sips -z $double $double "$LOGO_PNG" --out "$ICONSET/icon_${size}x${size}@2x.png" > /dev/null
    done
    iconutil -c icns "$ICONSET" -o "$LOGO_ICNS"
fi

# ── Step 3: Trolley ─────────────────────────────────────────────────
if ! command -v trolley &> /dev/null; then
    echo "ERROR: trolley CLI not found. Install with: brew install weedonandscott/tap/trolley" >&2
    exit 1
fi

if [[ "${1:-}" == "package" ]]; then
    step "Packaging with Trolley"
    trolley package --skip-failed-formats
else
    step "Running with Trolley (dev mode)"
    trolley run
fi
