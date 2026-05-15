#!/usr/bin/env bash
# Build a macOS .app bundle for lazy-take-notes.
# The app opens a new Terminal window and runs the TUI via uv.
# Output: build/LazyTakeNotes.app (can be moved to /Applications)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
APP_NAME="LazyTakeNotes"
APP_DIR="$REPO_ROOT/build/${APP_NAME}.app"
CONTENTS="$APP_DIR/Contents"
VERSION="$(python3 - <<PY
import pathlib
import sys
import tomllib

pyproject = pathlib.Path("$REPO_ROOT") / "pyproject.toml"
try:
    with pyproject.open("rb") as f:
        data = tomllib.load(f)
    version = data["project"]["version"]
except (FileNotFoundError, KeyError) as exc:
    sys.stderr.write(f"Error reading version from {pyproject}: {exc}\n")
    sys.exit(1)
else:
    print(version, end="")
PY
)"

echo "Building ${APP_NAME}.app v${VERSION}..."

# Clean previous build
rm -rf "$APP_DIR"
mkdir -p "$CONTENTS/MacOS" "$CONTENTS/Resources"

# --- Launcher script ---
cat > "$CONTENTS/MacOS/launcher" << LAUNCHER
#!/bin/bash
# Initialize a reasonable PATH (important for Finder launches)
if [ -x /usr/libexec/path_helper ]; then
    eval "\$(/usr/libexec/path_helper -s)"
fi

# Resolve the uv binary (may not be on PATH when launched from Finder)
if command -v uv &>/dev/null; then
    UV="uv"
elif [ -f "\$HOME/.local/bin/uv" ]; then
    UV="\$HOME/.local/bin/uv"
elif [ -f "\$HOME/.local/share/mise/shims/uv" ]; then
    UV="\$HOME/.local/share/mise/shims/uv"
elif [ -f "\$HOME/.cargo/bin/uv" ]; then
    UV="\$HOME/.cargo/bin/uv"
elif [ -f "/opt/homebrew/bin/uv" ]; then
    UV="/opt/homebrew/bin/uv"
elif [ -f "/usr/local/bin/uv" ]; then
    UV="/usr/local/bin/uv"
else
    osascript -e 'display dialog "uv not found. Install it first:\n\ncurl -LsSf https://astral.sh/uv/install.sh | sh" with title "Lazy Take Notes" buttons {"OK"} default button "OK" with icon stop'
    exit 1
fi

# Open a new Terminal window, use uv tool run from git (no local repo required)
osascript <<EOF
tell application "Terminal"
    do script "mkdir -p ~/Documents/LazyTakeNotes && cd ~/Documents/LazyTakeNotes && \"\$UV\" tool run --from git+https://github.com/CJHwong/lazy-meeting-note.git lazy-take-notes"
    activate
end tell
EOF
LAUNCHER
chmod +x "$CONTENTS/MacOS/launcher"

# --- Info.plist ---
cat > "$CONTENTS/Info.plist" << PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>launcher</string>
    <key>CFBundleName</key>
    <string>Lazy Take Notes</string>
    <key>CFBundleDisplayName</key>
    <string>Lazy Take Notes</string>
    <key>CFBundleIdentifier</key>
    <string>com.cjhwong.lazy-take-notes</string>
    <key>CFBundleVersion</key>
    <string>${VERSION}</string>
    <key>CFBundleShortVersionString</key>
    <string>${VERSION}</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>LSMinimumSystemVersion</key>
    <string>13.0</string>
    <key>NSHighResolutionCapable</key>
    <true/>
</dict>
</plist>
PLIST

echo "Built: $APP_DIR"
echo ""
echo "To install:"
echo "  cp -r $APP_DIR /Applications/"
echo ""
echo "Then launch via Spotlight (⌘ Space → 'Lazy Take Notes') or Finder."
