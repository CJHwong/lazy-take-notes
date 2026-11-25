#!/usr/bin/env bash
# Build the macOS .app: the native window (the `gui` feature), with
# coreaudio-tap beside the executable, an icon, and the Info.plist keys macOS
# needs before it lets the app record.
#
#   rust/scripts/package_app.sh          -> rust/target/app/Lazy Take Notes.app
#   rust/scripts/package_app.sh --run    build, then open the app
#
# The helper is the one the Python package ships. If it is missing, this runs
# scripts/build_native.sh, which needs the Xcode Command Line Tools.

set -euo pipefail

RUST_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(cd "$RUST_DIR/.." && pwd)"
TAP="$REPO_ROOT/src/lazy_take_notes/_native/bin/coreaudio-tap"
ICON_PNG="$REPO_ROOT/assets/icon.png"
APP_NAME="Lazy Take Notes"
BUNDLE_ID="com.cjhwong.lazy-take-notes"
EXECUTABLE="lazy-take-notes"
APP="$RUST_DIR/target/app/$APP_NAME.app"
MIC_USAGE="Lazy Take Notes records your microphone to transcribe speech."
AUDIO_USAGE="Lazy Take Notes records the sound other apps play, such as the other side of a call, to transcribe it."

RUN=0
for argument in "$@"; do
    case "$argument" in
        --run) RUN=1 ;;
        -h | --help)
            sed -n '2,10p' "$0"
            exit 0
            ;;
        *)
            printf 'ERROR: unknown argument: %s\n' "$argument" >&2
            exit 2
            ;;
    esac
done

if [ "$(uname -s)" != "Darwin" ] || [ "$(uname -m)" != "arm64" ]; then
    echo "ERROR: the .app builds on Apple silicon Macs only" >&2
    exit 1
fi

if [ ! -x "$TAP" ]; then
    bash "$REPO_ROOT/scripts/build_native.sh"
fi

VERSION=$(sed -n 's/^version = "\(.*\)"$/\1/p' "$RUST_DIR/Cargo.toml" | head -1)
[ -n "$VERSION" ] || { echo "ERROR: no version in Cargo.toml" >&2; exit 1; }

# rustup picks the toolchain from the working folder: rust-toolchain.toml pins it.
(cd "$RUST_DIR" && cargo build --release --features gui --bin lazy-take-notes-gui)

# A new bundle each time, so no file from an older build stays behind. The
# bundle is this script's own output under target/.
if [ -d "$APP" ]; then
    find "$APP" -delete
fi
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp "$RUST_DIR/target/release/lazy-take-notes-gui" "$APP/Contents/MacOS/$EXECUTABLE"
cp "$TAP" "$APP/Contents/MacOS/"

WORK="$(mktemp -d)"
ICONSET="$WORK/AppIcon.iconset"
mkdir -p "$ICONSET"
for size in 16 32 128 256 512; do
    sips -z "$size" "$size" "$ICON_PNG" --out "$ICONSET/icon_${size}x${size}.png" >/dev/null
    double=$((size * 2))
    sips -z "$double" "$double" "$ICON_PNG" --out "$ICONSET/icon_${size}x${size}@2x.png" >/dev/null
done
iconutil -c icns "$ICONSET" -o "$APP/Contents/Resources/AppIcon.icns"
find "$WORK" -delete

cat >"$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key><string>$APP_NAME</string>
    <key>CFBundleDisplayName</key><string>$APP_NAME</string>
    <key>CFBundleIdentifier</key><string>$BUNDLE_ID</string>
    <key>CFBundleExecutable</key><string>$EXECUTABLE</string>
    <key>CFBundleIconFile</key><string>AppIcon</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>CFBundleShortVersionString</key><string>$VERSION</string>
    <key>CFBundleVersion</key><string>$VERSION</string>
    <key>LSMinimumSystemVersion</key><string>15.0</string>
    <key>LSApplicationCategoryType</key><string>public.app-category.productivity</string>
    <key>NSHighResolutionCapable</key><true/>
    <key>NSMicrophoneUsageDescription</key><string>$MIC_USAGE</string>
    <key>NSAudioCaptureUsageDescription</key><string>$AUDIO_USAGE</string>
</dict>
</plist>
PLIST
plutil -lint "$APP/Contents/Info.plist" >/dev/null

# An ad-hoc signature, so macOS keeps the microphone permission between
# builds of the same bundle id. scripts/release.sh signs a release properly.
codesign --force --sign - "$APP/Contents/MacOS/coreaudio-tap"
codesign --force --sign - --identifier "$BUNDLE_ID" "$APP"
codesign --verify --strict "$APP"

echo "$APP"
if [ "$RUN" = 1 ]; then
    open "$APP"
fi
