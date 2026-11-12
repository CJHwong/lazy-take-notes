#!/usr/bin/env bash
# Build, sign, notarize, and package the macOS app.
#
#   scripts/release.sh          build, sign when possible, zip
#   scripts/release.sh --dmg    also produce a DMG
#
# Signing runs only when a "Developer ID Application" identity is present.
# Notarization runs only when LTN_NOTARY_PROFILE names a notarytool profile.
# The last line of output always states which of the two ran, so an unsigned
# artifact cannot pass for a release.
#
# Environment:
#   LTN_SIGN_IDENTITY    code signing identity (default: first Developer ID)
#   LTN_NOTARY_PROFILE   notarytool keychain profile (default: none)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

APP_NAME="Lazy Take Notes"
BUNDLE_ID="com.cjhwong.lazy-take-notes"
RELEASE_DIR="$REPO_ROOT/trolley/release"

case "$(uname -m)" in
    arm64) TARGET="aarch64-macos" ;;
    x86_64) TARGET="x86_64-macos" ;;
    *)
        printf 'ERROR: unsupported architecture: %s\n' "$(uname -m)" >&2
        exit 1
        ;;
esac
APP="$REPO_ROOT/trolley/build/$BUNDLE_ID/$TARGET/dist/$APP_NAME.app"

WITH_DMG=0
for argument in "$@"; do
    case "$argument" in
        --dmg) WITH_DMG=1 ;;
        -h | --help) sed -n '2,16p' "$0"; exit 0 ;;
        *)
            printf 'Unknown argument: %s\n' "$argument" >&2
            exit 2
            ;;
    esac
done

step() { printf '\n==> %s\n' "$1"; }
die() { printf 'ERROR: %s\n' "$1" >&2; exit 1; }

toml_value() {
    awk -v block="$2" -v key="$1" '
        /^\[.*\]$/ { in_block = ($0 == block); next }
        in_block && $1 == key { gsub(/"/, "", $3); print $3; exit }
    ' "$3"
}

detect_identity() {
    security find-identity -v -p codesigning 2>/dev/null \
        | sed -n 's/.*"\(Developer ID Application: [^"]*\)".*/\1/p' \
        | head -n 1
}

# --- Version -----------------------------------------------------------------
PROJECT_VERSION=$(toml_value version "[project]" pyproject.toml)
TROLLEY_VERSION=$(toml_value version "[app]" trolley.toml)
[ -n "$PROJECT_VERSION" ] || die "could not read the version from pyproject.toml"
[ -n "$TROLLEY_VERSION" ] || die "could not read the version from trolley.toml"
[ "$PROJECT_VERSION" = "$TROLLEY_VERSION" ] \
    || die "pyproject.toml says $PROJECT_VERSION but trolley.toml says $TROLLEY_VERSION"

# --- Build -------------------------------------------------------------------
# The build script packages the app and inserts the microphone usage
# description. macOS denies microphone access without it.
"$REPO_ROOT/scripts/trolley/build.sh"
[ -d "$APP" ] || die "trolley finished but $APP is missing"

# --- Sign --------------------------------------------------------------------
IDENTITY="${LTN_SIGN_IDENTITY:-$(detect_identity)}"
if [ -n "${IDENTITY:-}" ]; then
    step "Signing with $IDENTITY"
    # Inside out. The launcher script has the executable bit, so codesign treats
    # it as a subcomponent of the runtime and refuses while it is unsigned. Its
    # signature lives in extended attributes, so ship the zip this script makes
    # (ditto preserves them) rather than re-archiving the app with zip.
    codesign --force --timestamp --sign "$IDENTITY" "$APP/Contents/MacOS/lazy-take-notes_core"
    codesign --force --options runtime --timestamp --identifier "$BUNDLE_ID" \
        --sign "$IDENTITY" "$APP/Contents/MacOS/lazy-take-notes_runtime"
    codesign --force --options runtime --timestamp --identifier "$BUNDLE_ID" \
        --sign "$IDENTITY" "$APP"
    codesign --verify --strict "$APP"
else
    printf 'No Developer ID Application identity found. The bundle stays unsigned.\n' >&2
fi

# --- Package -----------------------------------------------------------------
step "Packaging"
mkdir -p "$RELEASE_DIR"
ZIP="$RELEASE_DIR/LazyTakeNotes-$PROJECT_VERSION.zip"
ditto -c -k --keepParent "$APP" "$ZIP"
[ -f "$ZIP" ] || die "the zip archive was not created"

# --- Notarize ----------------------------------------------------------------
PROFILE="${LTN_NOTARY_PROFILE:-}"
if [ -n "$PROFILE" ] && [ -n "${IDENTITY:-}" ]; then
    step "Notarizing"
    xcrun notarytool submit "$ZIP" --keychain-profile "$PROFILE" --wait
    xcrun stapler staple "$APP"
    spctl --assess --verbose=2 --type exec "$APP"
    # Repackage so the archive carries the stapled ticket.
    ditto -c -k --keepParent "$APP" "$ZIP"
elif [ -n "$PROFILE" ]; then
    printf 'Skipping notarization: the app is unsigned.\n' >&2
fi

# --- DMG ---------------------------------------------------------------------
if [ "$WITH_DMG" = 1 ]; then
    step "Building the DMG"
    DMG="$RELEASE_DIR/LazyTakeNotes-$PROJECT_VERSION.dmg"
    hdiutil create -volname "$APP_NAME" -srcfolder "$APP" -ov -format UDZO "$DMG"
    [ -f "$DMG" ] || die "the DMG was not created"
    if [ -n "${IDENTITY:-}" ]; then
        codesign --force --timestamp --sign "$IDENTITY" "$DMG"
    fi
fi

# --- Report ------------------------------------------------------------------
step "Done"
printf 'Version: %s\n' "$PROJECT_VERSION"
printf 'App:     %s\n' "$APP"
printf 'Archive: %s\n' "$ZIP"
if [ "$WITH_DMG" = 1 ]; then
    printf 'Disk image: %s\n' "$DMG"
fi

if [ -n "${IDENTITY:-}" ] && [ -n "$PROFILE" ]; then
    printf 'Status:  signed and notarized\n'
elif [ -n "${IDENTITY:-}" ]; then
    printf 'Status:  signed, not notarized. Gatekeeper still blocks it on other machines.\n'
else
    printf 'Status:  UNSIGNED. Gatekeeper blocks this on other machines.\n'
fi
