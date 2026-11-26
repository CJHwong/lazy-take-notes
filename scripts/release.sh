#!/usr/bin/env bash
# Build, sign, notarize, and package the macOS app (the native window that
# rust/scripts/package_app.sh builds).
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
RELEASE_DIR="$REPO_ROOT/rust/target/release-app"
APP="$REPO_ROOT/rust/target/app/$APP_NAME.app"
ENTITLEMENTS="$REPO_ROOT/rust/scripts/app.entitlements"

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
RUST_VERSION=$(toml_value version "[package]" rust/Cargo.toml)
[ -n "$PROJECT_VERSION" ] || die "could not read the version from pyproject.toml"
[ -n "$RUST_VERSION" ] || die "could not read the version from rust/Cargo.toml"
[ "$PROJECT_VERSION" = "$RUST_VERSION" ] \
    || die "pyproject.toml says $PROJECT_VERSION but rust/Cargo.toml says $RUST_VERSION"

# --- Build -------------------------------------------------------------------
"$REPO_ROOT/rust/scripts/package_app.sh"
[ -d "$APP" ] || die "the build finished but $APP is missing"
# macOS denies the microphone and the system audio, without a prompt, to an
# app that lacks these strings.
for key in NSMicrophoneUsageDescription NSAudioCaptureUsageDescription; do
    plutil -extract "$key" raw "$APP/Contents/Info.plist" >/dev/null 2>&1 \
        || die "$APP has no $key in its Info.plist"
done

# --- Sign --------------------------------------------------------------------
IDENTITY="${LTN_SIGN_IDENTITY:-$(detect_identity)}"
if [ -n "${IDENTITY:-}" ]; then
    step "Signing with $IDENTITY"
    # Inside out: the bundle signature covers the helper, so the helper is
    # signed first. The hardened runtime needs the audio-input entitlement, or
    # macOS blocks the microphone.
    codesign --force --options runtime --timestamp --entitlements "$ENTITLEMENTS" \
        --sign "$IDENTITY" "$APP/Contents/MacOS/coreaudio-tap"
    codesign --force --options runtime --timestamp --entitlements "$ENTITLEMENTS" \
        --identifier "$BUNDLE_ID" --sign "$IDENTITY" "$APP"
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
