#!/usr/bin/env bash
# Build the macOS .app with the Info.plist keys macOS needs for audio.
#
# Trolley 0.6.1 rejects Info.plist entries in trolley.toml and reports
# "unknown field info_plist_path", so the built bundle is patched after
# packaging. Without NSMicrophoneUsageDescription, macOS denies microphone
# access to the bundle without a prompt.
#
# Usage:
#   scripts/trolley/build.sh          build the .app
#   scripts/trolley/build.sh --run    build, patch, then open the .app
#
# Extra arguments are passed to `trolley package`.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

APP_NAME="Lazy Take Notes"
BUNDLE_ID="com.cjhwong.lazy-take-notes"
MIC_USAGE="Lazy Take Notes records your microphone to transcribe speech."

case "$(uname -m)" in
    arm64) TARGET="aarch64-macos" ;;
    x86_64) TARGET="x86_64-macos" ;;
    *)
        printf 'ERROR: unsupported architecture: %s\n' "$(uname -m)" >&2
        exit 1
        ;;
esac
APP="$REPO_ROOT/trolley/build/$BUNDLE_ID/$TARGET/dist/$APP_NAME.app"
PLIST="$APP/Contents/Info.plist"

RUN=0
PACKAGE_ARGS=()
for argument in "$@"; do
    case "$argument" in
        --run) RUN=1 ;;
        -h | --help)
            sed -n '2,14p' "$0"
            exit 0
            ;;
        *) PACKAGE_ARGS+=("$argument") ;;
    esac
done

if [ ${#PACKAGE_ARGS[@]} -eq 0 ]; then
    PACKAGE_ARGS=(--formats mac-app)
fi

command -v trolley >/dev/null 2>&1 \
    || {
        printf 'ERROR: trolley is not installed. Run: brew install weedonandscott/tap/trolley\n' >&2
        exit 1
    }

printf '\n==> Building the app bundle\n'
trolley package "${PACKAGE_ARGS[@]}"
if [ ! -d "$APP" ]; then
    printf 'ERROR: trolley finished but %s is missing\n' "$APP" >&2
    exit 1
fi

printf '\n==> Adding the microphone usage description\n'
plutil -remove NSMicrophoneUsageDescription "$PLIST" >/dev/null 2>&1 || true
plutil -insert NSMicrophoneUsageDescription -string "$MIC_USAGE" "$PLIST"
plutil -extract NSMicrophoneUsageDescription raw "$PLIST" >/dev/null \
    || {
        printf 'ERROR: the Info.plist patch did not stick\n' >&2
        exit 1
    }

printf 'App: %s\n' "$APP"

if [ "$RUN" = 1 ]; then
    printf '\n==> Opening the app\n'
    open "$APP"
fi
