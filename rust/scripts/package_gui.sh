#!/usr/bin/env bash
# Build the Linux window package: the binary, the .desktop file and the icon.
# setup.sh installs it from this tarball, so the inner folder and the three
# names are part of the contract.
#
#   rust/scripts/package_gui.sh -> rust/dist/lazy-take-notes-gui-<arch>-linux.tar.gz

set -euo pipefail

RUST_DIR="$(cd "$(dirname "$0")/.." && pwd)"

if [ "$(uname -s)" != "Linux" ]; then
    echo "ERROR: this builds the Linux window package only" >&2
    exit 1
fi

case "$(uname -m)" in
    x86_64) ARCH=x86_64 ;;
    aarch64 | arm64) ARCH=arm64 ;;
    *)
        echo "ERROR: unsupported machine: $(uname -m)" >&2
        exit 1
        ;;
esac

NAME="lazy-take-notes-gui-$ARCH-linux"
STAGE="$RUST_DIR/dist/$NAME"

# whisper.cpp builds with -march=native by default. A release built for the
# runner's CPU can crash on an older one, so take the baseline.
export GGML_NATIVE=OFF
if [ "$ARCH" = x86_64 ]; then
    export GGML_AVX=ON GGML_AVX2=ON GGML_FMA=ON GGML_F16C=ON
fi

(cd "$RUST_DIR" && cargo build --release --locked --features gui --bin lazy-take-notes-gui)

mkdir -p "$STAGE"
cp "$RUST_DIR/target/release/lazy-take-notes-gui" "$STAGE/"
cp "$RUST_DIR/../assets/icon.png" "$STAGE/lazy-take-notes.png"
cat >"$STAGE/lazy-take-notes.desktop" <<'DESKTOP'
[Desktop Entry]
Type=Application
Name=Lazy Take Notes
Comment=Takes meeting notes from your mic and system audio
Exec=lazy-take-notes-gui
Icon=lazy-take-notes
Terminal=false
Categories=Office;AudioVideo;
DESKTOP

COPYFILE_DISABLE=1 tar -czf "$RUST_DIR/dist/$NAME.tar.gz" -C "$RUST_DIR/dist" "$NAME"
if command -v shasum >/dev/null 2>&1; then
    (cd "$RUST_DIR/dist" && shasum -a 256 "$NAME.tar.gz" >"$NAME.tar.gz.sha256")
else
    (cd "$RUST_DIR/dist" && sha256sum "$NAME.tar.gz" >"$NAME.tar.gz.sha256")
fi
echo "$RUST_DIR/dist/$NAME.tar.gz"
