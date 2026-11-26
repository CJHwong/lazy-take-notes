#!/bin/sh
# Print the Homebrew cask for one release of the .app.
#
#   homebrew_cask.sh VERSION SHA256 [URL]
#
# The release job writes it to Casks/lazy-take-notes.rb in CJHwong/homebrew-tap.
# URL defaults to the release asset; a local test passes a file:// URL.
set -eu

if [ $# -lt 2 ]; then
    echo "usage: homebrew_cask.sh VERSION SHA256 [URL]" >&2
    exit 2
fi
version=$1
sha256=$2
url=${3:-'https://github.com/CJHwong/lazy-take-notes/releases/download/v#{version}/Lazy-Take-Notes-arm64-macos.zip'}

cat <<EOF
cask "lazy-take-notes" do
  version "$version"
  sha256 "$sha256"

  url "$url"
  name "Lazy Take Notes"
  desc "Takes meeting notes from your mic and system audio"
  homepage "https://github.com/CJHwong/lazy-take-notes"

  depends_on arch: :arm64
  depends_on macos: :sequoia

  app "Lazy Take Notes.app"
  # The app's program runs the command line when it gets arguments.
  binary "#{appdir}/Lazy Take Notes.app/Contents/MacOS/lazy-take-notes"

  # The app has no Developer ID signature, so Gatekeeper refuses it while
  # it carries the download mark that Homebrew adds. Installing from this
  # tap is the trust decision.
  postflight_steps do
    run "/usr/bin/xattr",
        args:           ["-dr", "com.apple.quarantine", "{{appdir}}/Lazy Take Notes.app"],
        writable_paths: ["Lazy Take Notes.app"],
        writable_base:  :appdir
  end

  zap trash: [
    "~/Library/Application Support/lazy-take-notes",
    "~/Library/Caches/lazy-take-notes",
  ]
end
EOF
