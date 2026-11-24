# Build the desktop app

This guide builds a macOS `.app` that runs lazy-take-notes in its own window. It uses [Trolley](https://github.com/weedonandscott/trolley) and [libghostty](https://github.com/ghostty-org/ghostty).

The machine that runs the app needs `uv`, `ffmpeg`, and Ollama or an OpenAI-compatible API.

## Build and open it

```bash
brew install weedonandscott/tap/trolley
scripts/trolley/build.sh --run     # build, then open the app
scripts/trolley/build.sh           # build only
```

The app is in `trolley/build/com.cjhwong.lazy-take-notes/<target>/dist/Lazy Take Notes.app`.

On its first start, the app installs lazy-take-notes with `uv tool install`, then uses that install on every later start. It does not update by itself. To update it:

```bash
uv tool install --force --from git+https://github.com/CJHwong/lazy-take-notes.git lazy-take-notes
```

## Test a branch

The app installs from `main`. To test unmerged code, point it at your checkout when you build:

```bash
LTN_DESKTOP_SOURCE=/path/to/lazy-take-notes scripts/trolley/build.sh --run
```

## Sign it for other machines

The built app is ad-hoc signed. To give it to someone else, it needs a Developer ID signature and notarization, or macOS on their machine blocks a downloaded copy:

```bash
scripts/release.sh                          # build, sign if possible, zip
scripts/release.sh --dmg                    # also make a DMG
LTN_NOTARY_PROFILE=ltn scripts/release.sh   # also notarize
```

The files land in `trolley/release/`.

- Signing runs when the keychain holds a `Developer ID Application` identity, or when `LTN_SIGN_IDENTITY` names one.
- Notarization runs when `LTN_NOTARY_PROFILE` names a `notarytool` keychain profile.
- Without them, the script still builds the zip. Its last line says `UNSIGNED`.

## What the build sets

`trolley.toml` sets these for the app:

| Setting | Value | Why |
|---|---|---|
| `LTN_OUTPUT_DIR` | `~/Documents/LazyTakeNotes` | Sessions from the app and from a terminal land in one folder |
| `LTN_KIOSK` | `1` | The app returns to the mode picker after a session, instead of closing |
| `TEXTUAL_DRIVER` | the Ghostty driver | The Python version only: it turns off a keyboard mode that broke the arrow keys in this window. The native build does not need it |
| `NSMicrophoneUsageDescription` | added to `Info.plist` by `build.sh` | macOS denies the mic to an app without it, and Trolley has no setting for it |

If the window has the wrong size after you move it to another display, drag one of its edges. [Desktop window and display scale](explain_desktop_display_scale.md) explains why.
