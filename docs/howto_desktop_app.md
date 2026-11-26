# Build the desktop app

This guide builds the macOS `.app`: lazy-take-notes in a native window. It builds on Apple silicon Macs only. Other machines use the `lazy-take-notes` command in a terminal.

The machine that runs the app needs `ffmpeg`, and Ollama or an OpenAI-compatible API.

## Build and open it

```bash
rust/scripts/package_app.sh --run   # build, then open the app
rust/scripts/package_app.sh         # build only
```

The app is in `rust/target/app/Lazy Take Notes.app`. The build needs the Rust toolchain that `rust/rust-toolchain.toml` pins, and the Xcode Command Line Tools for the system audio helper.

On its first recording, macOS asks for the microphone and for the system audio. The build signs the app ad hoc, so macOS keeps both answers between builds.

## Sign it for other machines

To give the app to someone else, it needs a Developer ID signature and notarization, or macOS on their machine blocks a downloaded copy:

```bash
scripts/release.sh                          # build, sign if possible, zip
scripts/release.sh --dmg                    # also make a DMG
LTN_NOTARY_PROFILE=ltn scripts/release.sh   # also notarize
```

The files land in `rust/target/release-app/`.

- Signing runs when the keychain holds a `Developer ID Application` identity, or when `LTN_SIGN_IDENTITY` names one. It turns on the hardened runtime with `rust/scripts/app.entitlements`, which allows the microphone.
- Notarization runs when `LTN_NOTARY_PROFILE` names a `notarytool` keychain profile.
- Without them, the script still builds the zip. Its last line says `UNSIGNED`.
- The version in `pyproject.toml` and in `rust/Cargo.toml` must match, or the script stops.

## What the app sets

| Setting | Value | Why |
|---|---|---|
| `LTN_OUTPUT_DIR` | `~/Documents/LazyTakeNotes`, unless already set | A Finder start runs in `/`, so the default `./output` cannot work. Sessions from the app and from a terminal land in one folder |
| `PATH` | the login shell's `PATH` | A Finder start has no login shell, so tools such as `ffmpeg` and `ollama` would be missing |
| `NSMicrophoneUsageDescription`, `NSAudioCaptureUsageDescription` | in `Info.plist` | macOS denies the microphone and the system audio, without a prompt, to an app without them |
