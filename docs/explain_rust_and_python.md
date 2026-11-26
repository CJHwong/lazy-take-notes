# Rust and Python versions

lazy-take-notes exists twice: the original Python app, and a native rewrite in Rust. They read the same config, templates, plugins and model cache, and they write the same session files. Either one opens the other's sessions.

## Which one runs

| Machine | App |
|---|---|
| Mac with Apple silicon | The native app |
| Intel Mac, Linux, Windows | The Python app |

The Python package decides. Every install runs its `take-note` command, which is a small launcher:

1. It looks for a native build for this machine in `src/lazy_take_notes/rust_release.py`. That file pins one release tag and the SHA-256 of each package.
2. On the first start, it downloads the package from the GitHub release into the model cache, under `bin/<tag>/`, and checks the SHA-256.
3. It replaces itself with the native app, so the terminal and the signals go straight to it.

If anything fails (no build for this machine, no network, a wrong checksum), the launcher runs the Python app and prints one line that says why. `LTN_PYTHON=1` always runs the Python app.

## What the native app is

One executable, plus `coreaudio-tap`, the small helper that captures system audio. Speech recognition runs inside it, so it needs no Python packages of its own. Python runs only when you use a plugin: the native app then starts a separate Python process, the plugin host, and talks to it.

## Why only Apple silicon for now

The native app captures system audio through Core Audio, which exists only on macOS. Intel Macs need their own build. Linux and Windows also need their own system-audio code. None of these has been built or tested yet, so those machines keep the Python app, which already supports them.

## How updates reach you

The `take-note` command from setup.sh runs the package from `main` through uvx, so it follows the pin on `main`.

An install with `uv tool install` is a fixed copy. It never updates. To update it:

```bash
uv tool install --force --from git+https://github.com/CJHwong/lazy-take-notes.git lazy-take-notes
```

The desktop app is a fixed build too. To update it, pull and build it again ([Build the desktop app](howto_desktop_app.md)).

## Where they differ

The native app copies the Python app's behavior, with a short list of deliberate differences: for example, keyboard control of the speaker bar, and a speaker pass when you pause. [rust/PLAN.md](../rust/PLAN.md) lists every one under "Known differences from the Python version".
