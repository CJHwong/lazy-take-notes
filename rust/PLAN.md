# Rust rewrite plan

The Rust version replaces the Python app once it is stable. Until then the Python version stays on PyPI and in `src/`.

## Decisions

- Location: `rust/` in this repo, on the `rust-rewrite` branch.
- Platforms at the swap: macOS. Linux and Windows system audio come later; Linux loopback needs a spike first.
- Plugins: none in the first version. Ollama and OpenAI-compatible are built in.
- Data: drop-in. Both versions read and write the same files.
- Look: ratatui's idiom, not Textual's. The key bindings and the text stay Python's; the drawing does not. See "Look and mouse" below.

## Look and mouse

The screens follow these rules, over the colors of the saved theme:

1. A box is a rounded border with its title in the top border. A focused box has a `$primary` border. Any other box has a quiet line: the background moved 30% toward the foreground (`widgets::quiet_line`), because most dark themes set `border-blurred` too close to the background.
2. A button is one row: bold text in its variant's color, or `▌ Label ▐` on that color while it has the focus.
3. An input is a 3-row rounded box. Where a label sat above it, the label is the box's title.
4. A select shows `value ▾` in a rounded box and opens a rounded list over the screen.
5. A list marks the selected row with `▶` and the highlight colors.
6. A scrollbar is one column on the box's right edge: a `│` track in the quiet line color and a `┃` thumb in `$primary`.
7. A dialog is a rounded box with its title in the border, over a lighter dim (40%) of the screen below.
8. A toast is a rounded box at the bottom right, its border colored by severity.

The mouse:

- The terminal reports mouse events while the app runs, and stops before an editor opens and at exit. To select text in Ghostty, hold Shift and drag.
- While drawing, each widget records its rectangle (`Ctx::hit`). The host sends a click or a wheel step to the top screen, with the widget under the pointer (`Screen::mouse`). A screen that does not handle the wheel gets up and down keys.
- A click focuses a box, an input or a select, and presses a button. In a picker, a click on a row highlights it and a second click picks it. The welcome menu opens a mode on one click. The wheel scrolls the box under the pointer.
- A click on a `[key] label` hint in a status bar or footer presses that key (`widgets::hint_targets`). Hints that name no single key (`[↑/↓]`) take no click.
- A click on a toast closes it. A click outside a dialog does nothing.

## Data contract

The two versions must work on the same data, in both directions:

- `config.yaml`: same path, same schema, `null` language overrides
- template YAML, built-in and user
- session folders: transcript, notes, history, context, `recording.wav` and `recording_N.wav`, `debug.log`, `.status.json`, `speakers.json` and `transcript_speakers.txt` (written by `diarize`)
- model cache: `assets/`, `downloads/`, `locks/`, `state.json`
- CLI subcommands and flags, key bindings

## Non-goals

- Python plugin compatibility
- New features, or behavior changes during the port. The look and the mouse are the one exception, asked for after the port.
- Removing the Python version

## Verification

- `scripts/gen_fixtures.py` records Python outputs into `tests/fixtures/`. `tests/parity.rs` asserts the Rust output is byte-equal. The mic boost is the exception. Python takes its block level from a float32 `np.dot`, and the summation order of that call depends on the BLAS numpy was built with (Accelerate on macOS, OpenBLAS on Linux), so no fixed order reproduces it. Rust sums in f64 and rounds once, which lands closest. Its gain and level must agree within 1e-6 per block, and its bleed state exactly. On the current fixtures, 5 of 6 cases are bit-exact and the sixth differs by at most 8e-11. `serde_json` needs its `float_roundtrip` feature here: without it, it misread 164 of the expected gains by 1 ulp. Rerun the script when the Python side changes.
- `scripts/llm_parity.py` drives both versions' LLM clients against one failing local server and compares the shown text and the request count.
- `scripts/whisper_parity.py` transcribes the start of a real recording in both versions the way the file worker does, with a template's hints, and compares every segment's text and times.
- `examples/cross_check.rs record DIR SECONDS` records the real mic and system audio into a session WAV, and prints health changes and the delivered audio against wall time. `tone` measures the test tone in the tap. `mic-levels DEVICE SECONDS` reads one input device. `list-devices SECONDS` prints the input list every 0.5 s. `record DIR SECONDS DEVICE` records from a chosen input; changing that device's nominal rate mid-run tests the mic's stop and reopen. Fault runs: `kill -STOP` and `kill -9` on the tap pid, and a small `hdiutil` disk image filled, freed, with the WAV moved and the folder removed.
- `scripts/tui_compare.sh "ARGS" KEY... snap` runs both TUIs in tmux at one size, sends the same keys, and diffs the screen text at each `snap`. Since the new look, the layouts differ: use it to compare the words on a screen, not their places. `scripts/gen_bindings.py` records every Python screen's merged key table into `tests/fixtures/bindings.json`; `tests/tui_bindings.rs` checks the Rust tables against it. `scripts/gen_themes.py` writes Textual's generated theme colors into `src/tui/themes.json`.
- `scripts/cross_check.py` compares both versions on copies of the real config, templates, sessions and model cache.
- Each fixed bug listed in AGENTS.md gets a named Rust test.
- Before a step is done: a real run (headless record of a call, transcribe of a saved file, Rust `view` of a session Python recorded, and the reverse).

## Checklist

1. [x] Domain types, ports, use cases (digest, compaction, label, quick actions, speaker attribution, template builder, validator, diff, transcription buffering and speech gate)
2. [x] Persistence and assets: config loader and writer, template loader and writer, session files, session reader and status, model catalog, cached store, download engine, file lock
3. [x] LLM clients (Ollama, OpenAI-compatible), whisper-rs transcriber, audio file loader
4. [x] Audio: cpal mic source, Core Audio tap source (existing Swift binary), mixed source with retry and health, mic boost, WAV writer with backlog, test tone
   - Not verified on hardware: a headset unplugged and plugged back in. A rate change on a live input (BlackHole) covers the same stop, retry and reopen path, and an aggregate device created while running shows up in the device list without any re-init.
5. [x] Session controller, headless record, preflight, shutdown signals, diarization (existing NeMo runtime)
   - CLI in this step: the `--config` and `--output-dir` options, `record --headless`, `transcribe --headless`, `status` and `diarize`. `ls`, `transcript` and `notes` are not in any step yet.
   - Real runs: `transcribe --headless` on a 95 s recording gives a byte-identical `transcript.txt` and the same stderr lines. A blocked preflight (dead Ollama port, unwritable folder) prints the same report. A live `record --headless` muted and unmuted on SIGUSR1, ran a periodic digest, and on SIGTERM ran the final digest and exited 0. A SIGINT to the whole process group (a terminal Ctrl-C) reaches only the main process: the whisper and tap children run in their own sessions, so the last sentence was transcribed before the final digest. `transcribe --headless` prints the same provider and missing-model warnings. Each version's `status` reads the other's sessions.
6. [x] TUI: shell, pickers, record, transcribe, view, preflight screen, settings, template builder
   - Done: pickers, record (with the preflight screen), transcribe, view, template builder, the kiosk shell (bare `lazy-take-notes`, `LTN_KIOSK=1`), settings (`config`, and Settings from the welcome picker).
   - Settings: `scripts/gen_settings_fixtures.py` records Python's form in 12 cases (no file, an empty file, a removed override, an emptied key, a custom model path, an unreadable file and more). The Rust form opens the same data and saves the same file, key order included, with the same toasts. In tmux, all five pages, paging, the Tab walk, an open select list, the discard dialog and Save match Python's screens cell for cell, colors included. The live timer in the preview and the scrollbar thumb's last eighth can differ.
   - Real runs: each screen next to Python's in tmux, the same keys sent to both. The builder ran against a stub OpenAI-compatible server: both sent the same requests, and the screens matched through generate, auto-fix, save, edit and the quit confirm.
   - The kiosk ran a recording, stopped it, and came back to the welcome picker with no whisper, `coreaudio-tap` or `caffeinate` child left and the WAV closed. A cancelled preflight also leaves no child. The kiosk routes match Python's screens: each mode from welcome and back, the builder from the template picker and back to it, View into a session and back, and a full Transcribe. SIGTERM during a kiosk recording exits in 2 s with the spoken words in `transcript.txt`.
   - A closed window (SIGHUP) during a recording: the audio flushes and the transcript is saved; failed draws are dropped.
   - Named tests cover the faults of `test_record_app_faults.py` that apply without a mouse: markup-like text, a failed speakers write, a rename onto an existing folder, a failed rename, a panic in a digest or quick action task, a raising message handler, the shutdown signals, Ctrl+C under a modal, and a thread posting while the loop does not run. A panic in a session task now comes back as a failed result (`Ctx::spawn_or`). Before, it left the digest or query flag set, and `d` answered "Digest already in progress" until the app closed. The Settings workers (model list, install, remove, Ollama list and pull, AI test) and the template builder call do the same, as Python wraps each of them in `except Exception`; a thread uses `Ctx::spawn_thread_or`. The audio probe, the folder count and the shell checks stay unguarded, as in Python. The rest of the fault rows have named tests too: a worker thread crash (logged with its stack; the session runs on), the status bar health text, a render that panics once, `close_session` called twice, a host that cannot install its handlers, and in a child process the exit watchdog after SIGTERM with a hung loop (exit 143 after the grace). The child test found that SIGTERM stayed caught and did nothing after a host closed, because signal-hook keeps its handler. A closed host now gives each signal its old action back through `flag::register_conditional_default`, as Python restores the handlers it replaced; a second host in the same process still catches SIGTERM.
   - Python's `drivers/terminal_output.py` (retry EBADF on fd 2) exists only because PortAudio calls `dup2()` onto fd 2 when it re-initializes. cpal never re-initializes and never touches fd 2, so the Rust TUI does not need it.
7. [x] Cross-version run: each version opens the other's sessions and config
   - `scripts/package.sh` builds a release and puts `coreaudio-tap` next to the executable in `dist/lazy-take-notes-<arch>-macos.tar.gz`. A release build looks for the helper only there.
   - Real runs from the extracted package outside the repo: preflight captured the test tone, and with the mic muted (SIGUSR1) a spoken sentence came through the tap into `transcript.txt` word for word. With the helper removed, preflight fails and names the missing path. The same recording also passed with the repo folder hidden by `sandbox-exec`, so the package needs nothing from the build tree.
   - In one home: Python Settings saved After lines 7, Rust Settings showed it and saved Wait seconds 45, and Python Settings showed both. Each version transcribed the same file headless with an identical `transcript.txt`, `status` printed the same lines in both, and each `view` opened the other's session.

## The switch

Existing installs keep their command: setup.sh's `take-note` wrapper runs `uvx --from git+https://github.com/CJHwong/lazy-take-notes.git lazy-take-notes` on every start, and that file is on users' disks. So the Python package's `lazy-take-notes` and `take-note` scripts are `launcher.py`. It runs the Rust build pinned in `rust_release.py`, else the Python app.

- A tag `v*` runs `.github/workflows/release.yml`: macOS arm64 only, `rust/scripts/package.sh`, a start check, and a GitHub Release with the tarball and its SHA-256.
- `uv run scripts/pin_rust_release.py <tag>` writes the tag and hashes into `rust_release.py`. The pin is always a commit after the release: `main` must never pin a release that does not exist.
- The launcher downloads into `<cache>/bin/<tag>/` (the model cache's root), checks the SHA-256, unpacks into a temp folder and renames it into place, under a file lock. It then `execv`s the binary, so signals and the terminal reach the Rust app directly.
- Any failure (no build for the machine, no network, a hash mismatch, a bad archive, an exec error) runs the Python app with one note on stderr. `LTN_PYTHON=1` forces the Python app.
- No data migration: both versions use the same config folder, cache, sessions and plugins.txt. The wrapper is not rewritten: it keeps fetching `main`, which is how users get updates.
- The plugin host stays a Python script outside the launcher, and `plugin_api` imports no launcher code.

### Platforms and Python removal (decided 2026-11-27, reviewed against the code 2026-11-27)

This replaces "Two tracks" (2026-11-24). The Rust app covers Intel Macs and Linux, terminal and window app both. Then the Python app goes. A small Python package stays, for the launcher that existing installs run and for the plugin host.

Paths in this section are relative to `rust/`, except paths that start with `../` (the repository root). A `file:line` was checked on 2026-11-27. If a line moved, search for the quoted code.

Decisions:

- Platforms: macOS arm64 (done), macOS x86_64, Linux x86_64, Linux arm64. No Windows.
- The window app ships on all four.
- Install: `setup.sh` downloads the release tarball for the machine and checks its SHA-256. Homebrew stays for the Mac window app.
- Plugin subcommands and `llm_providers` keep working, through the Python plugin host.

Open decisions for the owner. Each has a default. The plan uses the default until the owner says otherwise.

1. **Python-only plugin features stop at 0.6.0.** `transcription_providers`, `run_record`, and the `llm_client` and `transcriber` arguments of `run_transcribe` work only in the Python app (`../docs/ref_plugin_api.md:11,29`). The Rust app refuses an unknown `transcription_provider` with an error (`src/app/container.rs:126-134`). Default: accept the loss. `plugin_api` keeps the names, so a plugin that imports them still loads, and the docs say which parts work. The owner's `ltn-claude-code` uses only `plugin_api` and `llm_providers` (strings in its `.pyc` files).
2. **The launcher pin is behind.** `../src/lazy_take_notes/rust_release.py:6` pins `v0.2.0`, so every `uvx` install on Apple silicon runs 0.2.0, not 0.3.4. Default: pin the newest release now (Phase 0b), and after every release from now on (the release procedure, step 9).

Non-goals:

- Windows.
- Linux packages beyond a tarball and a `.desktop` file: no AppImage, Flatpak, deb or rpm.
- GPU transcription on Linux (Vulkan, CUDA).
- Any change to the plugin contract or its JSON-RPC protocol.
- A Developer ID signature or notarization.
- New features. Apple silicon behavior stays the same, except where a shared helper replaces a macOS call.
- A Python package version bump: `../pyproject.toml:7` stays `0.2.0`. Only `rust/Cargo.toml` carries the release version.

Limits of the proofs, stated so nobody reads more into them:

- Linux audio is proven in Docker with a PulseAudio null sink, and once on the owner's real desktop (Phase 3). No proof covers every distribution.
- Intel is proven under Rosetta on Apple silicon and by `--version` and a transcription on GitHub's Intel runner. Rosetta does not prove Metal on an Intel GPU; the Intel runner does.

#### Rules for every phase

1. One branch and one PR per phase, from the newest `origin/main`. Exception: Phases 1, 2 and 4 ship together as 0.4.0 and may share the branch `linux-terminal` and one PR.
2. Before the PR, run all of these with `rust/` as the working folder. Each must pass:
   ```bash
   cargo fmt --check
   cargo clippy --all-targets --locked -- -D warnings
   cargo clippy --features gui --all-targets --locked -- -D warnings
   cargo test --locked
   cargo test --features gui --locked
   uv run --script scripts/check_layers.py
   uv run --script scripts/check_design.py
   ```
   Then, in the repo root: `actionlint .github/workflows/*.yml`, and `prek run --all-files`.
3. After a dependency change, commit `rust/Cargo.lock` in the same commit. CI builds with `--locked` and fails on a stale lock.
4. A real run on the target platform is the proof. A green CI run is a gate, not the proof. Each phase lists its proof run.
5. Record each new behavior difference between platforms in "Known differences" below, numbered after the last one.
6. If a step fails twice in the same way, stop. Write down what the run showed in the PR, before a third try.
7. A bad release: do not move or delete its tag (`../docs/howto_develop_and_release.md`, "never move a tag that a pin uses"). Revert the pin commit on `main` to the last good tag. If the cask is bad, run `scripts/homebrew_cask.sh` for the last good version and push the cask to CJHwong/homebrew-tap by hand. Then fix, and release the next patch version.

#### Release procedure (each release runs all of it)

1. Bump `version` in `Cargo.toml` to `X.Y.Z`, run `cargo build` so `Cargo.lock` follows, and commit both as "chore: release X.Y.Z".
2. Push the commit to `main`. A manual workflow run needs the commit on GitHub.
3. Run the Release workflow by hand on `main` (Actions > Release > Run workflow, or `gh workflow run release.yml --ref main`). It is a dry run and publishes nothing. Download its artifacts with `gh run download <run id>` and check that each expected file is there (the phase lists them).
4. `git tag -a vX.Y.Z -m "lazy-take-notes vX.Y.Z" && git push origin vX.Y.Z`.
5. When the Release run ends, check that the release lists every expected asset and its `.sha256`.
6. Check the release log line `designated => identifier "com.cjhwong.lazy-take-notes" and certificate leaf = H"f334a7e3cad21d44855ae302ec8ffdf017e0bb6c"` for each `.app`.
7. Check `Casks/lazy-take-notes.rb` in CJHwong/homebrew-tap: it says `X.Y.Z`, and each `sha256` equals `shasum -a 256` of the released zip it names.
8. Run `brew update && brew upgrade --cask lazy-take-notes` on the owner's Mac, but only when no session records (`pgrep -fl __whisper-worker` prints nothing). Open the app: no permission prompt.
9. Pin: on a branch `pin-vX.Y.Z`, run `uv run scripts/pin_rust_release.py vX.Y.Z`, commit `../src/lazy_take_notes/rust_release.py`, and push the branch (uvx installs from GitHub). Check that `ASSETS` has one entry for each platform the release built. Then run, each printing `X.Y.Z`:
   - on the Mac: `uvx --from git+https://github.com/CJHwong/lazy-take-notes.git@pin-vX.Y.Z take-note --version`;
   - Intel, under Rosetta: the same with `--python cpython-3.12-macos-x86_64-none` added, so Python reports `x86_64` (a native uvx would pick the arm64 build);
   - Linux: the same in `docker run --rm ghcr.io/astral-sh/uv:bookworm`, for `--platform linux/amd64` and `linux/arm64`.

   Merge the branch: every old wrapper gets the release on its next start.

#### What each platform lacks today

| Area | Today | Where |
|---|---|---|
| GPU | `whisper-rs` always turns on `metal`, which builds only on macOS | `Cargo.toml:55` |
| System audio | `CoreAudioTapSource` only, built unconditionally | `src/app/container.rs:10` (`use`), `:137-148` (`build_audio_sources`, the call at `:146`), `:151-155` (`build_mixed_source`), `:163` (the test tone); `src/app/preflight.rs:16` (`use`), `:495-501` (`check_system_audio`), `:535` (the test tone) |
| System audio module | `pub mod coreaudio_tap;` has no `cfg`, and its tests build a `.app` path | `src/adapters/audio/mod.rs:4`, `src/adapters/audio/coreaudio_tap.rs:444`, `coreaudio_tap_tests.rs:125` |
| Debug helper path | a debug build falls back to the Python package's `coreaudio-tap` | `src/adapters/audio/coreaudio_tap.rs:33-36`, the hint at `src/app/preflight.rs:45` |
| Test tone | the tone is a cpal output played into the tap; `mixed.rs` takes it as `Option<TestTone>` and is platform-neutral | `src/adapters/audio/microphone.rs:273`, `src/adapters/audio/mixed.rs:27,132` |
| Keep awake | runs `caffeinate` | `src/app/keep_awake.rs:13` |
| Open a file, window app | runs `open -W -t` | `src/gui/template_picker.rs:291`, `src/gui/settings/mod.rs:541` |
| Open a file, terminal app | already handles Linux (`xdg-open`) | `src/tui/screens/template_picker.rs:43-48` |
| Clipboard, open a folder (terminal) | already handles Linux: `wl-copy`, `xclip`, `xdg-open` | `src/tui/system.rs` |
| Clipboard, show a folder (window) | gpui calls with Linux versions; no change | `cx.write_to_clipboard`, `cx.reveal_path` in `src/gui/session.rs:611`, `src/gui/view.rs:164` |
| Shortcuts (window) | every binding uses `cmd-`, which gpui maps to Super on Linux; only `secondary-` maps to Ctrl there | `src/gui/mod.rs:133-152`; gpui `keystroke.rs:158-176` |
| Window chrome | traffic lights, transparent title bar, Dock icon through cocoa and objc | `src/gui/mod.rs:9,131,183-186`, `src/gui/dock_icon.rs`, `src/gui/window_move.rs` (macOS moves the window by its frame; Linux keeps gpui's `start_window_move` in `widgets.rs`), `src/gui/shell.rs:51-54`, `src/gui/design.rs:54-56` |
| Accent color | runs the macOS `defaults read -g AppleAccentColor` on each theme build | `src/gui/design.rs:287-295` |
| Monospace font | `MONO = "Menlo"`, which Linux does not have | `src/gui/design.rs:146` |
| Privacy hints and buttons | the hints name macOS System Settings and the `coreaudio-tap` helper; `open_privacy` does nothing off macOS, but its buttons still show | `src/app/preflight.rs:41-42,45`; buttons at `src/gui/settings/pages.rs:564,570`, `src/tui/screens/settings/pages.rs:295-296` |
| "Show in Finder" label | macOS word on Linux | `src/gui/mod.rs:91`, `src/gui/session.rs:928`, `src/gui/view.rs:282` |
| Wayland and X11 | both already on: gpui-kit turns on gpui's `wayland`, and its default features add `x11` | gpui-kit 0.11.0 `Cargo.toml:237-240,330-336` |
| Dark mode | `window.appearance()`, which gpui implements on Linux; no change, the Phase 3 spike confirms it | `src/gui/theme.rs:16-28` |
| Packaging | `package.sh` stops on anything but Darwin and names the tarball `<uname -m>-macos` | `scripts/package.sh:15,18` |
| `.app` packaging | arm64 only, reads `target/release/` | `scripts/package_app.sh:44,65` |
| Release | one build job; "Publish the release" is its step; the upload runs only on a dry run; the notes say Apple silicon; the cask job reads one arm64 SHA | `../.github/workflows/release.yml:85-89,93,105,111,127-129` |
| Cask | `depends_on arch: :arm64` and `depends_on macos: :sequoia` | `scripts/homebrew_cask.sh:28-29` |
| Python launcher pin | `PLATFORMS` maps one asset name to `('Darwin', 'arm64')`; the script skips an asset it does not know | `../scripts/pin_rust_release.py:19-21,30-39` |
| Built-in templates | the Rust binary embeds the Python package's YAML files | `src/adapters/templates.rs:20` |
| Window icon | the Dock icon reads `../assets/icon.png` at build time | `src/gui/dock_icon.rs:22` |
| Speaker labels runtime | already pinned for macOS x86_64, Linux x86_64 and Linux arm64 | `src/adapters/model_catalog.rs:123-157` |
| Old Python smoke test | CI runs the Python `smoke_test.py` | `../.github/workflows/ci.yml:59-60` |
| Unix calls (`flock`, `statvfs`, signals, process groups) | work on Linux as they are | no change |

#### Phase 0: release 0.3.3 (done)

Released 2026-11-27, then 0.3.4 the same day. The certificate-signed upgrade asked for no permission.

#### Phase 0b: pin the newest release (open decision 2)

1. Run the release procedure, step 9, for `v0.3.5`. Only the Mac line applies: 0.3.5 has only the arm64 macOS asset.
2. **Proof:** the Mac line prints `0.3.5`. Merge.

#### Phase 1: Linux terminal app (x86_64 and arm64), part of 0.4.0

Branch: `linux-terminal`.

1. **Metal only on macOS.** In `Cargo.toml:55`, change `whisper-rs = { version = "0.16.0", features = ["metal"] }` to `whisper-rs = "0.16.0"`, and add:
   ```toml
   [target.'cfg(target_os = "macos")'.dependencies]
   whisper-rs = { version = "0.16.0", features = ["metal"] }
   ```
   Cargo merges a target table's features with the base line. Check: `cargo tree -e features -i whisper-rs` on the Mac still lists `metal`.
2. **cpal's PulseAudio host on Linux.** Add to the same kind of table:
   ```toml
   [target.'cfg(target_os = "linux")'.dependencies]
   cpal = { version = "0.18.2", features = ["pulseaudio"] }
   ```
   - The feature pulls in the `pulseaudio` crate 0.3, a pure Rust client of the PulseAudio protocol. It needs no `libpulse`. cpal on Linux always builds ALSA too, so the build needs `libasound2-dev`.
   - `cpal::default_host()` picks PulseAudio only when a PulseAudio socket is found and the connection succeeds. The client finds the socket through `PULSE_SERVER`, `PULSE_RUNTIME_PATH` or `XDG_RUNTIME_DIR`; a desktop session sets `XDG_RUNTIME_DIR`. Otherwise it falls back to ALSA without an error (`cpal-0.18.2/src/platform/mod.rs:840-848`, `host/pulseaudio/mod.rs:163-165`). PipeWire desktops run `pipewire-pulse`, which provides the socket.
3. **System audio source.** Add `src/adapters/audio/pulse_monitor.rs`, declared in `src/adapters/audio/mod.rs` as `#[cfg(target_os = "linux")] pub mod pulse_monitor;`. It holds `PulseMonitorSource`, which implements `AudioSource` (`src/ports/audio_source.rs:46`):
   - `open()`: take `cpal::default_host()`. If `host.id()` is not `cpal::HostId::PulseAudio`, return the error below: the ALSA fallback has no monitors. Take `host.default_output_device()` and its `device.id()`. A `DeviceId` holds the host and the PulseAudio name; read the name with `DeviceId::id(&self) -> &str` (`cpal-0.18.2/src/lib.rs:288-291`). Do not parse its `Display` string. Find the input device whose `id()` is `<output name>.monitor`. Open it with the stream code `MicrophoneSource` uses: move that code into a shared `pub(crate) fn` in `microphone.rs`; do not copy it.
   - The error, for no PulseAudio host, no default output, or no monitor: `AudioError` with the text "No PulseAudio or PipeWire monitor for the default output". Preflight shows it as a failed system audio check.
   - `continuous_capture()` returns `true`: a monitor delivers silence all the time, so a gap means the stream is stuck.
   - Tests in `src/adapters/audio/pulse_monitor_tests.rs`: the pure function `monitor_id(output: &str) -> String`, and the pure pick from a list of ids (a match, no match, two sinks).
4. **One place picks the system source.** In `src/app/container.rs`, add `pub fn system_audio_source(ensure_permission: bool) -> Arc<dyn AudioSource>`:
   - macOS: `Arc::new(CoreAudioTapSource::new(ensure_permission))`;
   - Linux: `let _ = ensure_permission;` (no permission on Linux), then `Arc::new(PulseMonitorSource::new())`. Without the `let _`, clippy fails on an unused variable.

   Replace the calls at `container.rs:146`, `container.rs:155` and `preflight.rs:500`. Put `#[cfg(target_os = "macos")]` on `pub mod coreaudio_tap;` (`src/adapters/audio/mod.rs:4`), which also drops its tests on Linux, and on the `use` lines `container.rs:10` and `preflight.rs:16`.
5. **The test tone on Linux: decide by one run.** `mixed.rs` takes the tone as `Option<TestTone>` and needs no change; its tests run on Linux with fakes. Only the callers choose. A PulseAudio monitor records what the default output plays, so the tone may verify the monitor as it verifies the tap. In the Phase 1 Docker image (step 14), run the smoke once and read the system audio line of preflight in `/tmp/out/*/debug.log`:
   - If the system audio check passes with "tone heard": keep `Some(Box::new(play_test_tone))` at `container.rs:163` and the preflight tone at `preflight.rs:535` on both platforms. No known difference.
   - If not: on Linux pass `None` at `container.rs:163`, and skip the tone in preflight with `#[cfg(target_os = "macos")]`, so the Linux check passes on the first buffer. On Linux, `SYS LOST` then comes only from a failed read or no delivery for 3 s. Add a known difference.
   Write the outcome and the run's output in the PR.
6. **Hide monitors from the mic list.** On Linux, `list_input_devices()` (`microphone.rs:30-40`) would list "Monitor of ..." sources. It collects names, and the config and `match_device` use names. In its loop, skip a device whose `device.id()` name ends in `.monitor`, then keep the name as now. Test the pure filter on `(id, name)` pairs.
7. **Keep awake on Linux.** In `src/app/keep_awake.rs`:
   - Linux: run `systemd-inhibit --what=idle:sleep --who=lazy-take-notes --why=Recording --mode=block sleep infinity`, and kill it on drop, like `caffeinate`.
   - Not found: log one `debug!` line and go on.
   - Keep the macOS branch as it is.
8. **Open a file.** Add `src/app/open_path.rs` with `pub fn open_path(path: &Path, text_editor: bool, wait: bool) -> std::io::Result<std::process::Child>`:
   - macOS: `open`, with `-t` when `text_editor` and `-W` when `wait`.
   - Linux: `xdg-open` (the two flags do nothing).

   Replace the two window-app call sites, `gui/template_picker.rs:291` and `gui/settings/mod.rs:541`. Both run `open -W -t`, so both pass `true, true`. `xdg-open` returns at once, so on Linux the window app's Settings reloads the config before the edit ends (`gui/settings/mod.rs:532-533`). Record a known difference: on Linux, press Reload after editing the file. Add a Reload button only if the owner asks. Leave `tui/screens/template_picker.rs` and `tui/system.rs` as they are: they already handle Linux. Test the argument list per OS with a pure `fn command(path: &Path, text_editor: bool, wait: bool, os: &str) -> Vec<String>`.
9. **Platform text.** In `src/app/preflight.rs:41-42`, give `MIC_PERMISSION_FIX` and `SYSTEM_PERMISSION_FIX` a `#[cfg(target_os = "linux")]` version:
   - mic: "Check that PulseAudio or PipeWire runs and that the input is not muted (pavucontrol)."
   - system audio: "Check that PulseAudio or PipeWire runs: pactl info shows a server and a default sink."

   `SYSTEM_HELPER_FIX` (`preflight.rs:45`) names the macOS `coreaudio-tap` helper, and `check_system_source` sets it on every open failure (`:504`). On Linux it is the system audio text above.

   Hide the privacy buttons on Linux with `cfg!(target_os = "macos")`: the terminal app draws them at `src/tui/screens/settings/pages.rs:295-296`, and the window app at `src/gui/settings/pages.rs:564,570`. Add a test that the Linux hints name no System Settings and no `coreaudio-tap`.
10. **Package for Linux.** `scripts/package.sh`:
    - Remove the Darwin-only stop (`:18-21`).
    - Name the tarball `lazy-take-notes-<arch>-<os>.tar.gz` (`:15`), with `<arch>` from `uname -m` mapped as `x86_64 -> x86_64`, `aarch64|arm64 -> arm64`, and `<os>` `macos` or `linux`.
    - Build and copy `coreaudio-tap` only on macOS (`:23-25` and `:33`).
    - Write `<name>.tar.gz.sha256` next to the tarball, with `shasum -a 256` on macOS and `sha256sum` on Linux, in the same `<hash>  <file>` format.
    - Export `GGML_NATIVE=OFF` before the build, and on x86_64 also `GGML_AVX=ON GGML_AVX2=ON GGML_FMA=ON GGML_F16C=ON`. whisper.cpp builds with `-march=native` by default (`whisper-rs-sys-0.15.0/whisper.cpp/ggml/CMakeLists.txt:102-106`), and the whisper-rs build script passes `GGML_*` variables through. A release built for the runner's CPU can crash with an illegal instruction on a user's older CPU.
11. **CI.** In `../.github/workflows/ci.yml`, add a job `rust-linux`, named `Rust (Linux ${{ matrix.arch }})`, with `strategy.matrix.include` of `{arch: x86_64, runner: ubuntu-24.04}` and `{arch: arm64, runner: ubuntu-24.04-arm}`, `runs-on: ${{ matrix.runner }}`, and `defaults.run.working-directory: rust`. Steps:
    - `actions/checkout@v4` and `astral-sh/setup-uv@v5`.
    - The apt retry options from `ci.yml:24-25`, installing `cmake clang libclang-dev libasound2-dev ffmpeg`. Copy the mirror line (`ci.yml:23`) only if `/etc/apt/apt-mirrors.txt` exists on the runner: guard it with `[ -f /etc/apt/apt-mirrors.txt ] &&`, since the arm64 image may not have it. The file worker tests run ffmpeg (`ci.yml:75-77`).
    - `rustup toolchain install`
    - `cargo clippy --all-targets --locked -- -D warnings`
    - `cargo test --locked`
    - `uv run --script scripts/check_layers.py`

    `cargo fmt --check` and `check_design.py` do not depend on the platform; the macOS job keeps running them.

    In the same PR, delete the step "Check virtual audio capture" (`ci.yml:59-60`). It runs the Python `smoke_test.py`, which step 14 replaces.
12. **Release.** In `../.github/workflows/release.yml`:
    - Add a job `linux` with the same matrix, the same apt line, and `rustup toolchain install`. It runs `bash rust/scripts/package.sh`, extracts the tarball into `$RUNNER_TEMP`, and runs `--version` and `--help` from it. It does not copy the `coreaudio-tap` check (`release.yml:33`).
    - Every build job uploads its tarball, zip and `.sha256` files with `actions/upload-artifact@v4`, named after the asset. Remove the `if: github.event_name == 'workflow_dispatch'` from the upload step (`release.yml:86`), so the upload runs on a tag push too.
    - Move "Publish the release" (`release.yml:93`) out of job `macos-arm64` into a new job `publish`, with `needs: [macos-arm64, linux]`, `if: github.event_name == 'push'`, `runs-on: ubuntu-24.04`, and `env: GH_TOKEN: ${{ github.token }}` and `GH_REPO: ${{ github.repository }}` (it has no checkout, so gh needs the repository named). It runs `actions/download-artifact@v4` with `merge-multiple: true`, then one `gh release create` with every file.
    - The `.sha256` files now come from the build jobs, not from `publish`: `package.sh` writes the tarball's, and "Build the app" adds `(cd rust/dist && shasum -a 256 Lazy-Take-Notes-arm64-macos.zip > Lazy-Take-Notes-arm64-macos.zip.sha256)` after the zip.
    - The release notes become: "The Rust build for macOS (Apple silicon and Intel) and Linux (x86_64 and arm64): the terminal app, and the window app as Lazy Take Notes.app on macOS. Install: brew install --cask CJHwong/tap/lazy-take-notes on macOS, or setup.sh everywhere." Until Phase 2 lands, leave "Intel" out.
    - Change job `homebrew` to `needs: publish` (`release.yml:111`).
    - Expected assets after Phase 1: `lazy-take-notes-arm64-macos.tar.gz`, `Lazy-Take-Notes-arm64-macos.zip`, `lazy-take-notes-x86_64-linux.tar.gz`, `lazy-take-notes-arm64-linux.tar.gz`, each with `.sha256`.
13. **Python launcher pin.** Add to `PLATFORMS` in `../scripts/pin_rust_release.py:19-21`:
    - `'lazy-take-notes-x86_64-linux.tar.gz': ('Linux', 'x86_64')`
    - `'lazy-take-notes-arm64-linux.tar.gz': ('Linux', 'aarch64')`
    - `'lazy-take-notes-x86_64-macos.tar.gz': ('Darwin', 'x86_64')`

    The script skips a known asset that a release lacks (`:30-39`), so all three can go in now.
14. **Smoke test in Docker.** Replace `../scripts/linux-smoke/Dockerfile` and `smoke_test.py` with a `Dockerfile`, `run_smoke.sh`, `words_check.py`, `speech.wav` and `speech.txt`. Add `../.dockerignore` with `rust/target`, `rust/dist`, `.venv` and `**/__pycache__`: without it, `docker build` sends `rust/target` (tens of GB) as the build context.
    - `Dockerfile`: `FROM rust:1-bookworm`. Install `pulseaudio pulseaudio-utils libasound2-dev cmake clang libclang-dev ffmpeg curl python3`. Copy the repo to `/src`, run `rustup toolchain install` in `/src/rust`, then `cargo build --release --locked`, then `install -m755 /src/rust/target/release/lazy-take-notes /usr/local/bin/`. Download the speech model:
      ```bash
      mkdir -p /models && curl -fL -o /models/ggml-base.en.bin https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin
      ```
      Then `useradd -m testuser`, `COPY scripts/linux-smoke/run_smoke.sh /usr/local/bin/`, `USER testuser`, `CMD ["/usr/local/bin/run_smoke.sh"]`. PulseAudio refuses to run as root.
    - `run_smoke.sh` writes the test config `~/ltn.yaml` with mode 0600:
      ```yaml
      transcription:
        model: /models/ggml-base.en.bin
      llm_provider: ollama
      ollama:
        host: http://127.0.0.1:9
      ```
      Port 9 has no server, so the AI check fails and headless runs transcript-only. The keys are checked: `ollama.host` is `OllamaConfig.host` (`src/config.rs:144,182`), and `transcription.model` applies to `default_en`, because the built-in `transcription.models` has entries only for `zh` and `zh-min-nan` (`src/config.rs:13-17`).
    - `run_smoke.sh` then runs the steps below. The pure Rust PulseAudio client finds the server only through `PULSE_SERVER`, `PULSE_RUNTIME_PATH` or `XDG_RUNTIME_DIR` (`pulseaudio-0.3.2/src/lib.rs:34-58`), and a container user has none of them, so the script sets `XDG_RUNTIME_DIR` first. The only input in the container is the null sink's monitor, which sends zeros before `paplay`, and preflight blocks a mic that sends only digital silence (`src/app/preflight.rs:475-479`, `src/app/headless_run.rs:221-223`). So the run mutes the mic and records system audio only:
      ```bash
      export XDG_RUNTIME_DIR=$(mktemp -d); chmod 700 "$XDG_RUNTIME_DIR"
      pulseaudio --start --exit-idle-time=-1
      pactl load-module module-null-sink sink_name=test_loopback
      pactl set-default-sink test_loopback
      lazy-take-notes record --headless --mute-mic --config ~/ltn.yaml --template default_en --output-dir /tmp/out &
      sleep 8   # the speech model loads first
      paplay /src/scripts/linux-smoke/speech.wav
      sleep 5
      kill -TERM %1; wait %1; code=$?
      ```
    - SIGTERM in headless mode still runs the last chunk and the final digest (AGENTS.md "Shutdown signals"), and the digest fails here. The implementer runs the image once on the dev Mac (Docker Desktop) and writes the exit code into `run_smoke.sh` as `EXPECTED_EXIT=<n>`, with a comment that names the date. The script asserts that code, not a guessed one.
    - Pass when all hold:
      - `code` equals `EXPECTED_EXIT`;
      - `python3 /src/scripts/linux-smoke/words_check.py /src/scripts/linux-smoke/speech.txt /tmp/out/*/transcript.txt` exits 0. The script lowercases both files, skips `#` comment lines, strips punctuation, splits on whitespace, and passes when at least half of the unique words in `speech.txt` appear in the transcript. Phase 2 and Phase 4 call the same script;
      - `/tmp/out/*/debug.log` has no `ERROR` line that contains `system audio`.
    - `speech.wav`: one sentence of 5 to 10 s from a LibriVox recording (public domain). The implementer picks the recording, then writes its URL, the start and the length into a comment at the top of `speech.txt`, and converts it with `ffmpeg -i in.mp3 -ss <start> -t <length> -ar 16000 -ac 1 speech.wav`. The words go into `speech.txt` below the comment, and the words check skips comment lines.
    - In the same PR, update AGENTS.md "Linux Smoke Test" to the new image. Its "Audio Capture" text about `SoundCardLoopbackSource` is the Python app's and goes in Phase 5.

**Proof:**
- `docker build -f scripts/linux-smoke/Dockerfile -t ltn-linux-smoke . && docker run --rm ltn-linux-smoke` passes on x86_64 (`--platform linux/amd64`).
- The same passes on arm64 (`--platform linux/arm64`).
- The step 5 preflight run, with its outcome in the PR.
- The macOS arm64 build still records a 20 s call with system audio, and the test tone still verifies the tap.

#### Phase 2: Intel Mac terminal app, part of 0.4.0

Branch: `linux-terminal`, or `intel-mac` from `main` if Phase 1 has merged.

1. **Cross-build on the arm64 runner.** In `rust-toolchain.toml`, add `targets = ["x86_64-apple-darwin"]` under `[toolchain]`.
2. **`package.sh --target <triple>`.** Parse an optional `--target <triple>` argument:
   - With it: `export CMAKE_OSX_ARCHITECTURES=x86_64` when the triple starts with `x86_64-apple`, build with `cargo build --release --locked --target <triple>`, read the binary from `target/<triple>/release/`, and take `<arch>` from the triple (`x86_64-apple-darwin` gives `x86_64`, `aarch64-apple-darwin` gives `arm64`).
   - Without it: build and name as Phase 1 left it.
   - `coreaudio-tap` is already universal (`../scripts/build_native.sh:22-38`). Check with `lipo -archs`: it prints `x86_64 arm64`.
3. **Release.** In job `macos-arm64`, add a step that runs `bash rust/scripts/package.sh --target x86_64-apple-darwin`, and upload `lazy-take-notes-x86_64-macos.tar.gz` and its `.sha256`.
4. **CI smoke on real Intel.** Add a release job `macos-intel` on `runs-on: macos-15-intel`, with `needs: macos-arm64`. It:
   - checks out the repo, downloads the x86_64 tarball artifact, and runs `--version` and `--help`;
   - runs `brew install ffmpeg` (transcribe decodes with ffmpeg), and downloads `ggml-base.en.bin` with the Phase 1 curl line into `$RUNNER_TEMP`;
   - writes the Phase 1 config with that model path, mode 0600;
   - runs `lazy-take-notes transcribe --headless --template default_en --config <config> --output-dir "$RUNNER_TEMP/out" scripts/linux-smoke/speech.wav`, then `python3 scripts/linux-smoke/words_check.py scripts/linux-smoke/speech.txt "$RUNNER_TEMP"/out/*/transcript.txt`.

   This loads whisper and Metal on a real Intel GPU. Add `macos-intel` to the `needs:` of `publish`.
5. **Metal on Intel.** Keep `metal`. If step 4 fails inside whisper's Metal backend, change the Phase 1 table to `[target.'cfg(all(target_os = "macos", target_arch = "aarch64"))'.dependencies]`, so Intel builds without Metal, and add a known difference.
6. Expected assets now add `lazy-take-notes-x86_64-macos.tar.gz` and its `.sha256`. Change the release notes to name Intel.

**Proof:** on an Apple silicon Mac, extract the x86_64 tarball and run it under Rosetta with the Phase 1 config:
- `arch -x86_64 ./lazy-take-notes transcribe --headless --template default_en --config <config> --output-dir <temp> scripts/linux-smoke/speech.wav`: `words_check.py` passes;
- `arch -x86_64 ./lazy-take-notes record --headless --mute-mic --template default_en --config <config> --output-dir <temp>` for 20 s while `afplay scripts/linux-smoke/speech.wav` plays: `words_check.py` passes on `transcript.txt`. Use a temp config at mode 0600 and a temp output folder, and do not run it while the owner records or plays audio.
- The `macos-intel` job passes.

#### Phase 3: window app on Intel Mac and Linux, release 0.5.0

Branch: `gui-everywhere`.

1. **gpui on Linux, first a spike.** In Docker, `FROM rust:1-bookworm`, with the Phase 1 apt line plus `libxkbcommon-dev libxkbcommon-x11-dev libwayland-dev libx11-xcb-dev libvulkan-dev libfontconfig-dev libssl-dev mesa-vulkan-drivers xvfb fonts-dejavu-core`. Run `cargo build --release --locked --features gui --bin lazy-take-notes-gui` in `rust/`, then `xvfb-run -s "-screen 0 1280x800x24" ./target/release/lazy-take-notes-gui` for 10 s.
   - Install what the build asks for and add it to the apt line. That line, final, is the "window packages" line used in CI, release and the screenshot image.
   - The spike passes when the build ends and the process stays up for 10 s with no panic in its stderr.
   - If the spike fails: comment on the PR with the error, and stop the Linux part. Phase 3 then ships only steps 2 and 3 (Intel `.app` and the two-arch cask) as 0.5.0, and "Known differences" records "no window app on Linux".
2. **Intel `.app`.** `scripts/package_app.sh` takes `--target <triple>` like `package.sh`:
   - Remove the arch check at `:44`. Keep the Darwin check.
   - Build with `--target <triple>` when given (exporting `CMAKE_OSX_ARCHITECTURES` as `package.sh` does), and copy from `target/<triple>/release/` (`:65`).
   - Write `target/app/<arch>/Lazy Take Notes.app`. In `release.yml`, change the zip and check paths (`:73,76-77`) to that folder for each arch.
   - The release job zips each `.app` as `Lazy-Take-Notes-<arch>-macos.zip`. The arm64 name stays the same, since the 0.3 casks point at it. Sign both with the same certificate.
3. **Cask for both arches.** `scripts/homebrew_cask.sh VERSION SHA_ARM SHA_INTEL [URL_ARM URL_INTEL]`:
   - Drop `depends_on arch: :arm64`. Keep `depends_on macos: :sequoia` (the window app was built and tested only on Sequoia; a Mac on an older macOS uses `setup.sh`). Record that in the README install section.
   - Put `sha256` and `url` in `on_arm do ... end` and `on_intel do ... end` blocks.
   - The two URL arguments default to the release assets; a local test passes `file://` URLs.
   - In `release.yml:127-129`, job `homebrew` downloads both `.zip.sha256` files and passes both hashes.

   Before the release: `sh scripts/homebrew_cask.sh 0.5.0 <sha_arm> <sha_intel> file://<arm zip> file://<intel zip> > /tmp/lazy-take-notes.rb`, then `brew style --cask /tmp/lazy-take-notes.rb`, then `brew install --cask /tmp/lazy-take-notes.rb` on the owner's Mac, only when no session records. If Homebrew refuses a cask outside a tap, put the file in a local tap instead: `brew tap-new <user>/ltn-test`, copy it to that tap's `Casks/`, and install `<user>/ltn-test/lazy-take-notes`; `brew untap` it after.
4. **macOS-only window code behind `cfg`.**
   - `Cargo.toml:35-36`: move `cocoa = { version = "=0.25.0", optional = true }` and `objc = { version = "=0.2.7", optional = true }` into `[target.'cfg(target_os = "macos")'.dependencies]`. Keep `gui = ["dep:gpui-kit", "dep:cocoa", "dep:objc"]` (`Cargo.toml:19`): Cargo builds a target-specific optional dependency only on its target. Check: `cargo tree --features gui --target x86_64-unknown-linux-gnu -i cocoa` prints nothing.
   - `src/gui/mod.rs:9,131`: `#[cfg(target_os = "macos")]` on `mod dock_icon` and its call.
   - `src/gui/mod.rs:183-186`: on macOS keep `appears_transparent: true` and `traffic_light_position`. On Linux use `TitlebarOptions { title: Some("Lazy Take Notes".into()), ..Default::default() }`, and gpui draws its decorations.
   - `src/gui/shell.rs:51-54` and `src/gui/design.rs:54-56`: the traffic-light inset becomes a `design.rs` token that is zero on Linux, so `check_design.py` still passes.
   - `src/gui/design.rs:287-295`: `accent_setting()` gets `#[cfg(target_os = "macos")]`, and a Linux version returns `BLUE` (`design.rs:184`). Put `#[cfg(target_os = "macos")]` on `use std::process::Command;` (`design.rs:9`) too, or clippy fails on an unused import. Known difference: the accent is always blue on Linux.
   - `src/gui/design.rs:146`: `MONO` is `"Menlo"` on macOS and `"DejaVu Sans Mono"` on Linux.
5. **Shortcuts.** Do not use gpui's `secondary-`: `KeyBinding::new` parses with no OS, so `secondary-` makes it panic (`gpui-pre-0.3.14/src/keymap/binding.rs:35-47`, `src/platform/keystroke.rs:158-161`). Instead, in `src/gui/mod.rs`, add:
   ```rust
   /// The platform's command key: Cmd on macOS, Ctrl on Linux.
   macro_rules! key {
       ($k:literal) => {
           if cfg!(target_os = "macos") { concat!("cmd-", $k) } else { concat!("ctrl-", $k) }
       };
   }
   ```
   Write each binding at `src/gui/mod.rs:133-152` with it: `"cmd-q"` becomes `key!("q")`, and `"shift-cmd-r"` becomes `key!("shift-r")`. On macOS the strings stay the same. In `SHORTCUTS` (`src/gui/session.rs:52-70`), the key column shows `⌘` on macOS and `Ctrl+` on Linux: make it two `cfg` constants. Check: the menu bar on macOS shows the same shortcuts as before, and a test parses every binding string with `Keystroke::parse`.
6. **"Show in Finder".** Add `pub const SHOW_FOLDER: &str` to `src/gui/design.rs`, as two `cfg` constants: "Show in Finder" on macOS, "Show in Folder" elsewhere. Use it at `gui/mod.rs:91`, `gui/session.rs:928` and `gui/view.rs:282`. In `SHORTCUTS`, "Show the session folder in Finder" (`gui/session.rs:64`) says "in the file manager" on Linux, in the same two constants as step 5.
7. **Linux window package.** Add `scripts/package_gui_linux.sh`. It builds `cargo build --release --locked --features gui --bin lazy-take-notes-gui` and writes `lazy-take-notes-gui-<arch>-linux.tar.gz` with its `.sha256`. The tarball holds the binary, `lazy-take-notes.desktop`, and `lazy-take-notes.png`. Make the 512 px icon once on the Mac with `sips -Z 512 ../assets/icon.png --out ../assets/icon-512.png`, and commit it, so the Linux build needs no image tool. The `.desktop` file:
   ```ini
   [Desktop Entry]
   Type=Application
   Name=Lazy Take Notes
   Exec=lazy-take-notes-gui
   Icon=lazy-take-notes
   Categories=AudioVideo;Office;
   ```
   Add the script to the release job `linux` for both arches, with the window packages line.
8. **CI.** Job `rust-linux` installs the window packages line too, and also runs `cargo clippy --features gui --all-targets --locked -- -D warnings` and `cargo test --features gui --locked`.
9. **setup-smoke for the window tarball.** Run the Phase 4 setup-smoke again against a 0.5.0 dry-run artifact: the GUI link, the `.desktop` file and the icon land in place.

Expected assets for 0.5.0: the 0.4.0 set, plus `Lazy-Take-Notes-x86_64-macos.zip`, `lazy-take-notes-gui-x86_64-linux.tar.gz`, `lazy-take-notes-gui-arm64-linux.tar.gz`, each with `.sha256`.

**Proof:**
- Intel `.app`, under Rosetta, opened from Finder: the window, a 20 s recording with system audio, Settings, the viewer, the template builder.
- Linux under Xvfb in the spike image: `xvfb-run` the app, then `import -window root` (ImageMagick, in that image only) captures the sidebar, a saved session, Settings and preflight into `/tmp/shots/`. Attach the four PNGs to the PR. The owner compares them with the macOS screens: same layout, no clipped text.
- **A real Linux desktop** (Wayland and X11, a real GPU) before the 0.5.0 tag. The owner provides the machine and runs this checklist, and writes the result in the PR:
  1. `setup.sh` installs the window app, and it opens from the app menu.
  2. A 20 s recording with system audio writes a transcript.
  3. Settings saves to a temp config passed with `--config`.
  4. The viewer opens the session.
  5. Ctrl+N starts a session, and the theme follows the desktop's dark mode.
  6. On GNOME under Wayland (no server-side decorations), the window has a title bar that moves and closes it. If not, record it and give the window gpui's client-side decorations in a follow-up.

#### Phase 4: `setup.sh` installs the binary, part of 0.4.0

Branch: `linux-terminal`. The file is `../setup.sh`. Today its "Step 1: uv" (`setup.sh:77-91`) installs uv, and the steps below replace that flow.

1. Pick the asset from `uname -s` and `uname -m`:
   - `Darwin x86_64` gives `x86_64-macos`
   - `Darwin arm64` gives `arm64-macos`
   - `Linux x86_64` gives `x86_64-linux`
   - `Linux aarch64` gives `arm64-linux`
   - Anything else stops with the list of supported systems.
2. Read the newest release tag with `curl -fsSL https://api.github.com/repos/CJHwong/lazy-take-notes/releases/latest | sed -n 's/.*"tag_name": *"\([^"]*\)".*/\1/p'`. `LTN_VERSION=v0.4.0` overrides it. If the call fails or prints nothing (rate limit, no network), stop with "Cannot read the newest release. Set LTN_VERSION=vX.Y.Z and run again."
3. Download the tarball and its `.sha256` with `curl -fL` from `${LTN_RELEASE_URL:-https://github.com/CJHwong/lazy-take-notes/releases/download}/<tag>/`. The variable lets setup-smoke serve dry-run artifacts from a local folder. Compare with `shasum -a 256` (macOS) or `sha256sum` (Linux), and stop on a mismatch.
4. Extract into `~/.local/share/lazy-take-notes/<tag>/`, and link `~/.local/bin/lazy-take-notes` and `~/.local/bin/take-note` to the binary in it. When the release has a GUI tarball for the machine (Linux, 0.5.0 on; `curl -fsIL` of its `.sha256` URL succeeds), also download and check it, link `~/.local/bin/lazy-take-notes-gui`, copy the `.desktop` file to `~/.local/share/applications/`, and copy the icon to `~/.local/share/icons/hicolor/512x512/apps/lazy-take-notes.png`.
5. "Already installed" means `~/.local/share/lazy-take-notes/<tag>/` exists and both links point into it. Then print "lazy-take-notes <tag> is already installed" and change nothing.
6. If `~/.local/bin` is not on `PATH`, print the line to add for the user's shell. Do not edit shell files. Lines that the old `setup.sh` added to `.zshrc` or `.bashrc` (`setup.sh:220-238`) stay and do no harm.
7. On macOS with Homebrew, print `brew install --cask CJHwong/tap/lazy-take-notes` as the recommended way, and ask "Install the terminal app from the tarball instead? (y/N)", reading from `/dev/tty` as `setup.sh:145` does. With no tty, or no answer, use the tarball.
8. Old wrappers, run before step 4: the old `setup.sh` wrote `take-note` to `/usr/local/bin` when it was writable, else to `~/.local/bin` (`setup.sh:189-201`). Check `/usr/local/bin/take-note`, `$(brew --prefix 2>/dev/null)/bin/take-note` and `~/.local/bin/take-note`. An old wrapper is a regular file (not a symlink) that contains `lazy-take-notes.git`. The Rust binary also contains the text `uvx` (`src/adapters/plugins/host.rs:27`), so do not match on `uvx`. Remove each old wrapper (with `sudo` if the folder needs it, asking first on `/dev/tty`) and print one line that says so. A wrapper left in `/usr/local/bin` would come first on `PATH` and hide the new link.
9. Keep the AI provider steps as they are. Drop the uv install step. The plugin how-to says to install uv before adding a plugin.

**Proof:** rewrite `../scripts/setup-smoke/Dockerfile` and `run_tests.sh` for the tarball flow (today they test the uv and Homebrew flow). The image reuses the Phase 1 audio setup (`run_smoke.sh`, `speech.wav`, the model). Run it in Docker, Debian bookworm, x86_64 and arm64, against the dry-run artifacts of the release workflow: copy them into `/srv/<tag>/` in the image, serve `/srv` with `python3 -m http.server 8000 --directory /srv &`, and run with `LTN_VERSION=<tag> LTN_RELEASE_URL=http://127.0.0.1:8000`:
- install from a clean HOME;
- `take-note --version` prints the release;
- a second run prints "already installed" and changes nothing (compare `ls -l` of the links before and after);
- an old wrapper planted at `~/.local/bin/take-note` is replaced, with the one-line note;
- `take-note record --headless --mute-mic --template default_en` against the null sink, run as in `run_smoke.sh`, passes `words_check.py`.

On macOS, run `setup.sh` with `HOME` set to a temp folder and answer `y` to the tarball question: the same checks, without the recording.

#### Phase 5: remove the Python app, release 0.6.0

Start only after 0.5.0 has run on the real Linux desktop, and after `main` pins a release with all four terminal assets (release procedure, step 9). Old wrappers fetch `main` on every start, so the merge of this PR changes every old install at once.

Branch: `remove-python-app`.

**What must keep working.** Every install made by the old `setup.sh` has a `take-note` wrapper that runs `uvx --from git+https://github.com/CJHwong/lazy-take-notes.git lazy-take-notes` on each start ("The switch" above). That command must keep starting the Rust app. So the package keeps its name and its three scripts (`../pyproject.toml:48-50`):

| Script | After Phase 5 |
|---|---|
| `lazy-take-notes`, `take-note` | `launcher.py`: runs the pinned Rust build. With no build for the machine, or any failure, it prints one line naming `setup.sh` and exits 1. No Python fallback. |
| `lazy-take-notes-plugin-host` | `plugin_host.py`, with its imports repointed (step 1) |

The old wrapper also exports `LTN_PLUGIN_WRAPPER=1` and passes `--with <plugin>` for each line of `plugins.txt` (`setup.sh:252-261`). After Phase 5 the variable does nothing, and uvx still installs the plugins into the launcher's environment on each start. That costs start time and breaks nothing. Running the new `setup.sh` removes the old wrapper (Phase 4, step 8).

Steps:

1. **The kept package.** In `../src/lazy_take_notes/`:
   - Keep `launcher.py`, `rust_release.py`, `plugin_host.py`, `plugin_api.py`.
   - Add `_api_types.py` with the types `plugin_api` re-exports, copied from their layer files with the same names and fields:
     - `ChatMessage` (`l1_entities/chat_message.py`), `SessionArtifacts` (`l1_entities/session_files.py`), `TranscriptSegment` (`l1_entities/transcript.py`);
     - the ports `AudioSource`, `LLMClient`, `ChatResponse`, `ModelResolver`, `Transcriber` (`l2_use_cases/ports/`). `audio_source.py:7` and `transcriber.py:7` import numpy for annotations only: write `from __future__ import annotations` and import numpy under `if TYPE_CHECKING:`, so the kept package does not need numpy;
     - `OllamaProviderConfig`, `OpenAIProviderConfig` and `InfraConfig` (`l4_frameworks_and_drivers/config.py:92-116`) without the imports at `:9-10`. `build_app_config`, `load_theme` and `DEFAULT_THEME` are not copied;
     - `TranscriptionBackend` (`l4_frameworks_and_drivers/container.py:37`);
     - `run_record` and `run_transcribe` stubs that raise `RuntimeError("runs only inside the plugin host")`. The host replaces both at runtime (`plugin_host.py:126-169`).
   - `plugin_api.py` re-exports from `_api_types` with the same public names.
   - Keep the old import paths that plugins and the host use, as real modules (the host sets attributes on one at `plugin_host.py:166`):
     - `l4_frameworks_and_drivers/__init__.py`, empty;
     - `l4_frameworks_and_drivers/plugin_api.py`, which re-exports everything from `_api_types`;
     - `l4_frameworks_and_drivers/config.py`, which re-exports only `InfraConfig`, `OllamaProviderConfig` and `OpenAIProviderConfig` (the host imports it at `plugin_host.py:342-343`).
   - `plugin_host.py:27-28`: import `ChatMessage` and `SessionArtifacts` from `lazy_take_notes._api_types`.
   - `launcher.py`: add `def give_up(reason: str) -> NoReturn`, which prints `lazy-take-notes: no build for this machine (<reason>). Install with: curl -fsSL https://raw.githubusercontent.com/CJHwong/lazy-take-notes/main/setup.sh | bash` to stderr and exits 1. Replace the `LTN_PYTHON` check (`:126-127`) with `give_up("the Python app was removed in 0.6.0")`. Delete `run_python()` (`:143-148`). In `main()` (`:151-161`), replace both `run_python()` calls with `give_up(<the reason already printed there>)`. Keep `main` as the entry point (`pyproject.toml:48-49`). Update the docstring at `:3-7`.
   - `rust_release.py:3`: "TAG None means no release is pinned: the launcher exits with the setup.sh line."
2. **Move the templates into the Rust tree.** `src/adapters/templates.rs:20` embeds `../../../src/lazy_take_notes/templates/<name>.yaml`. Move `../src/lazy_take_notes/templates/` to `templates/` (in `rust/`), change the path in the `include_str!` to `../../templates/`, and run `cargo test --locked` before deleting anything else. Then run `grep -rn "lazy_take_notes/templates" ../docs ../README.md ../AGENTS.md` and fix each hit.
3. **Move `coreaudio-tap` into the Rust tree.**
   - Move `../native/` to `native/`, and `../scripts/build_native.sh` to `scripts/build_native.sh`. In the moved script, change `BINARY` (`build_native.sh:11`) to `target/native/coreaudio-tap` under `rust/`, and add `mkdir -p "$(dirname "$BINARY")"` before the `lipo -create` line (`:35`).
   - `scripts/package.sh:14` and `scripts/package_app.sh:20`: `TAP="$RUST_DIR/target/native/coreaudio-tap"`, built by the moved script when missing.
   - `src/adapters/audio/coreaudio_tap.rs:33-36`: the debug fallback becomes `env!("CARGO_MANIFEST_DIR")` joined with `target/native/coreaudio-tap`.
   - `src/app/preflight.rs:45`: the hint says `bash rust/scripts/build_native.sh`.
   - Delete `../src/lazy_take_notes/_native/`, and the `artifacts` entry for it in `../pyproject.toml:69-71`.
   - Check: `cargo run -- record` on the Mac, and the preflight system audio check passes.
4. **Tests.**
   - Replace `../tests/conftest.py` (it imports every layer, `:10-34`) with only what the two kept tests need.
   - `../tests/test_launcher.py`: delete the Python-fallback tests at `:94,115,158,169,183,191` (`:191` is `test_run_python_is_the_click_cli`); read each before deleting, and keep any part that tests the Rust path. Add tests for the exit-1 line, with and without `LTN_PYTHON=1`.
   - `../tests/test_plugin_host.py:20-22`: import `TranscriptSegment` and `ChatResponse` from `lazy_take_notes._api_types`. Keep the `l4_frameworks_and_drivers.plugin_api` import: it tests the old path.
   - Add `../tests/test_api_types.py`: `import lazy_take_notes.plugin_api` works with numpy not installed, and every name in its `__all__` resolves.
5. **Delete** (only after steps 2 and 3 pass `cargo test --locked`):
   - the rest of `l1_entities/`, `l2_use_cases/`, `l3_interface_adapters/`, `l4_frameworks_and_drivers/` (keeping the three modules from step 1), and `__main__.py`;
   - their tests under `../tests/`, and `../tests/scripts/test_release_sh.py` with `../scripts/release.sh`;
   - `../docs/explain_rust_and_python.md`.
6. **`../pyproject.toml`.**
   - Dependencies: keep `platformdirs` (`launcher.py`), `pydantic` (`InfraConfig`) and `click` (plugin commands). Drop every other runtime dependency, including `numpy`, `certifi` and `pydantic-settings`. Run `uv lock`.
   - The dev group: drop `textual-dev` and `import-linter`, and add `numpy`, so `ty` resolves the numpy import under `TYPE_CHECKING`. The runtime package still does not need it.
   - Delete the `[tool.importlinter]` table and all its contracts (from `:118`), and the `omit` entry for `__main__.py` (`:109`).
   - In `../.pre-commit-config.yaml`, delete the `import-linter` hook. `ty` and `ruff` keep running on the kept package.
7. **CI.** In `../.github/workflows/ci.yml`:
   - The `test` job runs `uv run pytest tests/ -v --cov=lazy_take_notes --cov-fail-under=90` on Python 3.11 to 3.13. The kept package is small; if coverage falls under 90, add tests rather than lower the bar.
   - Drop the portaudio, PulseAudio and null sink steps from the `quality` and `test` jobs (`ci.yml:21-30,46-55`). The Python jobs no longer touch audio.
8. **Docs.**
   - README: install is Homebrew on Mac or `setup.sh` everywhere, the Rust and Python split note goes, and the window app needs macOS Sequoia (Phase 3).
   - `../docs/ref_plugin_api.md`: drop the "Native build" column and the Python-only rows, and state which parts stopped at 0.6.0 (open decision 1).
   - `../docs/howto_plugins.md:100-103`: delete the Python-version text and the `uvx --with` example. State that plugins need uv and run in the plugin host.
   - `../docs/howto_develop_and_release.md`: point the release steps at this plan's release procedure, and drop `scripts/release.sh`, the Python app's commands, and the single-SHA cask.
   - AGENTS.md: delete the Python architecture, data flow, audio capture, testing and "Operational Commands" sections (`lint-imports`, `ty check src tests`). The "Rust Layers" table becomes the layout section. Keep the plugin contract. The root `CLAUDE.md` is `@AGENTS.md` and needs no change.
   - This file, "The switch": replace "else the Python app" and the bullet about failures with "Any failure prints one line naming setup.sh and exits 1."
9. **Release** 0.6.0 with the release procedure. Its pin (step 9) points every old wrapper at 0.6.0.

**Proof:**
- `uvx --from git+https://github.com/CJHwong/lazy-take-notes.git@remove-python-app take-note --version` prints the Rust version on macOS arm64, on macOS x86_64 under Rosetta, and on Linux in Docker.
- The same command with `LTN_PYTHON=1` prints the `setup.sh` line and exits 1.
- `uvx --from git+https://github.com/CJHwong/lazy-take-notes.git@remove-python-app lazy-take-notes-plugin-host` starts and answers a JSON-RPC request.
- The owner runs this one, because only the owner has the plugin source: with `ltn-claude-code` in `plugins.txt` and `LTN_PLUGIN_HOST_FROM` pointing at the branch, the plugin's subcommand runs, and a session that uses its LLM provider writes a digest.

#### Releases

| Version | Content | Gate before the tag |
|---|---|---|
| 0.3.3, 0.3.4 | Phase 0 | done |
| (pin only) | Phase 0b: pin v0.3.5 | the uvx run on the Mac |
| 0.4.0 | Phases 1, 2, 4: Linux and Intel terminal app, new `setup.sh`, pin for macOS arm64, macOS x86_64 and Linux | the Linux Docker smoke on both arches, the Rosetta runs, the `macos-intel` job, setup-smoke |
| 0.5.0 | Phase 3: window app on Intel and Linux, two-arch cask | the Rosetta `.app` run, the Xvfb screenshots, the real Linux desktop checklist |
| 0.6.0 | Phase 5: Python app removed | the uvx wrapper runs, the plugin host run, the owner's plugin run |

#### What the owner provides

1. A decision on the two open items at the top (or the defaults stand).
2. A real Linux desktop (a machine or a VM with a GPU) and the Phase 3 checklist run.
3. The `ltn-claude-code` plugin run in Phase 5. The copy under `../plugins/` holds only `.pyc` files.
4. The review of the Xvfb screenshots in Phase 3.

## GUI mode (decided 2026-11-25)

A native window built with GPUI Kit. It replaces the Trolley `.app`. The terminal app stays the default for the `take-note` command and does not change.

- Cargo feature `gui`, off by default. `gpui-kit` (pinned `=0.11.0`) is an optional dependency that only `--features gui` compiles. The release tarball the Python launcher runs is built without it.
- `gui/` is a front end beside `tui/`. It imports `app/` and below, never `tui/`, and `tui/` never imports `gui/`. `gpui` and `gpui_kit` appear only in `gui/`. Each screen is a view over the same UI-free runtime the TUI uses (`app/session/`, and runtimes for preflight, settings and the template builder).
- The `.app` executable is a second binary that needs the `gui` feature. With no arguments it opens the window. With arguments it runs the normal command line, so the whisper child (`current_exe()` plus a subcommand) works inside the bundle.
- A Finder start has no login shell: the app restores `PATH` the way `ltn-launch` does, and its Info.plist carries the microphone and system audio usage strings.
- Closing the window or Cmd-Q while recording takes the shutdown path: stop the audio, wait for the flush, then exit.
- The toolchain moves to Rust 1.98: GPUI Kit does not build on 1.93 or 1.96.
- A tokio worker reaches a GPUI view through the existing channels: the view awaits the receiver in a foreground task (tested in a spike).

Scope: every TUI screen. That is the menu, the pickers, preflight, record, transcribe, view, Settings, the template builder and the modals. When the `.app` passes its real runs, `scripts/trolley/` and the Trolley parts of `release.sh` and the desktop app guide go.

Non-goals:
- No Python change.
- No Intel, Linux or Windows GUI. Superseded on 2026-11-27: "Platforms and Python removal" ships the window app on Intel Macs and Linux.
- No TUI behavior change.
- No feature the TUI does not have.

## Plugins

Python plugins keep working: the Rust app runs them in a separate Python process, the plugin host, and talks to it in JSON. Nothing in a plugin changes.

**Launch.** The host is the Python package's `lazy-take-notes-plugin-host` script, started as `uvx --from <host> --with <each line of plugins.txt> lazy-take-notes-plugin-host`. `<host>` is `lazy-take-notes @ git+https://github.com/CJHwong/lazy-take-notes.git`; `LTN_PLUGIN_HOST_FROM` replaces it: a local checkout (built with `--no-cache`, since uv caches a folder's build by its pyproject.toml only) or another spec. The plugins pin the released URL, so another source goes to uv as an `--overrides` file too. uv supplies Python, so the user needs only uv. If the launch fails, it runs again with `--offline` to use uv's cache. With no plugins.txt lines, no host starts.

**Transport.** One JSON-RPC 2.0 object per line on the host's stdin and stdout, UTF-8. A message with `method` is a request (with `id`) or a notification (no `id`); a message with `result` or `error` answers the receiver's own request. Each side numbers its own requests. The host's stderr goes to the terminal (a plugin's `click.echo(err=True)` and tracebacks); the host moves Python's `sys.stdout` to stderr, so a plugin's print never breaks the stream.

**Messages.**

| From | Method | Params | Result |
|---|---|---|---|
| host | `hello` (notification, first line) | `protocol: 1`, `llm_providers: [name]`, `commands: [{name, help}]`, `load_errors: [{plugin, dist, group, error}]` (`dist`: the package, or null) | |
| app | `llm.create` | `provider`, `config`: the merged `InfraConfig` as an object, plugin keys included | `{client}` |
| app | `llm.chat` | `client`, `model`, `messages: [{role, content}]` | `{content, prompt_tokens}` |
| app | `llm.chat_single` | `client`, `model`, `prompt` | `{content}` |
| app | `llm.check_connectivity` | `client` | `{ok, message}` |
| app | `llm.check_models` | `client`, `models` | `{missing: [name]}` |
| app | `command.run` | `name`, `argv`, `config_path`, `output_dir` | `{exit_code}` |
| host | `session.transcribe` (only during `command.run`) | `audio_path`, `segments: [{text, wall_start, wall_end}]`, `label`, `template_name`, `language` (each may be null) | `{session_dir, notes_path}`, or null when the user cancelled the template picker |
| app | `shutdown` | | `{}`, then the host exits |

An error is `{code, message, data: {type}}`, where `type` is the Python exception class; the app shows it as `type: message`, as Python showed a plugin's exception. Codes: -32601 unknown method, -32602 bad params, -32000 the plugin raised.

**Inside the host.** Before it loads any entry point, the host replaces `run_transcribe` in `lazy_take_notes.plugin_api` with one that sends `session.transcribe` and waits for the answer, then returns `SessionArtifacts`. `run_record` raises a `RuntimeError` saying the Rust app does not run it yet. LLM calls run on one asyncio loop, so a digest and a quick action can run at once. `command.run` calls the click command with `standalone_mode=False` and `obj={config_path, output_dir}`; `SystemExit` becomes the exit code, any other exception prints its traceback to stderr and gives 1.

**Errors in the app.** A plugin never stops the app. Each failure says what happened and what to do:

- uv not installed: the plugin needs uv, with the install command.
- The host exits before `hello` (a bad spec, no network and no cache): the last stderr lines.
- No `hello` within the start limit (120 s; the first start installs Python and packages, and the app says so).
- A `protocol` the app does not know: update the plugins or the app.
- A plugin in `load_errors`: one warning line, as Python printed.
- The host dies or sends a line that is not JSON during a call: that call fails with the host's last stderr lines; later calls fail at once with the same text.
- An unknown provider lists the built-in and plugin providers.

**Not in the first version:** transcription providers, `run_record`, and the `llm_client=` and `transcriber=` overrides. No known plugin uses them.

## Known differences from the Python version

1. A template field with a conversion (`!r`), a format spec (`:>5`), an attribute or an index is a format error. Python accepts most of these. No built-in or known user template uses one. Python's `{name.attr}` raised an uncaught AttributeError; Rust reports a format error.
2. `Invalid JSON` and `Schema validation failed` messages in the template builder carry serde's text, not pydantic's.
3. LLM errors carry the Python class names and texts (`ResponseError: boom (status code: 500)`, `RateLimitError: Error code: 429 - {...}`), checked by `scripts/llm_parity.py`. Three cases differ: a transport failure other than a refused connection or a timeout (such as a reset) uses reqwest's text; a 200 whose body misses a field pydantic required (a server that is not Ollama, a reply with no `role`) fails with a short sentence in Rust, or not at all, instead of pydantic's text; a `Retry-After` given as an HTTP date is ignored in favor of the backoff.
4. A successful digest removes only the buffer lines it covered. Python cleared the whole buffer, so a line that arrived during the LLM call never reached a periodic digest.
5. A bare pywhispercpp model name (`base.en`, `large-v3-turbo`) uses the copy in pywhispercpp's models folder when it exists. Otherwise the store downloads it into the cache under `assets/whisper/custom/ggerganov__whisper.cpp/`. Python handed the name to pywhispercpp, which downloaded into its own folder.
6. The mic always runs at its native rate. An integer ratio (48, 96 kHz) is decimated by the mean of each group, as in Python. Any other ratio (44.1 kHz, 24 kHz call-mode headsets) goes through a windowed-sinc polyphase filter; Python used PortAudio's resampler. Mic blocks are always 512 samples.
7. On a two-channel input with `channels=1`, both versions record only the first channel. Rust reads its exact level; Python's PortAudio capture reads about 3 dB low with a slow waver.
8. A configured mic name matches an exact device name, else the one device whose name holds every word of it. sounddevice also matched the host API name.
9. When the disk has a few bytes free but less than one chunk, a retry file gets a partial write. Python counted the bytes before the write failed, so it kept that file and wrote the chunk again in the next one. Rust deletes the partial file and reuses its name.
10. Preflight error texts name Python's exception class only where Rust has a matching case: `RuntimeError` for whisper and the tap, `PortAudioError` for the mic. Other failures carry the Rust error text.
11. A panic is written to `debug.log` with a backtrace. A native crash (a segfault in a C library) is not; Python's `faulthandler` wrote one.
12. Log lines written before the session folder exists (preflight) go nowhere: `debug.log` opens with the session.
13. Plugins run in a Python host process (see "Plugins"), not in the app's own process. Plugin commands do not show in `--help`: listing them would start Python on every help. `transcribe` itself has no subtitle option; only a plugin's `run_transcribe` replays subtitles.
14. A Markdown heading wider than its panel wraps. Textual cut an H3 to one row (`width: auto`), so its end was lost.
15. A fenced code block shows no horizontal scrollbar glyph; Textual drew one even when it had no size.
16. An unexpected error toast reads `Unexpected error: <panic message>`. Python named the exception class (`Unexpected error: KeyError: 'x'`).
17. No mouse: click and wheel do nothing. The terminal's own selection copies text, in place of Textual's `ctrl+c` on selected text.
18. The session context box has no selection, undo, redo, cut, copy or paste keys of its own. The terminal's paste still works. Up and down move by logical lines, not by wrapped rows.
19. A speaker chip cannot be clicked to rename the speaker, because there is no mouse (see 17).
20. Python's Textual highlighted numbers and strings in plain text (Rich `repr` highlighting). The Rust TUI does not.
21. In the file picker, a typed path that opens a folder can leave Python showing the old folder's preview, because a late highlight event arrives after the rebuild. The Rust picker always previews the new folder's first entry. The template picker has the same race: a filter that matches nothing can leave Python showing, and acting on, the last template that matched. Rust shows "No matching templates" and `a`, `e` and `x` do nothing.
22. A template that does not load is left out of the template picker, with a warning in `debug.log`. Python crashed the picker.
23. An editor that fails to start after `e` in the template picker is logged, and the picker stays open. Python crashed.
24. Home, End, Page Up and Page Down in a list scroll the list and leave the highlight where it is, as Textual's `ListView` does. Earlier Rust builds moved the highlight.
25. A few colors differ by up to 5 levels per channel: inline code in Markdown, and dim text after Textual's rounding. For inline code, Python's background is darker than the warning color at 10% over the background, and I have not found the step in Textual that makes it so. The left edge of an input's border sits on the theme background, not the terminal's default background.
26. `record` without `--headless` shows the preflight screen in the saved theme. Python showed it in Textual's default theme there, and in the saved theme only inside the kiosk shell.
27. The "Quick actions disabled" warning shows once when a recording starts. Python showed the same toast twice.
28. A template that the builder saves, and a `config.yaml` that Settings saves, hold the same data as Python's files but not the same bytes. PyYAML writes a multi-line string in single quotes and folds long lines at 80 columns; Rust writes a `|-` block and keeps each line whole. Each version reads the other's files.
29. A setup error in the template builder (an unreadable `config.yaml`, an unknown provider) shows as `Error: <message>`. Python put the exception's class name before the message.
30. The template builder's "Thinking" dots step every 0.5 s. Python stepped every 0.4 s; the Rust screen updates four times a second.
31. A mode that cannot start in the kiosk shows `Could not start: <message>`. Python put the exception's class name before the message.
32. A worker or task result from a screen that has closed is dropped, as Textual drops a message to a removed screen. Before the kiosk shell, each Rust screen ran in its own host, so this could not happen.
33. In the kiosk, a second Transcribe asks for a file again. Python kept the first file and skipped the file picker, because the shell stored the picked path in its options.
34. `--help` uses clap's layout, not click's. The commands, flags and their descriptions are the same.
35. `plugin add` checks a spec by starting the host with it, so a plugin that installs but fails to import is refused; Python only ran `--version`. It no longer prints the setup.sh hint: the app reads plugins.txt itself. `ls`, `notes` and `transcript` are ported: on 99 real sessions, `ls`, `ls -a`, `ls --json`, both readers by name and by default, a labeled transcript, and each error print the same bytes and exit code as Python. Piped into `head`, `transcript` and `notes` end with exit 1 and no message, as click does; a plain `print!` panicked there with exit 101. `ls` and `status` write through the same path.
36. In Settings, a model row is a Tab stop and Enter chooses it. Python chose a row only with the mouse, so Tab went from one row's buttons to the next row's.
37. When config.yaml cannot be parsed, the toast quotes the Rust YAML parser's message. The sentences around it are the same.
38. A value that fails validation shows the Rust validator's message, not pydantic's.
39. The download progress bar is drawn with the same width, colors and percentage, but its partial cells are not Textual's exact glyphs.
40. In the template builder, Python showed "Unexpected error: ScreenStackError: Can't pop screen; there must be at least one screen on the stack." when Escape came right after a save. Rust shows no error there. The Python bug stays as it is.
41. The look follows ratatui's idiom, not Textual's (see "Look and mouse"). The words, the key bindings and what each key does are the same; boxes, buttons, inputs, lists, scrollbars, dialogs and toasts are drawn differently, and most screens use fewer rows.
42. Textual handled the mouse on every widget, drag included. Rust handles clicks and the wheel as in "Look and mouse", and no drag: no text selection inside an input or text area, and no scrollbar dragging.
43. Started with no command, the app always comes back to the welcome menu when a mode ends; Esc on the menu quits. Python did this only with `LTN_KIOSK=1` and otherwise exited after one mode. `LTN_KIOSK=1` still asks for the system audio permission at start, as before.
44. Speaker chips wrap onto more rows when they do not fit one (Nemotron finds up to 8 speakers); Python kept one row. View mode shows the labeled transcript and the chips, and the first click on a chip highlights that speaker's lines, and a second click renames the speaker (in view mode, in `speakers.json` and `transcript_speakers.txt`). A click in the transcript clears the highlight. View mode dims the timestamps, as the session screens do. Python's view showed the plain transcript only.
45. The stop runs one last diarization pass over the whole recording. The final notes and `q` wait for it, so the last minutes carry speaker labels too. A shutdown signal or `ctrl+q` does not wait. Python's worker only ran a pass per interval, so speech after the last pass stayed unlabeled.
46. A toast with the same text and severity as one on screen renews that toast instead of adding a copy, so a key pressed again while its job runs shows one "please wait". Textual stacked every copy. The input loop also takes every terminal event that already waits before it draws, so a wheel burst costs one frame.
47. The mic boost covers more cases than Python's. The gain may pass 10x, up to 50x, while the boosted room noise stays under the silence gate, so a mic that a call turned about 30x down still reaches the target. Speaker bleed limits the gain instead of holding it at 1x: the mic's measured pickup of the speakers keeps the boosted echo under a quarter of the call audio, so headphones keep the full gain and laptop speakers get what their echo allows. The 1 s hold at the start of a call turn applies only until a pickup is measured. A pickup known only as an upper bound (no bleed seen yet) takes a lower bound at once, so a short notification in a noisy room does not hold down a later call; a measured pickup fades over 60 s. Call audio counts as sound above 3x its own floor, not above a fixed 0.003, so steady comfort noise stops counting after about 10 s. Mic audio that the predicted echo explains does not set the speech level. The gain aims the user's speech at the call's speech level (averaged over 5 s, kept while the call is quiet), never under 0.05; with no call it aims at 0.05 as Python does. Every 30 s of audio, `debug.log` gets a `Mic boost` line: the mic and call levels and floors, the call's speech level, the user's speech level, the aim, the average gain, the share of speech boosted, the echo pickup, and the time judged or limited.
48. Each speaker pass during a recording reads a copy of `recording.wav` whose header sizes match the copied audio. The recorder writes a chunk before it updates the header, and nemo-speech reads data past the declared size as a broken chunk ("truncated WAV chunk"). Python reads the live file, and 3 of 28 passes failed that way in one 30-minute meeting. The copy costs one file copy per pass (about 115 MB per recorded hour).
49. A pause runs a speaker pass at once over any new audio, so the labels catch up while the session waits. The 60 s interval starts again after it. Python runs passes only on the interval and at the stop. Mute does not run a pass: the recording goes on.
50. `record --headless` labels the speakers once the recording ends: one pass over the whole recording, before the final digest, writes `speakers.json` and `transcript_speakers.txt`, and the final notes carry the labels. Python's headless mode runs no diarization. A failed pass keeps the plain notes with one warning line. With `save_audio: false` it warns once and skips the pass, as the TUI does.
51. The speaker bar works from the keyboard, on the record, transcribe and view screens. It is a Tab stop between the transcript and the digest once a speaker is heard. On the bar, Left and Right choose a chip (they stop at the ends), Enter opens its name prompt, and Escape drops the selection. A click on a chip also puts the focus on the bar, so Enter after one click renames. The selected chip is underlined while the bar has the focus. Python has the mouse only.
52. The debug log line `transcription triggered` says `starts_fresh=` where Python says `first_chunk=`. The value is the same: true for the first chunk and for a chunk that starts with a held segment, so the old name looked wrong on every chunk cut at 25 s.
53. The `.app` window has no welcome menu. A sidebar holds New Recording (⌘N) and Transcribe File… (⌘O), the saved sessions grouped by start day (Today, Yesterday, Previous 7 Days, Older, read from the folder name), Templates, and Settings (⌘,). Esc no longer quits; Cmd-Q, Cmd-W and the close button do. The Trolley app showed the terminal menu.
54. In the `.app`, a recording or transcription keeps running while the pane shows a saved session, Templates or Settings. Its sidebar row carries a dot (red while it records). One runs at a time: ⌘N or ⌘O during one shows it again. Closing the window still stops it through the shutdown path, whatever the pane shows. When it ends, the pane opens its saved session.
55. The `.app` follows the macOS appearance, light or dark, and changes with it. The Textual theme in Settings applies to the terminal app only.
56. In the `.app`, Transcribe File opens the macOS file dialog, not the terminal file picker.
57. The `.app` session pane has no key footer, and its one-letter keys are gone. Labeled toolbar buttons in the title bar row do the work: Pause or Resume, Mute or Unmute, Stop (Finish once stopped), Digest Now, a Quick Actions menu, and Show in Finder. Copy buttons sit in the Transcript and Digest headings. The Session menu repeats each command with a Cmd key: ⌘. stop, ⌘↩ finish, ⌘D digest, ⇧⌘M mute, ⌘1 to ⌘5 quick actions, ⇧⌘C and ⌥⌘C copy, ⌘L rename, ⇧⌘R show in Finder. Space still pauses, and Tab and the speaker keys still work. A menu item is off while its command does not apply. The TUI keys `s`, `d`, `m`, `c`, `l`, `o`, `h`, `q` and `1` to `5` do nothing in the window.
58. In the `.app`, the session title is the label, edited in place: a click on it, then Return or a click elsewhere, renames the session. Escape drops the edit. The title shows the label's underscores as spaces. The terminal app asks for the label in a prompt (`l`).
59. In the `.app`, Finish ends the session (the terminal app's `q`): the final digest runs, then the pane opens the saved session. It is a button only once the recording or transcription stopped, as `q` does nothing before `s` in the terminal app.
60. In the `.app`, Copy takes the transcript or the digest by its own button, not by the focused panel.
61. In the `.app`, one click on a speaker chip opens its name prompt, and the pointer over a chip marks that speaker's lines. The terminal app selects on the first click and renames on the second.
62. The `.app` status strip draws the state with a colored dot, the elapsed time, a bar level meter, and the capture alerts and the mute as badges. It says `Next digest: N of M lines` and `Last digest Ns ago` where the terminal bar says `buf N/M` and `last Ns ago`.
63. The `.app` Help menu shows the template, its quick actions and the keyboard shortcuts. The terminal help lists the terminal keys and the status bar legend.
64. In the `.app`, the template picker and the preflight screen are one start panel. A click selects a template and shows its preview. A double click, Return or the Start Recording (or Transcribe) button starts it. The checks then run in place of the preview. A click on another template, or Back, stops the checks and shows the preview again; the terminal app's Escape on the preflight screen leaves the flow. Cancel leaves it in both.
65. The `.app` template list has no letter keys. New, Edit and Edit with AI are buttons under the list, and Delete is a trash button that is off for a built-in template. The terminal picker uses `n`, `e`, `a` and `x`, and warns when `x` hits a built-in template. The arrows, Return, Tab and Escape still work.
66. The `.app` preflight has no `r` key: Check Again is a button. Each check shows a colored icon from the system appearance, its detail on a second line, and its fix under it. The terminal screen colors the marks pure green, yellow, red and cyan.
67. The `.app` Edit button opens the template in the macOS text editor (`open -t`) and reads it again when that window closes. The terminal app uses `$VISUAL`, then `$EDITOR`, then the platform's opener.
68. The `.app` shows a saved session in the pane, in the same layout as a live one: the transcript left, the notes right, a Copy button in each heading, the speaker chips, and Show in Finder in the title row. It replaces the session picker and the view screen. One click on a chip names the speaker, and the pointer over a chip marks their lines. The view screen marks on the first click and names on the second, and opens the folder with `o`.
69. The `.app` saved session title renames the session folder, with the same rules as a live session. The terminal view screen has no rename. It is the one `.app` command the terminal app lacks.
70. The `.app` saved session has no Tab focus between its columns and no arrow keys on the speaker chips.
71. The `.app` template builder puts the chat on the left with a message field and a Send button, the template preview on the right, and Save and Close in the title row. Save is off until the AI proposed a template, so the terminal's "generate one first" warning never shows. While the AI answers, the Send button shows "Thinking…" with a spinner; the terminal animates dots.
72. In the `.app`, leaving the template builder page through the sidebar, ⌘N, ⌘O or ⌘, asks about an unsaved template, as Close does. A recording or transcription flow keeps its builder while the pane shows another page.
73. The `.app` Settings pane puts Save and Revert under the pages. Revert loads config.yaml again and drops every unsaved change. The terminal app has no Revert.
74. The `.app` Open Config File button opens config.yaml in the macOS text editor (`open -W -t`) and reads it again when that editor quits. A config.yaml that cannot be read shows "Use Open Config File to fix it." in place of the terminal app's "Press [e] to fix the file."
75. The `.app` Settings has no Appearance page. The window follows the macOS light or dark appearance, and the terminal theme colors the terminal app only. Save keeps the theme in config.yaml as it is.
76. In the `.app` Settings pane, Esc in a field focuses the pane, as the terminal app goes back to its page list. Esc on the pane leaves Settings, and asks first when a change is not saved. The sidebar, ⌘N, ⌘O and ⌘, ask the same question. Quit, ⌘W and the close button do not ask.
77. When the terminal app's saved session viewer cannot read a file, its log line is now `<message> (<folder>)`, the same text the `.app` reader logs. It was `Cannot read <path>: <error>`.
78. The `.app` live session shows Background for AI (the session context) always open, on an accent tint, under the notes. It grows while it has focus. The terminal app shows the field without a tint.
79. The `.app` live session shows the state, the clock, the level meter and an audio alert in the title row, with Show in Finder, Pause, Mute and Stop. Digest Now, Quick Actions and Copy sit in the notes heading. The terminal app shows the state in its status bar.
80. The `.app` template preview explains the template in three cards: what the Notes panel does during the meeting and how often it updates, the quick actions to ask any time, and the words it listens for. The instructions fold under them. The terminal app shows the quick actions, the hints and the system prompt as one text.
81. The `.app` transcript draws each speaker's turn as one rounded block, with the speaker's name and the turn's start time on top. A line with no speaker label is a block of its own. The terminal app writes the time and the label before every line. Copy keeps the labeled lines with their times.
82. The `.app` starts every session from one New Session button: the user picks the template first, then Record or Transcribe File in the picker's title bar. The terminal app asks for the mode first. The menu has New Session (`⌘N`) only; `⌘O` is gone.
83. The `.app` uses plain words where the terminal app uses its own terms: Note Type (template), Import Audio (Transcribe File), Start Recording, Update Notes (Digest Now), Ask AI (quick actions), Background for AI (session context), AI Instructions, Vocabulary (hints), Edit File, Who's Speaking (speaker labels), and the Settings page Notes (Digest). Config keys and file names keep the old terms. Text that `app/` shares with the terminal app keeps them too: the activity line, the "Final digest ready" toast and the empty note type list.
84. The `.app` says "Final notes ready." (or "Final notes failed. Update Notes tries again.") when the final digest ends; the terminal app adds "Press q to quit." A file session in the `.app` shows its place in the audio (the end of the last line) and "Done" when it ends; the terminal app shows the time the work took and "Stopped". The empty note type list names the New button and Show built-in note types. The digest texts of a session (`app::session::Text`) are the terminal's by default; the `.app` says "Updating notes…", "Writing final notes…", "Update Notes writes the notes now.", "The notes are already updating.", "No new lines for the notes yet." and "Finishing when the notes are done." instead. Dark mode in the `.app` uses the macOS dark palette, not the kit's near-black theme. The `.app` names the template check "Note type", and its first-run notice about built-in note types names New and the Settings switch, not the `[n]` key.
85. The `.app` gives each speaker's turns their own color, from the macOS system colors, by the speaker's place in the speaker list. The terminal app draws every turn in one color.
86. The `.app` reads the session folder off the main thread and shows "Reading saved sessions…" until the read ends, so a slow folder or the macOS Documents prompt never keeps the window from opening. The terminal app reads it before it draws.
87. While a session runs, the `.app` disables New Session and ⌘N, and Settings disables every Remove button with "Models can't be removed while a session runs." The terminal app has no New Session or model removal during a session.
88. The `.app` binds Page Up, Page Down, Home and End to the transcript or the notes, whichever the pointer last moved over. A long speaker strip scrolls sideways with the wheel or trackpad, with no bar: the strip has no room below the chips. Both transcripts measure every line, so the scrollbar keeps one size.
89. The `.app`'s speaker name dialog has Clear Name when the speaker has a name. It removes the name from `speakers.json` and puts `S<n>` back in the transcript. The terminal app's prompt keeps the old name on an empty entry, as before.
