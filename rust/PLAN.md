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

### Platforms and Python removal (decided 2026-11-27)

This replaces "Two tracks" (2026-11-24). The Rust app covers Intel Macs and Linux, terminal and window app both. Then the Python app goes. A small Python package stays, for the launcher that existing installs run and for the plugin host.

Decisions:

- Platforms: macOS arm64 (done), macOS x86_64, Linux x86_64, Linux arm64. No Windows.
- The window app ships on all four.
- Install: `setup.sh` downloads the release tarball for the machine and checks its SHA-256. Homebrew stays for the Mac window app.
- Plugins keep working unchanged, through the Python plugin host.

Non-goals:

- Windows.
- Linux packages beyond a tarball and a `.desktop` file: no AppImage, Flatpak, deb or rpm.
- GPU transcription on Linux (Vulkan, CUDA).
- Any change to the plugin contract or its JSON-RPC protocol.
- A Developer ID signature or notarization.
- New features. Apple silicon behavior stays the same, except where a shared helper replaces a macOS call.

#### Rules for every phase

1. One branch and one PR per phase. Branch from the newest `origin/main`.
2. Before the PR, run all of these with `rust/` as the working folder. Each must pass:
   ```bash
   cargo fmt --check
   cargo clippy --all-targets -- -D warnings
   cargo clippy --features gui --all-targets -- -D warnings
   cargo test
   cargo test --features gui
   uv run scripts/check_layers.py
   uv run scripts/check_design.py
   ```
   Then `actionlint .github/workflows/*.yml` in the repo root.
3. A real run on the target platform is the proof. A green CI run is a gate, not the proof. Each phase lists its proof run.
4. Before a tag, run the Release workflow by hand on the branch (Actions > Release > Run workflow). It is a dry run and publishes nothing.
5. Record each new behavior difference between platforms in "Known differences" below, numbered after the last one.
6. If a step fails twice in the same way, stop and write down what the run showed before trying a third time.

#### What each platform lacks today

| Area | Today | Where |
|---|---|---|
| GPU | `whisper-rs` always turns on `metal`, which builds only on macOS | `Cargo.toml` |
| System audio | `CoreAudioTapSource` only, built unconditionally | `src/app/container.rs:139-147` (the preflight pair), `src/app/container.rs:150-155` (`build_mixed_source`), `src/app/preflight.rs:495-501` (`check_system_audio`) |
| System audio check | the test tone and the 30 s silence check are Core Audio tap features | `src/adapters/audio/mixed.rs`, `src/adapters/audio/microphone.rs:273` |
| Keep awake | runs `caffeinate` | `src/app/keep_awake.rs:13` |
| Open a file, window app | runs `open -W -t` | `src/gui/template_picker.rs:291`, `src/gui/settings/mod.rs:538` |
| Open a file, terminal app | already handles Linux (`xdg-open`) | `src/tui/screens/template_picker.rs:43-48` |
| Clipboard, open a folder (terminal) | already handles Linux: `wl-copy`, `xclip`, `xdg-open` | `src/tui/system.rs` |
| Window chrome | traffic lights, transparent title bar, Dock icon through cocoa and objc | `src/gui/mod.rs:126,172-175`, `src/gui/dock_icon.rs`, `src/gui/shell.rs:50`, `src/gui/design.rs:52-55` |
| Packaging | `package.sh` stops on anything but Darwin and names the tarball `<uname -m>-macos` | `scripts/package.sh:15,18` |
| `.app` packaging | arm64 only | `scripts/package_app.sh`, `.github/workflows/release.yml` |
| Python launcher pin | `PLATFORMS` maps one asset name to `('Darwin', 'arm64')` | `../scripts/pin_rust_release.py:19-21` |
| Built-in templates | the Rust binary embeds the Python package's YAML files | `src/adapters/templates.rs:20` |
| Window icon | the Dock icon reads `assets/icon.png` at build time | `src/gui/dock_icon.rs:22` |
| Speaker labels runtime | already pinned for macOS x86_64, Linux x86_64 and Linux arm64 | `src/adapters/model_catalog.rs:123-157` |
| Unix calls (`flock`, `statvfs`, signals, process groups) | work on Linux as they are | no change |

#### Phase 0: release 0.3.3 (window opens before the session list)

1. Merge PR #81 when CI passes.
2. Bump `rust/Cargo.toml` and `rust/Cargo.lock` to `0.3.3`, commit "chore: release 0.3.3", push `main`.
3. `git tag -a v0.3.3 -m "lazy-take-notes v0.3.3" && git push origin v0.3.3`.
4. When the Release run ends, check three things:
   - the log line `designated => identifier "com.cjhwong.lazy-take-notes" and certificate leaf = H"f334a7e3…"`;
   - `Casks/lazy-take-notes.rb` in CJHwong/homebrew-tap says `0.3.3`;
   - its `sha256` equals `shasum -a 256` of the released zip.
5. Run `brew update && brew upgrade --cask lazy-take-notes`.

**Proof:** this is the first upgrade between two certificate-signed builds. Open the app. The window shows at once, and macOS asks for no permission again. Start a 10 s recording: the mic and system audio work without a prompt.

#### Phase 1: Linux terminal app (x86_64 and arm64), release 0.4.0 together with Phases 2 and 4

Branch: `linux-terminal`.

1. **Metal only on macOS.** In `Cargo.toml`, change `whisper-rs = { version = "0.16.0", features = ["metal"] }` to `whisper-rs = "0.16.0"`, and add:
   ```toml
   [target.'cfg(target_os = "macos")'.dependencies]
   whisper-rs = { version = "0.16.0", features = ["metal"] }
   ```
   Check: `cargo tree -e features -i whisper-rs` on the Mac still lists `metal`.
2. **cpal's PulseAudio host on Linux.** Add:
   ```toml
   [target.'cfg(target_os = "linux")'.dependencies]
   cpal = { version = "0.18.2", features = ["pulseaudio"] }
   ```
   Cargo merges this with the base line `cpal = "0.18.2"` (`Cargo.toml:27`), so Linux gets the base dependency plus the feature. With the feature on, `cpal::default_host()` picks PulseAudio when its server runs (`cpal-0.18.2/src/platform/mod.rs:840`). PipeWire desktops run `pipewire-pulse`, so this covers them too. The feature adds the `pulseaudio` crate as a transitive dependency, a pure Rust client.
3. **System audio source.** Add `src/adapters/audio/pulse_monitor.rs`, behind `#[cfg(target_os = "linux")]`, with `PulseMonitorSource`, which implements `AudioSource` (`src/ports/audio_source.rs:46`):
   - `open()`: take `default_host().default_output_device()`, read its id with `device.id()`, and find the input device whose id is `<name>.monitor`. A `DeviceId` holds `HostId::PulseAudio` and the PulseAudio name (`cpal-0.18.2/src/host/pulseaudio/mod.rs:515-523`). Compare the name part through `DeviceId`'s accessor; do not parse its `Display` string (`pulseaudio:<name>`). Open it with the same stream code `MicrophoneSource` uses. Move that code into a shared function in `microphone.rs`; do not copy it.
   - No PulseAudio server, or no monitor: return `AudioError` with the text "No PulseAudio or PipeWire monitor for the default output". Preflight shows that text.
   - `continuous_capture()` returns `true`: a monitor delivers silence all the time, so a gap means the stream is stuck.
   - Tests in `pulse_monitor_tests.rs`: the pure function `monitor_id(output: &str) -> String`, and the pick from a list of ids (match, no match, two sinks).
4. **One place picks the system source.** In `src/app/container.rs`, add `pub fn system_audio_source(ensure_permission: bool) -> Arc<dyn AudioSource>`:
   - macOS: `CoreAudioTapSource::new(ensure_permission)`;
   - Linux: `PulseMonitorSource::new()`.

   Replace the uses in `container.rs:139-147`, `container.rs:150-155` and `preflight.rs:495-501`. The `coreaudio_tap` module, and its `use` lines, become `#[cfg(target_os = "macos")]`.
5. **Silence check and test tone, macOS only.** `play_test_tone` (`microphone.rs:273`) and the 30 s system-silence check in `mixed.rs` exist because the tap can hang silently. Put them behind `#[cfg(target_os = "macos")]`. On Linux, `SYS LOST` comes only from a failed read or no delivery for 3 s. Add a known difference.
6. **Hide monitors from the mic list.** On Linux, `list_input_devices()` (`microphone.rs:30-40`) would list "Monitor of …" sources. It collects names, and the config and `match_device` use names. In its loop, skip a device whose `device.id()` name part ends in `.monitor`, then keep the name as now. Test the pure filter on `(id, name)` pairs.
7. **Keep awake on Linux.** In `src/app/keep_awake.rs`:
   - Linux: run `systemd-inhibit --what=idle:sleep --who=lazy-take-notes --why=Recording --mode=block sleep infinity`, and kill it on drop, like `caffeinate`.
   - Not found: log one `debug!` line and go on.
   - Keep the macOS branch as it is.
8. **Open a file.** Add `src/app/open_path.rs` with `pub fn open_path(path: &Path, text_editor: bool) -> std::io::Result<std::process::Child>`:
   - macOS: `open` (with `-t` when `text_editor`). The window app waits with `-W`, so add a `wait: bool` argument that adds `-W`.
   - Linux: `xdg-open` (the `text_editor` and `wait` arguments do nothing).

   Replace the two window-app call sites, `gui/template_picker.rs:291` and `gui/settings/mod.rs:538`. Both run `open -W -t`. Leave `tui/screens/template_picker.rs` and `tui/system.rs` as they are: they already handle Linux. Test the argument list per OS with a pure `fn command(path, text_editor, wait, os) -> Vec<String>`.
9. **Package for Linux.** `scripts/package.sh`:
   - Remove the Darwin-only stop.
   - Name the tarball `lazy-take-notes-<arch>-<os>.tar.gz`, with `<arch>` from `uname -m` mapped as `x86_64 -> x86_64`, `aarch64|arm64 -> arm64`, and `<os>` `macos` or `linux`.
   - Copy `coreaudio-tap` only on macOS.
10. **CI.** In `.github/workflows/ci.yml`, add a job `Rust (Linux ${{ matrix.arch }})` with a matrix of `ubuntu-24.04` (x86_64) and `ubuntu-24.04-arm` (arm64):
    - `sudo apt-get install -y cmake clang libclang-dev libpulse-dev libasound2-dev`
    - `rustup toolchain install`
    - `cargo test`, `cargo clippy --all-targets -- -D warnings`
    - `uv run scripts/check_layers.py`
11. **Release.** In `.github/workflows/release.yml`:
    - Add a job `linux` with the same matrix and apt packages. It runs `package.sh` and checks `--version` and `--help` from the extracted tarball.
    - Each build job uploads its tarball and `.sha256` as a workflow artifact.
    - Today "Publish the release" is a step inside job `macos-arm64` (`release.yml:93`), and job `homebrew` has `needs: macos-arm64` (`release.yml:111`). Move publishing into a new job `publish` with `needs: [macos-arm64, linux]` and the same `if:` as the old step. It downloads every artifact and publishes once. Change `homebrew` to `needs: publish`.
12. **Python launcher pin.** `PLATFORMS` in `../scripts/pin_rust_release.py:19-21` maps a release asset name to `(platform.system(), platform.machine())`. Add these entries, so `uvx` users on Linux run the Rust build after the 0.4.0 pin:
    - `'lazy-take-notes-x86_64-linux.tar.gz': ('Linux', 'x86_64')`
    - `'lazy-take-notes-arm64-linux.tar.gz': ('Linux', 'aarch64')`

    Add the Intel entry in Phase 2, not here: the script fails on an asset the release does not have.
13. **Smoke test in Docker.** Replace `../scripts/linux-smoke/Dockerfile` and `smoke_test.py`:
    - `FROM rust:1-bookworm`. Install `pulseaudio pulseaudio-utils libpulse-dev libasound2-dev cmake clang libclang-dev curl`. Copy the repo, run `rustup toolchain install` in `rust/`, then `cargo build --release`, then `install -m755 rust/target/release/lazy-take-notes /usr/local/bin/`.
    - Download the speech model in the image, so the test needs no model catalog:
      ```bash
      curl -fL -o /models/ggml-base.en.bin https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin
      ```
    - Write the test config `/home/testuser/ltn.yaml`, mode 0600:
      ```yaml
      transcription:
        model: /models/ggml-base.en.bin
      llm_provider: ollama
      ollama:
        host: http://127.0.0.1:9
      ```
      Port 9 has no server, so the AI check fails and headless runs transcript-only.
    - As a non-root user (PulseAudio refuses root), run:
      ```bash
      pulseaudio --start --exit-idle-time=-1
      pactl load-module module-null-sink sink_name=test_loopback
      pactl set-default-sink test_loopback
      lazy-take-notes record --headless --config ~/ltn.yaml --template default_en --output-dir /tmp/out &
      sleep 8   # the speech model loads first
      paplay scripts/linux-smoke/speech.wav
      sleep 5
      kill -TERM %1; wait %1
      ```
    - SIGTERM in headless mode still runs the last chunk and the final digest (AGENTS.md "Shutdown signals"). The digest fails here, because no AI server runs. Run the container once by hand and write down the exit code it gives. The test asserts that code, not a guessed one.
    - Pass when:
      - the exit code is the one written down;
      - `/tmp/out/*/transcript.txt` holds at least half of the words in `scripts/linux-smoke/speech.txt`;
      - `/tmp/out/*/debug.log` has no `ERROR` line about system audio.
    - `speech.wav`: take one sentence of 5 to 10 s from a LibriVox recording (public domain), and convert it with `ffmpeg -i in.mp3 -ss <start> -t <length> -ar 16000 -ac 1 speech.wav`. Write its words into `speech.txt`. Commit both under `../scripts/linux-smoke/`.
    - AGENTS.md "Linux Smoke Test" describes the old image. Update its text in the same PR.
    - The keys are checked: `ollama.host` is `InfraConfig.ollama.host` (`src/config.rs:182`), and `transcription.model` applies to `default_en`, because the built-in `transcription.models` has entries only for `zh` and `zh-min-nan` (`src/config.rs:13-17`).

**Proof:**
- `docker build -f scripts/linux-smoke/Dockerfile -t ltn-linux-smoke . && docker run --rm ltn-linux-smoke` passes on x86_64.
- The same passes on arm64 (`--platform linux/arm64`).
- The macOS arm64 build still records a 20 s call with system audio.

#### Phase 2: Intel Mac terminal app

Branch: `intel-mac`, or the Phase 1 branch if Phase 1 is still open.

1. **Cross-build on the arm64 runner.** `rust-toolchain.toml` has no `targets` key. Add `targets = ["x86_64-apple-darwin"]` under `[toolchain]`. Build with:
   ```bash
   CMAKE_OSX_ARCHITECTURES=x86_64 cargo build --release --target x86_64-apple-darwin
   ```
   `package.sh` takes an optional `--target <triple>` and reads the binary from `target/<triple>/release/`. It takes `<arch>` from the triple when `--target` is given (`x86_64-apple-darwin` → `x86_64`, `aarch64-apple-darwin` → `arm64`), else from `uname -m` as now (`package.sh:15`). `coreaudio-tap` is already universal: check with `lipo -archs`.
2. **Release.** In the macOS release job, build and package both targets, and upload `lazy-take-notes-x86_64-macos.tar.gz` and its `.sha256`.
3. **CI smoke on real Intel.** Add a release job on `macos-15-intel` (an x64 image, listed in actions/runner-images) that downloads the x86_64 tarball artifact and runs `--version` and `--help`. Add it to the `needs:` of `publish`.
4. **Pin.** Add `'lazy-take-notes-x86_64-macos.tar.gz': ('Darwin', 'x86_64')` to `PLATFORMS` in `../scripts/pin_rust_release.py`.
5. **Metal on Intel.** Keep `metal`. If whisper's Metal backend fails on the Intel runner's GPU, drop `metal` for `x86_64-apple-darwin` only and record it.

**Proof:** on an Apple silicon Mac, extract the x86_64 tarball and run under Rosetta:
- `arch -x86_64 ./lazy-take-notes record --headless` for 20 s with a played sentence: the words appear in `transcript.txt`;
- `arch -x86_64 ./lazy-take-notes transcribe --headless <wav>`: the same words.

#### Phase 3: window app on Intel Mac and Linux, release 0.5.0

Branch: `gui-everywhere`.

1. **Intel `.app`.** `scripts/package_app.sh` takes `--target <triple>` like `package.sh`:
   - It writes `target/app/<arch>/Lazy Take Notes.app`.
   - The release job zips each `.app` as `Lazy-Take-Notes-<arch>-macos.zip`. The arm64 name stays the same, since the 0.3 casks point at it.
2. **Cask for both arches.** `scripts/homebrew_cask.sh VERSION SHA_ARM SHA_INTEL`:
   - Drop `depends_on arch: :arm64`.
   - Put `sha256` and `url` in `on_arm do … end` and `on_intel do … end` blocks.

   Run `brew style --cask` on the output and `brew install --cask ./lazy-take-notes.rb` from a local file before release.
3. **gpui on Linux, first a spike.** On an x86_64 Linux machine or in Docker, run `cargo build --features gui`.
   - Install what the build asks for, starting with `libxkbcommon-dev libxkbcommon-x11-dev libwayland-dev libx11-xcb-dev libvulkan-dev libfontconfig-dev libssl-dev`.
   - If gpui-kit 0.11 does not build or does not open a window, stop the phase and report. The terminal app still ships.
4. **macOS-only window code behind `cfg`.**
   - `Cargo.toml`: move the two lines `cocoa = { version = "=0.25.0", optional = true }` and `objc = { version = "=0.2.7", optional = true }` into `[target.'cfg(target_os = "macos")'.dependencies]`. Keep `gui = ["dep:gpui-kit", "dep:cocoa", "dep:objc"]` as it is: Cargo builds a target-specific optional dependency only on its target, so Linux never builds them. Check with `cargo tree --features gui --target x86_64-unknown-linux-gnu -i cocoa`, which must print nothing.
   - `src/gui/mod.rs:9,126`: `#[cfg(target_os = "macos")]` on `mod dock_icon` and its call.
   - `src/gui/mod.rs:172-175`: on macOS keep `appears_transparent: true` and `traffic_light_position`. On Linux use `TitlebarOptions::default()` and let gpui draw its decorations.
   - `src/gui/shell.rs:50` and `src/gui/design.rs:52-55`: the traffic-light inset becomes a `design.rs` token that is zero on Linux, so `check_design.py` still passes.
   - `src/gui/theme.rs` `follow_system`: on Linux, read gpui's window appearance. If gpui reports nothing, use light.
5. **Linux window package.** Package the GUI binary as `lazy-take-notes-gui-<arch>-linux.tar.gz`, with the binary, `lazy-take-notes.desktop` and `icon.png`. Make the icon from `../assets/icon.png` with `sips -Z 512` (macOS) or `convert -resize 512x512` (Linux). The `dock_icon.rs:22` embed is macOS only after step 4, so Linux needs the file in the package. The `.desktop` file:
   ```ini
   [Desktop Entry]
   Type=Application
   Name=Lazy Take Notes
   Exec=lazy-take-notes-gui
   Icon=lazy-take-notes
   Categories=AudioVideo;Office;
   ```
   Add it to the release job for both Linux arches, with `.sha256`.
6. **CI.** The Linux CI job also runs `cargo clippy --features gui --all-targets -- -D warnings` and `cargo test --features gui`, with the window packages installed.

**Proof:**
- Intel `.app`, under Rosetta, opened from Finder: the window, a 20 s recording with system audio, Settings, the viewer, the template builder.
- Linux, in Docker under Xvfb with Mesa's software Vulkan (`mesa-vulkan-drivers`): launch, then screenshots of the sidebar, a saved session, Settings and preflight. Compare them to the macOS screens: same layout, no clipped text.
- **A real Linux desktop** (Wayland and X11, a real GPU) before the 0.5.0 tag: the window opens, a 20 s recording with system audio, Settings saves to a temp config, the viewer opens a session. The owner provides the machine. Xvfb proves only that it renders.

#### Phase 4: `setup.sh` installs the binary, part of 0.4.0

Branch: with Phase 1. The file is `../setup.sh` in the repo root. Today its "Step 1: uv" (`setup.sh:77-91`) installs uv; the steps below replace that flow.

1. Pick the asset from `uname -s` and `uname -m`:
   - `Darwin x86_64` → `x86_64-macos`
   - `Darwin arm64` → `arm64-macos`
   - `Linux x86_64` → `x86_64-linux`
   - `Linux aarch64` → `arm64-linux`
   - Anything else stops with the list of supported systems.
2. Read the newest release tag from `https://api.github.com/repos/CJHwong/lazy-take-notes/releases/latest`. `LTN_VERSION=v0.4.0` overrides it.
3. Download the tarball and its `.sha256`, compare with `shasum -a 256` (macOS) or `sha256sum` (Linux), and stop on a mismatch.
4. Extract into `~/.local/share/lazy-take-notes/<tag>/`, and link `~/.local/bin/lazy-take-notes` and `~/.local/bin/take-note` to it. From 0.5.0 on, when the release has a Linux GUI tarball, also download it, link `lazy-take-notes-gui`, and copy the `.desktop` file to `~/.local/share/applications/` and the icon to `~/.local/share/icons/hicolor/512x512/apps/lazy-take-notes.png`.
5. If `~/.local/bin` is not on `PATH`, print the line to add for the user's shell. Do not edit shell files.
6. On macOS with Homebrew, print `brew install --cask CJHwong/tap/lazy-take-notes` as the recommended way, and continue with the tarball only after a yes.
7. Keep the AI provider steps as they are. Drop the uv install step. The plugin how-to says to install uv before adding a plugin.
8. An existing `take-note` that is the old uvx wrapper (a script that contains `uvx`): replace it, and say so in one line.

**Proof:** rewrite `../scripts/setup-smoke/Dockerfile` and `run_tests.sh` for the tarball flow (today they test the uv and Homebrew flow). Then run them in Docker, Debian bookworm, x86_64 and arm64:
- install from a clean HOME;
- `take-note --version` prints the release;
- a second run of `setup.sh` reports "already installed" and changes nothing;
- `take-note record --headless` against the null sink from Phase 1 writes a transcript.

On macOS, run `setup.sh` with `HOME` set to a temp folder.

#### Phase 5: remove the Python app, release 0.6.0

Start only after 0.5.0 has run on the real Linux desktop.

Branch: `remove-python-app`.

**What must keep working.** Every install made by the old `setup.sh` has a `take-note` wrapper that runs `uvx --from git+https://github.com/CJHwong/lazy-take-notes.git lazy-take-notes` on each start ("The switch" above). That command must keep starting the Rust app. So the package keeps its name and its three scripts:

| Script | After Phase 5 |
|---|---|
| `lazy-take-notes`, `take-note` | `launcher.py`: runs the pinned Rust build. With no build for the machine, or any failure, it prints one line naming `setup.sh` and exits 1. No Python fallback. |
| `lazy-take-notes-plugin-host` | `plugin_host.py`, unchanged |

Steps:

1. **Keep in `src/lazy_take_notes/`:**
   - `launcher.py` (fallback removed);
   - `rust_release.py`;
   - `plugin_host.py`;
   - `plugin_api.py`.
   - What `plugin_api` re-exports, moved flat into the package root, with the same public names:
     - `ChatMessage`, `SessionArtifacts`, `TranscriptSegment`;
     - the ports `AudioSource`, `LLMClient`, `ChatResponse`, `ModelResolver`, `Transcriber`;
     - `InfraConfig`, `TranscriptionBackend`;
     - stubs for `run_record` and `run_transcribe` that raise `RuntimeError("runs only inside the plugin host")`. The host replaces both at runtime (`plugin_host.py:129-168`).
   - Keep the old import paths that plugins may use as one-line re-export modules: `lazy_take_notes.l4_frameworks_and_drivers.plugin_api` and `lazy_take_notes.l4_frameworks_and_drivers.config` (the host imports it at `plugin_host.py:342`).
2. **Move the templates into the Rust tree.** `src/adapters/templates.rs:20` embeds `../../../src/lazy_take_notes/templates/<name>.yaml`. Move `../src/lazy_take_notes/templates/` to `rust/templates/`, change the path in the `include_str!` to `../../templates/`, and run `cargo test` before deleting anything else.
3. **Move `coreaudio-tap` into the Rust tree.**
   - Move `native/` to `rust/native/`, and `scripts/build_native.sh` to `rust/scripts/build_native.sh`.
   - `package.sh` and `package_app.sh` build the helper into `rust/target/native/coreaudio-tap` when it is missing, instead of reading `src/lazy_take_notes/_native/bin/`.
   - Delete `src/lazy_take_notes/_native/`.
4. **Delete** (only after steps 2 and 3 pass `cargo test`):
   - the rest of `l1_entities/`, `l2_use_cases/`, `l3_interface_adapters/`, `l4_frameworks_and_drivers/`;
   - `__main__.py`;
   - their tests under `tests/`, except `test_launcher.py`, `test_plugin_host.py` and the fakes they use;
   - `scripts/release.sh`;
   - the Python-only docs.
5. **`pyproject.toml`.** Keep only what the kept files import: `platformdirs` for `launcher.py`, `pydantic` for `InfraConfig`, `click` for plugin commands. Drop textual, pywhispercpp, sounddevice, soundcard, numpy, openai, ollama and the rest. Run `uv lock`.
6. **Lint config.** In `pyproject.toml`, delete every `import-linter` contract that names a deleted layer, and the `lint-imports` hook from `.pre-commit-config.yaml` if no contract is left. `ty` and `ruff` keep running on the kept package.
7. **CI.** The Python matrix runs only `tests/test_launcher.py` and `tests/test_plugin_host.py`.
8. **Docs:**
   - README: install is Homebrew on Mac or `setup.sh` everywhere, and the Rust and Python split note goes.
   - `docs/explain_rust_and_python.md` is deleted.
   - The plugin how-to states that plugins need uv and run in the plugin host.
   - AGENTS.md: delete the Python architecture, Python data flow and Python test sections. The "Rust Layers" table becomes the layout section. Keep the plugin contract.
   - `rust/PLAN.md` "The switch": the launcher no longer falls back to Python.
9. **Last pin.** After the 0.6.0 release, run `uv run scripts/pin_rust_release.py v0.6.0`, so every platform's old wrapper runs 0.6.0.

**Proof:**
- `uvx --from git+https://github.com/CJHwong/lazy-take-notes.git@remove-python-app take-note --version` prints the Rust version on macOS arm64, on macOS x86_64 under Rosetta, and on Linux in Docker.
- The same command with `LTN_PYTHON=1` no longer starts a Python app. It prints the `setup.sh` line.
- With the owner's real `ltn-claude-code` plugin in `plugins.txt` and `LTN_PLUGIN_HOST_FROM` pointing at the branch: the plugin's subcommand runs, and a session that uses its LLM provider writes a digest.
- `uvx --from <branch> lazy-take-notes-plugin-host` starts and answers a JSON-RPC request.

#### Releases

| Version | Content | Gate before the tag |
|---|---|---|
| 0.3.3 | Phase 0 | the permission check on upgrade |
| 0.4.0 | Phases 1, 2, 4: Linux and Intel terminal app, new `setup.sh`, launcher pin for all four platforms | the Linux Docker smoke on both arches, Rosetta runs, setup-smoke |
| 0.5.0 | Phase 3: window app on Intel and Linux, two-arch cask | Rosetta `.app` run, Xvfb screenshots, the real Linux desktop run |
| 0.6.0 | Phase 5: Python app removed | the uvx wrapper, plugin and host runs |

#### What the owner provides

1. A real Linux desktop (a machine or a VM) for the Phase 3 gate.
2. The source of `ltn-claude-code`. The copy under `plugins/` holds only `.pyc` files.

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
