# Rust rewrite plan

The Rust version replaces the Python app once it is stable. Until then the Python version stays on PyPI and in `src/`.

## Decisions

- Location: `rust/` in this repo, on the `rust-rewrite` branch.
- Platforms at the swap: macOS. Linux and Windows system audio come later; Linux loopback needs a spike first.
- Plugins: none in the first version. Ollama and OpenAI-compatible are built in.
- Data: drop-in. Both versions read and write the same files.

## Data contract

The two versions must work on the same data, in both directions:

- `config.yaml`: same path, same schema, `null` language overrides
- template YAML, built-in and user
- session folders: transcript, notes, history, context, `recording.wav` and `recording_N.wav`, `debug.log`, `.status.json`, `speakers.json` and `transcript_speakers.txt` (written by `diarize`)
- model cache: `assets/`, `downloads/`, `locks/`, `state.json`
- CLI subcommands and flags, key bindings

## Non-goals

- Python plugin compatibility
- New features, or behavior changes during the port
- Removing the Python version

## Verification

- `scripts/gen_fixtures.py` records Python outputs into `tests/fixtures/`. `tests/parity.rs` asserts the Rust output is byte-equal. The mic boost is the exception: numpy and Rust round float32 sums differently, so its gain and level must agree within 1e-4 per block, and its bleed state exactly. Rerun the script when the Python side changes.
- `scripts/llm_parity.py` drives both versions' LLM clients against one failing local server and compares the shown text and the request count.
- `scripts/whisper_parity.py` transcribes the start of a real recording in both versions the way the file worker does, with a template's hints, and compares every segment's text and times.
- `examples/cross_check.rs record DIR SECONDS` records the real mic and system audio into a session WAV, and prints health changes and the delivered audio against wall time. `tone` measures the test tone in the tap. `mic-levels DEVICE SECONDS` reads one input device. `list-devices SECONDS` prints the input list every 0.5 s. `record DIR SECONDS DEVICE` records from a chosen input; changing that device's nominal rate mid-run tests the mic's stop and reopen. Fault runs: `kill -STOP` and `kill -9` on the tap pid, and a small `hdiutil` disk image filled, freed, with the WAV moved and the folder removed.
- `scripts/tui_compare.sh "ARGS" KEY... snap` runs both TUIs in tmux at one size, sends the same keys, and diffs the screen text at each `snap`. `scripts/gen_bindings.py` records every Python screen's merged key table into `tests/fixtures/bindings.json`; `tests/tui_bindings.rs` checks the Rust tables against it. `scripts/gen_themes.py` writes Textual's generated theme colors into `src/tui/themes.json`.
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
6. [ ] TUI: shell, pickers, record, transcribe, view, preflight screen, settings, template builder
   - Done: pickers, record (with the preflight screen), transcribe, view, template builder, the kiosk shell (bare `lazy-take-notes`, `LTN_KIOSK=1`). Left: settings.
   - Real runs: each screen next to Python's in tmux, the same keys sent to both. The builder ran against a stub OpenAI-compatible server: both sent the same requests, and the screens matched through generate, auto-fix, save, edit and the quit confirm.
   - The kiosk ran a recording, stopped it, and came back to the welcome picker with no whisper, `coreaudio-tap` or `caffeinate` child left and the WAV closed. A cancelled preflight also leaves no child.
   - A closed window (SIGHUP) during a recording: the audio flushes and the transcript is saved; failed draws are dropped.
   - Python's `drivers/terminal_output.py` (retry EBADF on fd 2) exists only because PortAudio calls `dup2()` onto fd 2 when it re-initializes. cpal never re-initializes and never touches fd 2, so the Rust TUI does not need it.
7. [ ] Cross-version run: each version opens the other's sessions and config
   - Ship `coreaudio-tap` next to the executable. A release build looks for it only there.

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
13. `transcribe` has no subtitle replay: that came only from plugins, which are dropped.
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
25. A few colors differ by up to 5 levels per channel: inline code in Markdown, and dim text after Textual's rounding. The left edge of an input's border sits on the theme background, not the terminal's default background.
26. `record` without `--headless` shows the preflight screen in the saved theme. Python showed it in Textual's default theme there, and in the saved theme only inside the kiosk shell.
27. The "Quick actions disabled" warning shows once when a recording starts. Python showed the same toast twice.
28. A template that the builder saves, and a `config.yaml` that Settings saves, hold the same data as Python's files but not the same bytes. PyYAML writes a multi-line string in single quotes and folds long lines at 80 columns; Rust writes a `|-` block and keeps each line whole. Each version reads the other's files.
29. A setup error in the template builder (an unreadable `config.yaml`, an unknown provider) shows as `Error: <message>`. Python put the exception's class name before the message.
30. The template builder's "Thinking" dots step every 0.5 s. Python stepped every 0.4 s; the Rust screen updates four times a second.
31. A mode that cannot start in the kiosk shows `Could not start: <message>`. Python put the exception's class name before the message.
32. A worker or task result from a screen that has closed is dropped, as Textual drops a message to a removed screen. Before the kiosk shell, each Rust screen ran in its own host, so this could not happen.
