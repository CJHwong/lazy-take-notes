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
- session folders: transcript, notes, history, context, `recording.wav` and `recording_N.wav`, `debug.log`, `.status.json`
- model cache: `assets/`, `downloads/`, `locks/`, `state.json`
- CLI subcommands and flags, key bindings

## Non-goals

- Python plugin compatibility
- New features, or behavior changes during the port
- Removing the Python version

## Verification

- `scripts/gen_fixtures.py` records Python outputs into `tests/fixtures/`. `tests/parity.rs` asserts the Rust output is byte-equal. Rerun the script when the Python side changes.
- `scripts/llm_parity.py` drives both versions' LLM clients against one failing local server and compares the shown text and the request count.
- `scripts/cross_check.py` compares both versions on copies of the real config, templates, sessions and model cache.
- Each fixed bug listed in AGENTS.md gets a named Rust test.
- Before a step is done: a real run (headless record of a call, transcribe of a saved file, Rust `view` of a session Python recorded, and the reverse).

## Checklist

1. [x] Domain types, ports, use cases (digest, compaction, label, quick actions, speaker attribution, template builder, validator, diff, transcription buffering and speech gate)
2. [x] Persistence and assets: config loader and writer, template loader and writer, session files, session reader and status, model catalog, cached store, download engine, file lock
3. [x] LLM clients (Ollama, OpenAI-compatible), whisper-rs transcriber, audio file loader
4. [ ] Audio: cpal mic source, Core Audio tap source (existing Swift binary), mixed source with retry and health, mic boost, WAV writer with backlog, test tone
5. [ ] Session controller, headless record, preflight, shutdown signals, diarization (existing NeMo runtime)
6. [ ] TUI: shell, pickers, record, transcribe, view, preflight screen, settings, template builder
7. [ ] Cross-version run: each version opens the other's sessions and config

## Known differences from the Python version

1. A template field with a conversion (`!r`), a format spec (`:>5`), an attribute or an index is a format error. Python accepts most of these. No built-in or known user template uses one. Python's `{name.attr}` raised an uncaught AttributeError; Rust reports a format error.
2. `Invalid JSON` and `Schema validation failed` messages in the template builder carry serde's text, not pydantic's.
3. LLM errors carry the Python class names and texts (`ResponseError: boom (status code: 500)`, `RateLimitError: Error code: 429 - {...}`), checked by `scripts/llm_parity.py`. Three cases differ: a transport failure other than a refused connection or a timeout (such as a reset) uses reqwest's text; a 200 whose body misses a field pydantic required (a server that is not Ollama, a reply with no `role`) fails with a short sentence in Rust, or not at all, instead of pydantic's text; a `Retry-After` given as an HTTP date is ignored in favor of the backoff.
4. A successful digest removes only the buffer lines it covered. Python cleared the whole buffer, so a line that arrived during the LLM call never reached a periodic digest.
5. A bare pywhispercpp model name (`base.en`, `large-v3-turbo`) uses the copy in pywhispercpp's models folder when it exists. Otherwise the store downloads it into the cache under `assets/whisper/custom/ggerganov__whisper.cpp/`. Python handed the name to pywhispercpp, which downloaded into its own folder.
