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
- Each fixed bug listed in AGENTS.md gets a named Rust test.
- Before a step is done: a real run (headless record of a call, transcribe of a saved file, Rust `view` of a session Python recorded, and the reverse).

## Checklist

1. [x] Domain types, ports, use cases (digest, compaction, label, quick actions, speaker attribution, template builder, validator, diff, transcription buffering and speech gate)
2. [ ] Persistence and assets: config loader and writer, template loader and writer, session files, session reader and status, model catalog, cached store, download engine, file lock
3. [ ] LLM clients (Ollama, OpenAI-compatible), whisper-rs transcriber, audio file loader
4. [ ] Audio: cpal mic source, Core Audio tap source (existing Swift binary), mixed source with retry and health, mic boost, WAV writer with backlog, test tone
5. [ ] Session controller, headless record, preflight, shutdown signals, diarization (existing NeMo runtime)
6. [ ] TUI: shell, pickers, record, transcribe, view, preflight screen, settings, template builder
7. [ ] Cross-version run: each version opens the other's sessions and config

## Known differences from the Python version

1. A template field with a conversion (`!r`), a format spec (`:>5`), an attribute or an index is a format error. Python accepts most of these. No built-in or known user template uses one. Python's `{name.attr}` raised an uncaught AttributeError; Rust reports a format error.
2. `Invalid JSON` and `Schema validation failed` messages in the template builder carry serde's text, not pydantic's.
3. The LLM error kind in `LLM error: <kind>: <message>` comes from the Rust HTTP client.

## Python bugs kept on purpose

1. A successful digest clears the whole buffer, including lines that arrived during the LLM call. The next periodic digest never sees them; the final digest does. Fix in both versions, or in Rust after the swap.
