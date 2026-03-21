# Agent Context for lazy-take-notes

> For usage, installation, and configuration see [README.md](README.md).

## Operational Commands (Primary Instructions)

Use `uv` for all project management tasks.

- **Setup/Sync**:
```bash
uv sync
```

* **Run App**:
```bash
uv run lazy-take-notes
```


* **Run Tests**:
```bash
uv run pytest tests/ -v
```


* **Verify Architecture (Import Contracts)**:
```bash
uv run lint-imports
```

* **Linux Smoke Test** (requires Docker):
```bash
docker build -f scripts/linux-smoke/Dockerfile -t ltn-linux-smoke .
docker run --rm ltn-linux-smoke
```
Runs the full pytest suite on Linux (Debian bookworm) with PulseAudio, then exercises `SoundCardLoopbackSource` against a real PulseAudio null-sink loopback device. Verifies device discovery, audio capture, and clean shutdown.


## Code Style

* **`noqa` must include a reason**: Every `# noqa: XXXX` comment **MUST** have a `--` reason suffix explaining why the suppression is justified.
  * Good: `# noqa: S603 -- fixed arg list, not shell=True`
  * Bad: `# noqa: S603`


## Testing Guidelines

* **Coverage target**: Aim for **100% line coverage** whenever possible. Every new feature or fix must include tests. Use `# pragma: no cover` only for code that genuinely cannot be tested in isolation (e.g. hardware-dependent thread launchers, subprocess wiring).
* **Architecture**: The project follows **Clean Architecture** (L1 Entities -> L2 Use Cases -> L3 Adapters -> L4 Frameworks).
* **Test directory convention**: `tests/` mirrors the source tree one-to-one. Each source subdirectory has a corresponding test subdirectory (e.g. `l3_interface_adapters/gateways/` → `tests/l3_interface_adapters/gateways/`). New test files go in the matching subdir.
* **Mocking Strategy**:
* Do **NOT** use `@patch` on concrete libraries (e.g., `ollama`, `openai`, `sounddevice`) in L2 tests.
* **MUST** use the provided protocol-conforming fakes located in `tests/conftest.py` (e.g., `FakeLLMClient`, `FakeTranscriber`).
* Library mocking is only permitted at the L3 (Adapter) boundary.


## Concurrency Model

```
Audio Worker (thread)          Digest Task (async)          Query Task (async)
  AudioSource + whisper          LLM heavy model              LLM fast model
        │                              │                           │
        │ TranscriptChunk              │ DigestReady               │ QueryResult
        │ AudioWorkerStatus            │ DigestError               │
        │ AudioLevel                   │                           │
        ▼                              ▼                           ▼
  ┌─────────────────────── App (event loop) ───────────────────────┐
  │ on_transcript_chunk → update panel, buffer lines, trigger      │
  │ on_digest_ready → update panel, persist to disk                │
  │ on_query_result → show modal                                   │
  │ Digest trigger: buffer >= min_lines AND elapsed >= min_interval│
  │                 OR buffer >= max_lines (force-trigger)         │
  │ Mutual exclusion: digest + query tasks run exclusive=True      │
  └────────────────────────────────────────────────────────────────┘
```

Digest and query are on-demand async tasks (`self.run_worker(..., exclusive=True, group=...)`) — not persistent background workers. `_digest_running` / `_query_running` flags prevent double-firing.

## Audio Capture

Always uses `MixedAudioSource` (mic + system audio blended, 0.5 attenuation anti-clipping). No audio mode picker — the user presses `[m]` during recording to mute/unmute the microphone.

- macOS: `MixedAudioSource(SounddeviceAudioSource(), CoreAudioTapSource())`
- Linux/Windows: `MixedAudioSource(SounddeviceAudioSource(), SoundCardLoopbackSource())`

`DependencyContainer._build_mixed_source()` selects the system audio backend by platform. `MixedAudioSource.mic_muted` flag (bool, GIL-atomic) controls mic muting at read time — reader threads always run.

## Data Flow

1. **Audio capture**: AudioSource → numpy float32 buffer → VAD trigger → whisper transcribe (off-thread) → TranscriptSegment
2. **Transcript buffering**: App receives TranscriptChunk → updates panel, persists to disk, appends to DigestState.buffer
3. **Digest trigger**: When buffer >= min_lines AND elapsed >= min_interval, or buffer >= max_lines (force) → launch async digest task
4. **Digest cycle**: Template-driven prompt (with user session context) → LLMClient.chat → JSON parse → DigestData → persist + update panel
5. **Token compaction**: When prompt_tokens exceeds threshold, conversation history is compacted to 3 messages (system, compressed state, last response)
6. **Quick actions**: Positional keybinding (1–5) → format prompt from template with current digest + recent transcript → LLM fast model → modal display
7. **File transcription**: `lazy-take-notes transcribe <file>` → TemplatePicker → ffmpeg decode → chunked transcription via `FileTranscriptionWorker` (thread) → streaming TUI output + auto-trigger final digest on completion
8. **Session viewer**: `lazy-take-notes view` → SessionPicker (browse saved sessions) → ViewApp (read-only, standalone TextualApp — no controller/workers)
9. **Recording**: When `save_audio: true`, WAV is written alongside output — always processed 16 kHz int16

## Design Decisions

- **Thread worker for audio**: sounddevice and whisper.cpp are blocking C libraries, cannot run in asyncio. Transcription runs in a `ThreadPoolExecutor` within the audio worker thread.
- **Async tasks for LLM**: LLMClient (Ollama or OpenAI-compatible) integrates with Textual's event loop via async methods. Digest and query are spawned on-demand with `exclusive=True` to prevent overlapping calls.
- **Single-threaded state**: DigestState lives on the controller, only mutated on event loop — no locks needed
- **Template-driven**: All prompts, labels, and quick actions are defined in YAML templates — core logic is locale-agnostic
- **Message passing**: Workers communicate with the App exclusively through Textual Messages — clean separation of concerns
- **Transcriber** and **LLMClient** are fully isolated behind L2 ports — `OllamaLLMClient` and `OpenAICompatLLMClient` are interchangeable; new implementations can be added with zero L2 changes
- **AudioSource** protocol: `SounddeviceAudioSource`, `CoreAudioTapSource`, `SoundCardLoopbackSource`, and `MixedAudioSource` are interchangeable behind a common interface; `DependencyContainer` always builds `MixedAudioSource` with platform-appropriate system backend
- **SessionController** (L3) owns all business state (DigestState, segments, latest_digest, user_context); App (L4) is thin compose + routing
- **DependencyContainer** (L4) is the composition root — inject fakes for testing
- **Template picker**: Interactive TUI launched before the main app; selects template to configure the session
- **Session picker**: Interactive TUI for `view` subcommand; lists saved session directories and loops back after each ViewApp exit
- **Session context**: User-editable text area in the digest column; included in digest prompts and persisted on final digest
- **Plugin system**: External packages register CLI subcommands via Python `entry_points` (group `lazy_take_notes.plugins`). Plugins import shared helpers from `lazy_take_notes.plugin_api` — this is the stable public surface

## Plugin System

Plugins are external packages that add subcommands to the CLI (e.g. `lazy-take-notes my-source <input>`).

### Contract

1. Declare an entry point in group `lazy_take_notes.plugins`:
```toml
# plugin's pyproject.toml
[project.entry-points."lazy_take_notes.plugins"]
my-source = "ltn_my_source:my_command"
```

2. The entry point must resolve to a `click.BaseCommand` (typically `@click.command`).

3. The entry point name becomes the subcommand name (e.g. `my-source` → `lazy-take-notes my-source`).

### Plugin API

Import from `lazy_take_notes.plugin_api` — not from internal modules.

```python
from lazy_take_notes.plugin_api import run_transcribe, TranscriptSegment

@click.command('my-source')
@click.argument('input_path')
@click.pass_context
def my_command(ctx, input_path):
    segments = my_parser(input_path)  # plugin-specific logic
    run_transcribe(ctx, subtitle_segments=segments, label='my session')
```

`run_transcribe(ctx, *, audio_path=None, subtitle_segments=None, label=None)` handles config loading, template picker, session directory, LLM preflight, dependency wiring, and TUI launch. Provide `audio_path` for whisper transcription or `subtitle_segments` for subtitle replay.

### Isolation

- Plugins that fail to load print a warning to stderr, never crash the CLI.
- Each plugin manages its own dependencies (add them to the plugin package, not to core).
