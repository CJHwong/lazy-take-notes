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
uv run ltn
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

* **Architecture**: The project follows **Clean Architecture** (L1 Entities -> L2 Use Cases -> L3 Adapters -> L4 Frameworks).
* **Mocking Strategy**:
* Do **NOT** use `@patch` on concrete libraries (e.g., `ollama`, `sounddevice`) in L2 tests.
* **MUST** use the provided protocol-conforming fakes located in `tests/conftest.py` (e.g., `FakeLLMClient`, `FakeTranscriber`).
* Library mocking is only permitted at the L3 (Adapter) boundary.


## Concurrency Model

```
Audio Worker (thread)          Digest Task (async)          Query Task (async)
  AudioSource + whisper          ollama heavy model           ollama fast model
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

## Audio Modes

Selected at startup via template picker (all platforms):

- **MIC_ONLY** — `SounddeviceAudioSource` (PortAudio, cross-platform)
- **SYSTEM_ONLY** — macOS: `CoreAudioTapSource` (ScreenCaptureKit); Linux/Windows: `SoundCardLoopbackSource` (PulseAudio/WASAPI loopback)
- **MIX** — `MixedAudioSource` (mic + system blended, 0.5 attenuation anti-clipping)

`DependencyContainer._build_audio_source()` selects the system audio backend by platform.

## Data Flow

1. **Audio capture**: AudioSource → numpy float32 buffer → VAD trigger → whisper transcribe (off-thread) → TranscriptSegment
2. **Transcript buffering**: App receives TranscriptChunk → updates panel, persists to disk, appends to DigestState.buffer
3. **Digest trigger**: When buffer >= min_lines AND elapsed >= min_interval, or buffer >= max_lines (force) → launch async digest task
4. **Digest cycle**: Template-driven prompt (with user session context) → ollama.AsyncClient.chat → JSON parse → DigestData → persist + update panel
5. **Token compaction**: When prompt_tokens exceeds threshold, conversation history is compacted to 3 messages (system, compressed state, last response)
6. **Quick actions**: Positional keybinding (1–5) → format prompt from template with current digest + recent transcript → ollama fast model → modal display
7. **Audio file batch mode**: CLI `--audio-file PATH` → ffmpeg decode → chunked transcription → single final digest (headless, no TUI)
8. **Recording**: When `save_audio: true`, WAV is written alongside output — mic mode records at native sample rate, system/mixed mode records processed 16 kHz int16

## Design Decisions

- **Thread worker for audio**: sounddevice and whisper.cpp are blocking C libraries, cannot run in asyncio. Transcription runs in a `ThreadPoolExecutor` within the audio worker thread.
- **Async tasks for LLM**: ollama.AsyncClient integrates naturally with Textual's event loop. Digest and query are spawned on-demand with `exclusive=True` to prevent overlapping calls.
- **Single-threaded state**: DigestState lives on the controller, only mutated on event loop — no locks needed
- **Template-driven**: All prompts, labels, and quick actions are defined in YAML templates — core logic is locale-agnostic
- **Message passing**: Workers communicate with the App exclusively through Textual Messages — clean separation of concerns
- **Transcriber** and **LLMClient** are fully isolated behind L2 ports — new implementations (FasterWhisper, OpenAI, etc.) can be added with zero L2 changes
- **AudioSource** protocol: `SounddeviceAudioSource`, `CoreAudioTapSource`, `SoundCardLoopbackSource`, and `MixedAudioSource` are interchangeable behind a common interface; `DependencyContainer` selects per platform + audio mode
- **SessionController** (L3) owns all business state (DigestState, segments, latest_digest, user_context); App (L4) is thin compose + routing
- **DependencyContainer** (L4) is the composition root — inject fakes for testing
- **Template picker**: Interactive TUI launched before the main app; selects template + audio mode to configure the session
- **Session context**: User-editable text area in the digest column; included in digest prompts and persisted on final digest
