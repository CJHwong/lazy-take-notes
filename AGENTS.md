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


## Testing Guidelines

* **Architecture**: The project follows **Clean Architecture** (L1 Entities -> L2 Use Cases -> L3 Adapters -> L4 Frameworks).
* **Mocking Strategy**:
* Do **NOT** use `@patch` on concrete libraries (e.g., `ollama`, `sounddevice`) in L2 tests.
* **MUST** use the provided protocol-conforming fakes located in `tests/conftest.py` (e.g., `FakeLLMClient`, `FakeTranscriber`).
* Library mocking is only permitted at the L3 (Adapter) boundary.


## Concurrency Model

```
Audio Worker (thread)          Digest Worker (async)        Query Worker (async)
  sounddevice + whisper          ollama heavy model           ollama fast model
        │                              │                           │
        │ TranscriptChunk              │ DigestReady               │ QueryResult
        │ AudioWorkerStatus            │ DigestError               │
        ▼                              ▼                           ▼
  ┌─────────────────────── App (event loop) ───────────────────────┐
  │ on_transcript_chunk → update panel, buffer lines, trigger      │
  │ on_digest_ready → update panel, persist to disk                │
  │ on_query_result → show modal                                   │
  │ Digest trigger: buffer >= min_lines AND elapsed >= min_interval│
  └────────────────────────────────────────────────────────────────┘
```

## Data Flow

1. **Audio capture**: sounddevice InputStream → numpy float32 buffer → VAD trigger → whisper transcribe → TranscriptSegment
2. **Transcript buffering**: App receives TranscriptChunk → updates panel, persists to disk, appends to DigestState.buffer
3. **Digest trigger**: When buffer size and time thresholds met → launch async digest worker
4. **Digest cycle**: Template-driven prompt → ollama.AsyncClient.chat → JSON parse → DigestData → persist + update panel
5. **Token compaction**: When prompt_tokens exceeds threshold, conversation history is compacted to 3 messages (system, compressed state, last response)
6. **Quick actions**: Keybinding → format prompt from template with current digest + recent transcript → ollama fast model → modal display

## Design Decisions

- **Thread worker for audio**: sounddevice and whisper.cpp are blocking C libraries, cannot run in asyncio
- **Async workers for LLM**: ollama.AsyncClient integrates naturally with Textual's event loop
- **Single-threaded state**: DigestState lives on the controller, only mutated on event loop — no locks needed
- **Template-driven**: All prompts, labels, and quick actions are defined in YAML templates — core logic is locale-agnostic
- **Message passing**: Workers communicate with the App exclusively through Textual Messages — clean separation of concerns
- **Transcriber** and **LLMClient** are fully isolated behind L2 ports — new implementations (FasterWhisper, OpenAI, etc.) can be added with zero L2 changes
- **SessionController** (L3) owns all business state (DigestState, segments, latest_digest); App (L4) is thin compose + routing
- **DependencyContainer** (L4) is the composition root — inject fakes for testing
