# Plugin API

A plugin is a Python package. It imports only from `lazy_take_notes.plugin_api`. The native build runs plugins in a separate Python process, the plugin host, so the same plugin works in both versions. Some parts of the API work only in the Python version; the tables say which.

The plugin host runs on macOS and Linux. The Windows terminal app does not load plugins yet.

## Entry points

| Group | Value | Adds | Native build |
|---|---|---|---|
| `lazy_take_notes.plugins` | A `click` command | A subcommand. The entry point's name is the command name | Yes |
| `lazy_take_notes.llm_providers` | A callable `(InfraConfig) -> LLMClient` | An AI provider. Select it with `llm_provider: <name>` | Yes |
| `lazy_take_notes.transcription_providers` | A callable `(InfraConfig) -> TranscriptionBackend` | A speech-to-text backend. Select it with `transcription_provider: <name>` | No |

A plugin's own config goes under any key in `config.yaml`. Read it from `infra.model_extra`.

A plugin that fails to load prints one warning. It never stops the app.

## `run_transcribe`

```python
run_transcribe(
    ctx: click.Context,
    *,
    audio_path: Path | None = None,
    subtitle_segments: list[TranscriptSegment] | None = None,
    label: str | None = None,
    template_name: str | None = None,
    language: str | None = None,
    llm_client: LLMClient | None = None,     # Python version only
    transcriber: Transcriber | None = None,  # Python version only
) -> SessionArtifacts | None
```

Runs a full transcribe session: the template picker, the session folder, the AI check, the screen, and the final notes.

| Parameter | Meaning |
|---|---|
| `ctx` | The `click` context of your command |
| `audio_path` | An audio file to transcribe |
| `subtitle_segments` | Text you already have. No speech model runs. Wins over `audio_path` |
| `label` | The session label, in the folder name and the header |
| `template_name` | A template key. Skips the picker |
| `language` | A locale such as `en`. Uses its default template |

It returns `None` when the user cancels the template picker.

## `run_record`

Python version only. The native build raises `RuntimeError`.

```python
run_record(ctx, *, label=None, llm_client=None, transcriber=None,
           audio_source=None, template_name=None, language=None,
           mute_mic=False) -> SessionArtifacts | None
```

## Types

| Name | Fields or methods |
|---|---|
| `TranscriptSegment` | `text: str`, `wall_start: float`, `wall_end: float` (seconds from the session start) |
| `SessionArtifacts` | `session_dir: Path`, `notes_path: Path \| None` |
| `LLMClient` | `async chat(model, messages) -> ChatResponse`, `async chat_single(model, prompt) -> str`, `check_connectivity() -> (bool, str)`, `check_models(models) -> list[str]` (the missing ones) |
| `ChatMessage`, `ChatResponse` | The message and reply types of `LLMClient` |
| `InfraConfig` | The merged config. Plugin keys are in `model_extra` |
| `Transcriber`, `AudioSource`, `ModelResolver`, `TranscriptionBackend` | The speech-to-text types. Python version only |
