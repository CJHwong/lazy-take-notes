# Configuration

The app reads `config.yaml` from your config folder. `take-note config` edits it for you. `--config <file>` reads another file.

| System | Config folder |
|---|---|
| macOS | `~/Library/Application Support/lazy-take-notes/` |
| Linux | `~/.config/lazy-take-notes/` |
| Windows (Python version) | `C:\Users\<you>\AppData\Local\lazy-take-notes\` |

The same folder holds `templates/` (your templates) and `plugins.txt` (your plugins).

A key you leave out takes its default. Every default is shown below.

## AI provider

```yaml
llm_provider: ollama          # ollama, openai, or the name of a provider plugin

ollama:
  host: "http://localhost:11434"

openai:                       # any OpenAI-compatible API
  api_key: null               # null reads OPENAI_API_KEY from the environment
  base_url: "https://api.openai.com/v1"

digest:
  model: "gpt-oss:20b"        # makes the notes
interactive:
  model: "gpt-oss:20b"        # answers quick actions (keys 1 to 5)
```

## Transcription

```yaml
transcription:
  model: hf://ggerganov/whisper.cpp/ggml-large-v3-turbo-q8_0.bin
  models:                     # a model per language, by the template's locale
    zh: hf://alan314159/Breeze-ASR-25-whispercpp/ggml-model-q8_0.bin
    zh-min-nan: hf://phate334/Breeze-ASR-26-GGML/ggml-model-q8_0.bin
  input_device: null          # null: the system's default microphone
  chunk_duration: 25.0        # seconds of audio per transcription, at most
  overlap: 1.0                # seconds shared by two chunks
  silence_threshold: 0.01     # level below which audio counts as silence
  pause_duration: 1.5         # seconds of silence that end a chunk early

recognition_hints:            # words the speech model should expect, in every template
  - Kubernetes
  - JIRA
```

A model is a short name (such as `large-v3-turbo-q8_0` or `breeze25-q8`), an `hf://` URI, or a file path. The `zh` and `zh-min-nan` entries are built in. Set one to `null` to use `transcription.model` for that language.

## Notes

```yaml
digest:
  min_lines: 15               # new lines before the next notes update
  min_interval: 60.0          # seconds between two updates, at least
  max_lines: null             # an update runs at this many new lines, whatever the interval; null: twice min_lines
  compact_token_threshold: 100000   # past this prompt size, the conversation is compacted
```

## Session files

```yaml
output:
  directory: ~/Documents/LazyTakeNotes
  save_audio: true            # keep recording.wav; also needed for speaker labels
  save_notes_history: true    # keep each notes version in history/
  save_context: true          # keep the session context text in context.txt
  save_debug_log: false       # write debug.log
  auto_label: true            # with no --label, the AI names the session folder
```

With `save_audio: false`, the recording is still kept when transcription fails, because it is then the only full record. See [Session files](ref_output_files.md).

## Diarization

```yaml
diarization:
  enabled: false              # label speakers during a recording
  device: auto                # auto, cpu, or metal
  interval_s: 60              # seconds between two speaker passes
  binary: ""                  # empty: the runtime installed from Settings
  model: ""                   # empty: the model installed from Settings
```

## Appearance and templates

```yaml
theme: textual-dark
show_builtin_templates: true
```

## Environment variables

| Variable | Effect |
|---|---|
| `LTN_OUTPUT_DIR` | The session folder, as `--output-dir` |
| `LTN_PYTHON=1` | Run the Python version on a Mac with Apple silicon |
| `LTN_KIOSK=1` | Return to the mode picker after a session instead of exiting (the desktop app sets it) |
| `LTN_NEMO_BINARY`, `LTN_NEMO_MODEL` | Paths to your own speaker runtime and model |
| `LTN_PLUGIN_HOST_FROM` | Where the plugin host comes from: a local checkout, or another package spec |
| `OPENAI_API_KEY` | The API key, when `openai.api_key` is not set |
