<p align="center">
  <img src="docs/logo-banner.png" alt="lazy-take-notes" width="360">
</p>

Terminal app for live transcription and note-taking. Records your mic, transcribes speech to text, and periodically generates structured digests of what's happening.

![screenshot](docs/screenshot.png)

## Requirements

- Python 3.11+
- A microphone
- A transcription engine ([whisper.cpp](https://github.com/ggerganov/whisper.cpp) by default)
- An LLM backend ([Ollama](https://ollama.com) by default, or any [OpenAI-compatible API](https://platform.openai.com/))

## Install

### Quick setup (macOS)

Run this in Terminal — it installs everything and sets up a `take-note` shortcut:

```bash
curl -fsSL https://raw.githubusercontent.com/CJHwong/lazy-take-notes/main/setup.sh | bash
```

Then run `take-note record`.

### Manual install

```bash
# try without installing (uv required)
uvx --from git+https://github.com/CJHwong/lazy-take-notes.git lazy-take-notes

# or clone and install locally
uv sync

# or pip
pip install -e .
```

**New to this?** See the [Getting Started guide](docs/GETTING_STARTED.md) for step-by-step setup instructions.

## Run

```bash
take-note                                     # interactive mode selector
take-note record                              # start recording
take-note record -l sprint-review             # record with session label
take-note transcribe recording.m4a            # transcribe an audio file
take-note view                                # browse saved sessions
take-note config                              # open the settings editor
take-note create-template                     # build a custom template with AI
take-note --config path/to/config.yaml        # custom config
take-note --output-dir ./my_session           # custom output dir
```

> `lazy-take-notes` works as an alias for `take-note`.

## Keys

| Key     | Action                          |
| ------- | ------------------------------- |
| `Space` | Pause / resume recording        |
| `s`     | Stop recording                  |
| `c`     | Copy focused panel to clipboard |
| `Tab`   | Switch panel focus              |
| `h`     | Help                            |
| `q`     | Quit                            |

Templates can add more keys for quick actions (catch up, action items, etc). Press `h` in the app to see all available bindings.

## Config

Config lives in your OS config directory:

| OS      | Path                                             |
| ------- | ------------------------------------------------ |
| macOS   | `~/Library/Application Support/lazy-take-notes/` |
| Linux   | `~/.config/lazy-take-notes/`                     |
| Windows | `C:\Users\<you>\AppData\Local\lazy-take-notes\`  |

Example `config.yaml`:

```yaml
# LLM provider: 'ollama' (default) or 'openai' (any OpenAI-compatible API)
llm_provider: ollama

ollama:
  host: "http://localhost:11434"

# OpenAI-compatible provider (OpenAI, Gemini, Groq, Together, vLLM, etc.)
# openai:
#   api_key: sk-...               # or set OPENAI_API_KEY env var
#   base_url: "https://api.openai.com/v1"

recognition_hints:                  # global hints for the speech recogniser
  - "Kubernetes"                    # applied to every template (merged with per-template hints)
  - "JIRA"

transcription:
  model: "large-v3-turbo-q8_0"    # default whisper model
  models:                         # per-locale overrides
    zh: "breeze-q8"               # Breeze ASR, optimized for Traditional Chinese
  chunk_duration: 25.0
  overlap: 1.0
  silence_threshold: 0.01
  pause_duration: 1.5
digest:
  model: "gpt-oss:20b"            # heavy model for periodic digests
  min_lines: 15
  min_interval: 60
  compact_token_threshold: 100000
interactive:
  model: "gpt-oss:20b"            # fast model for quick actions
output:
  directory: "./output"
  save_audio: true                # save recording.wav alongside transcript
  save_notes_history: true        # keep numbered snapshots in history/
  save_context: true              # save session context text
  save_debug_log: false           # write debug.log (off by default)
```

## Templates

Templates control the LLM prompts, labels, and quick-action keys for a session. The template picker launches at startup — built-ins are listed there.

To add your own or override a built-in, drop a `.yaml` file in the `templates/` subdirectory of your config path (see table above). See [docs/TEMPLATES.md](docs/TEMPLATES.md) for the full schema and variable reference.

## Output

After a session:

```
output/
├── transcript.txt            # timestamped transcript
├── notes.md                  # latest notes/digest (markdown)
├── context.txt               # user-provided context (when save_context: true)
├── recording.wav             # audio recording (when save_audio: true)
├── debug.log                 # debug log (when save_debug_log: true)
└── history/                  # numbered snapshots (when save_notes_history: true)
    ├── notes_001.md
    ├── notes_002.md
    └── notes_003_final.md    # final digest on quit/stop
```

## Development

```bash
uv sync                  # install deps
uv run pytest tests/ -v  # run tests
uv run lint-imports      # check layer contracts
```

Architecture details are in [AGENTS.md](AGENTS.md). To build your own source plugin, see the [Plugin Development Guide](docs/PLUGINS.md).

## License

MIT
