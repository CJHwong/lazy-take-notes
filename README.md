# lazy-take-notes

Terminal app for live transcription and note-taking. Records your mic, transcribes speech to text, and periodically generates structured digests of what's happening.

## Requirements

- Python 3.11+
- A microphone
- A transcription engine ([whisper.cpp](https://github.com/ggerganov/whisper.cpp) by default)
- An LLM backend ([Ollama](https://ollama.com) by default)

## Install

```bash
# with uv (recommended)
uv sync

# or pip
pip install -e .
```

## Run

```bash
ltn                                                     # start with defaults
ltn --config ~/.config/lazy-take-notes/config.yaml      # custom config
ltn --output-dir ./my_session                           # custom output dir
ltn --audio-file recording.m4a                          # batch-transcribe a file
```

## Keys

| Key     | Action                          |
| ------- | ------------------------------- |
| `Space` | Pause / resume recording        |
| `s`     | Stop recording                  |
| `c`     | Copy focused panel to clipboard |
| `Tab`   | Switch panel focus              |
| `h`     | Help                            |
| `q`     | Quit (runs final digest first)  |

Templates can add more keys for quick actions (catch up, action items, etc). Press `h` in the app to see all available bindings.

## Config

`~/.config/lazy-take-notes/config.yaml`:

```yaml
transcription:
  model: "large-v3-turbo-q8_0"    # default whisper model
  models:                         # per-locale overrides
    zh: "breeze-q8"               # Breeze ASR, optimized for Traditional Chinese
  chunk_duration: 10.0
digest:
  model: "gemma3:27b"      # heavy model for periodic digests
  min_lines: 15
  min_interval: 60
interactive:
  model: "gemma3:12b"      # fast model for quick actions
ollama:
  host: "http://localhost:11434"
template: "default_zh_tw"   # template file key (see TEMPLATES.md)
output:
  directory: "./output"
```

## Templates

Templates control the LLM prompts, labels, and quick-action keys for a session. The template picker launches at startup — built-ins are listed there.

To add your own or override a built-in, drop a `.yaml` file in `~/.config/lazy-take-notes/templates/`. See [TEMPLATES.md](TEMPLATES.md) for the full schema and variable reference.

## Output

After a session:

```
output/
├── transcript_raw.txt        # timestamped transcript
├── digest.json               # latest digest (machine-readable)
├── digest.md                 # latest digest (human-readable)
└── history/
    ├── digest_001.json
    ├── digest_002.json
    └── digest_003_final.json
```

## Development

```bash
uv sync                        # install deps
uv run pytest tests/ -v        # run tests
uv run lint-imports            # check layer contracts
```

Architecture details are in [AGENTS.md](AGENTS.md).

## License

MIT
