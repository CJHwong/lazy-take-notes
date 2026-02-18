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
ltn --template path/to/template.yaml                    # custom template
ltn --output-dir ./my_session                           # custom output dir
ltn --list-templates                                    # show available templates
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
template: "default_zh_tw"
output:
  directory: "./output"
```

## Templates

Templates are YAML files that control prompts, labels, and quick actions. Run `ltn --list-templates` to see what's available.

### Creating your own

Drop a YAML file in `~/.config/lazy-take-notes/templates/` and it's auto-discovered by name — no full path needed. User templates override built-ins of the same name.

```yaml
# ~/.config/lazy-take-notes/templates/ux_interview.yaml
metadata:
  name: "ux_interview"
  description: "User research interview analyzer"
  locale: "en"

system_prompt: |
  You are a UX researcher's assistant. You receive transcript segments from a
  user research interview and extract structured insights. Track the
  participant's exact words — direct quotes are gold. Distinguish between
  observed behavior, stated preferences, and emotional reactions.
  Return Markdown with sections:
  ## Participant Profile, ## Pain Points, ## Feature Requests,
  ## Direct Quotes, ## Behavioral Observations.

digest_user_template: |
  New transcript ({line_count} lines):
  {new_lines}
  {user_context}
  Please update the research notes.

final_user_template: |
  Interview is over. Final transcript ({line_count} lines):
  {new_lines}
  Full transcript:
  {full_transcript}
  Produce the final research notes.

quick_actions:
  - key: "1"
    label: "Pain Points"
    description: "List all pain points expressed so far"
    prompt_template: |
      Current research notes:
      {digest_markdown}
      Recent transcript:
      {recent_transcript}
      List every pain point the participant has expressed, with direct quotes.
  - key: "2"
    label: "Sentiment"
    description: "Analyze participant sentiment and engagement"
    prompt_template: |
      Current research notes:
      {digest_markdown}
      Recent transcript:
      {recent_transcript}
      How is the participant feeling about the product? Note shifts in tone.
```

```bash
ltn --list-templates            # your template shows up with [user] tag
ltn --template ux_interview     # use it by name
```

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
