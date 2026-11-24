# Settings screen

`take-note config` opens the settings screen. The mode picker also opens it. The screen edits `config.yaml`; see [Configuration](ref_configuration.md) for each key.

| Page | Sections |
|---|---|
| Models | Text generation (provider, Ollama host or OpenAI base URL and key, digest and quick-action models), Speech-to-Text, Language models, Speaker assets, Advanced |
| Digest | When to summarize (minimum lines, minimum interval, maximum lines), Advanced (token compaction) |
| Audio | Input check (a live level test), Microphone, Audio processing, Speaker labels |
| Sessions | Session storage (folder and what to save), Custom vocabulary, Templates |
| Appearance | Theme |

| Key | Action |
|---|---|
| `ctrl+s` | Save |
| `Tab`, `shift+tab` | Next and previous field |
| `Esc` in a page | Back to the page list |
| `Esc` on the page list | Leave Settings. The app asks first if a change is not saved |

Settings writes `config.yaml` readable by your user only, because it can hold an API key.

The **Models** page also installs and removes models. A download resumes after a break, shows its speed and time left, and can be cancelled. See [The model cache](explain_model_cache.md).
