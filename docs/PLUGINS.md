# Plugin Development Guide

lazy-take-notes supports source plugins — external packages that add subcommands to the CLI. A plugin fetches or transforms content from any source and hands it off to the standard transcription TUI.

## Quick start

A minimal plugin is two files: `pyproject.toml` and a Python module with a Click command.

### pyproject.toml

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "ltn-my-source"
version = "0.1.0"
dependencies = [
    "lazy-take-notes @ git+https://github.com/CJHwong/lazy-take-notes.git",
    "click>=8.0",
]

[project.entry-points."lazy_take_notes.plugins"]
my-source = "ltn_my_source:my_command"

[tool.hatch.metadata]
allow-direct-references = true

[tool.hatch.build.targets.wheel]
packages = ["src/ltn_my_source"]
```

### src/ltn_my_source/\_\_init\_\_.py

```python
import click
from lazy_take_notes.plugin_api import run_transcribe, TranscriptSegment

@click.command("my-source")
@click.argument("input_path")
@click.pass_context
def my_command(ctx, input_path):
    """Transcribe from my custom source."""
    segments = parse_my_format(input_path)
    run_transcribe(ctx, subtitle_segments=segments, label="my session")

def parse_my_format(path):
    # your logic here — return a list of TranscriptSegment
    return [
        TranscriptSegment(text="Hello world", wall_start=0.0, wall_end=3.0),
    ]
```

That's it. Install the plugin and the command appears:

```bash
lazy-take-notes my-source /path/to/input
```

## How it works

1. Your package declares an entry point in the `lazy_take_notes.plugins` group
2. At startup, the CLI discovers all installed plugins and registers them as subcommands
3. Your Click command runs, does its source-specific work, then calls `run_transcribe`
4. `run_transcribe` handles everything else: config loading, template picker, session directory, LLM preflight, dependency wiring, and TUI launch

## API reference

### `run_transcribe`

```python
run_transcribe(
    ctx: click.Context,
    *,
    audio_path: Path | None = None,
    subtitle_segments: list[TranscriptSegment] | None = None,
    label: str | None = None,
) -> None
```

| Parameter | Description |
|-----------|-------------|
| `ctx` | Click context — passed through from your `@click.pass_context` command |
| `audio_path` | Path to an audio file for whisper transcription |
| `subtitle_segments` | Pre-parsed segments for subtitle replay (no whisper needed) |
| `label` | Session label (appears in the output directory name and TUI header) |

Provide `audio_path` for audio that needs speech-to-text, or `subtitle_segments` for pre-parsed text. If both are provided, subtitle replay is used.

### `TranscriptSegment`

```python
TranscriptSegment(
    text: str,           # the transcribed text
    wall_start: float,   # start time in seconds from session start
    wall_end: float,     # end time in seconds from session start
)
```

## Examples

### Subtitle source (pre-parsed text, no whisper)

Your plugin parses subtitles, an SRT file, a VTT file, or any text format into `TranscriptSegment`s and passes them directly. No audio processing needed.

```python
run_transcribe(ctx, subtitle_segments=segments, label="my video")
```

### Audio source (needs whisper transcription)

Your plugin downloads or converts audio into a local file and passes the path. The standard whisper pipeline handles the rest.

```python
run_transcribe(ctx, audio_path=Path("/tmp/downloaded.wav"), label="my recording")
```

### Mixed source (subtitle preferred, audio fallback)

Try subtitles first; fall back to audio download if unavailable.

```python
if subtitle_segments:
    run_transcribe(ctx, subtitle_segments=subtitle_segments, label=title)
else:
    run_transcribe(ctx, audio_path=audio_path, label=title)
```

## Testing your plugin

Mock `run_transcribe` in tests to verify your plugin passes the right arguments without launching the TUI:

```python
from unittest.mock import patch
from click.testing import CliRunner

def test_my_command(tmp_path):
    runner = CliRunner()
    with patch("ltn_my_source.run_transcribe") as mock_run:
        result = runner.invoke(my_command, [str(tmp_path / "input.txt")],
                               obj={"config_path": None, "output_dir": None})

    assert result.exit_code == 0
    mock_run.assert_called_once()
    assert mock_run.call_args[1]["subtitle_segments"] is not None
```

## Distribution

```bash
# Users install alongside lazy-take-notes
uv tool install lazy-take-notes --with ltn-my-source

# Or run directly from GitHub (no install)
uvx --from "lazy-take-notes @ git+https://github.com/CJHwong/lazy-take-notes.git" \
    --with "ltn-my-source @ git+https://github.com/you/ltn-my-source.git" \
    lazy-take-notes my-source /path/to/input
```
