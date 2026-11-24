# Write a plugin

This guide writes a plugin that adds a command to `take-note`. The command reads text or audio from your own source and opens it in the transcribe screen. The [plugin API](ref_plugin_api.md) lists everything a plugin can use.

## 1. Make the package

A plugin is two files.

`pyproject.toml`:

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

`src/ltn_my_source/__init__.py`:

```python
import click
from lazy_take_notes.plugin_api import TranscriptSegment, run_transcribe


@click.command("my-source")
@click.argument("input_path")
@click.pass_context
def my_command(ctx, input_path):
    """Transcribe from my custom source."""
    segments = parse_my_format(input_path)
    run_transcribe(ctx, subtitle_segments=segments, label="my session")


def parse_my_format(path):
    return [TranscriptSegment(text="Hello world", wall_start=0.0, wall_end=3.0)]
```

The entry point's name, `my-source`, becomes the command: `take-note my-source /path/to/input`.

## 2. Pick what you hand over

- You have text with times, such as subtitles: pass `subtitle_segments`. No speech model runs.
- You have audio: download or convert it to a local file, and pass `audio_path`. The app transcribes it.
- You have both sometimes: pass the segments when you have them, else the audio.

```python
if segments:
    run_transcribe(ctx, subtitle_segments=segments, label=title)
else:
    run_transcribe(ctx, audio_path=audio_path, label=title)
```

## 3. Test it

Replace `run_transcribe` in the test, so no screen opens:

```python
from unittest.mock import patch
from click.testing import CliRunner

from ltn_my_source import my_command


def test_my_command(tmp_path):
    with patch("ltn_my_source.run_transcribe") as run:
        result = CliRunner().invoke(
            my_command,
            [str(tmp_path / "input.txt")],
            obj={"config_path": None, "output_dir": None},
        )
    assert result.exit_code == 0
    assert run.call_args.kwargs["subtitle_segments"]
```

## 4. Install it

```bash
take-note plugin add "ltn-my-source @ git+https://github.com/you/ltn-my-source.git"
take-note my-source /path/to/input
```

`plugin add` writes the spec to `plugins.txt` in your config folder. The first start after an install takes longer, because uv installs the plugin's Python packages.

The native build reads `plugins.txt` itself. The Python version reads it only through the `take-note` command that setup.sh installs. With another install of the Python version, pass the plugin to uv:

```bash
uvx --from "lazy-take-notes @ git+https://github.com/CJHwong/lazy-take-notes.git" \
    --with "ltn-my-source @ git+https://github.com/you/ltn-my-source.git" \
    lazy-take-notes my-source /path/to/input
```
