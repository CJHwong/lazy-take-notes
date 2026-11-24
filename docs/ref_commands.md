# Commands

`take-note` and `lazy-take-notes` are the same command. With no subcommand, it opens the mode picker.

## Options for every command

| Option | Effect |
|---|---|
| `-c`, `--config <file>` | Read this config file instead of the default |
| `-o`, `--output-dir <folder>` | Save sessions here. Also `LTN_OUTPUT_DIR` |
| `-V`, `--version` | Print the version |
| `-h`, `--help` | Print help |

## Sessions

| Command | What it does |
|---|---|
| `record` | Record the mic and system audio, transcribe them, and keep notes |
| `transcribe <file>` | Transcribe an audio or video file (mp3, wav, m4a, webm, aiff, flac, ogg and more), then write final notes. Needs ffmpeg |
| `view` | Browse saved sessions, read-only. You can still name speakers |

`record` and `transcribe` take these options:

| Option | Effect |
|---|---|
| `-l`, `--label <text>` | Add a label to the session folder name, for example `sprint-review` |
| `--template <key>` | Use this template and skip the picker, for example `lecture_notes_en` |
| `--language <locale>` | Use this language's default template: `en`, `zh-TW` or `zh-min-nan` |
| `--headless` | Run without the screen and write files only. It needs `--template` or `--language`. Follow it with `status` |
| `--mute-mic` | `record` only: start with the mic muted, so only system audio is recorded |

A headless recording stops on Ctrl-C or `kill -INT <pid>`, then writes the final notes. `kill -USR1 <pid>` mutes and unmutes the mic.

## Read saved sessions

`[SESSION]` is a session folder name or path. Without it, the command uses the newest session.

| Command | What it prints |
|---|---|
| `ls` | The sessions, newest first. `-a` lists all of them instead of the newest 20, and `--json` prints JSON for scripts |
| `status [SESSION]` | The state of a headless session |
| `transcript [SESSION]` | The transcript. A session with speaker labels prints the labeled one |
| `notes [SESSION]` | The notes |

## Speakers

| Command | What it does |
|---|---|
| `diarize <SESSION_DIR>` | Label the speakers of a saved session. Writes `speakers.json` and `transcript_speakers.txt` |

`diarize` takes `--device auto|cpu|metal` (default `auto`) and `--timeout <seconds>` (default 300).

## Setup

| Command | What it does |
|---|---|
| `config` | Open the [settings screen](ref_settings.md) |
| `create-template` | Build a template with AI. See [Create and edit templates](howto_templates.md) |
| `plugin add <spec>` | Add a plugin, for example `'ltn-youtube @ git+https://...'` |
| `plugin remove <name>` | Remove a plugin by its package name |
| `plugin list` | List the plugins |

A plugin can add its own commands. See [Write a plugin](howto_plugins.md).
