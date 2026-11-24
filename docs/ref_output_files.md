# Session files

Each session is a folder in `~/Documents/LazyTakeNotes`, or in the folder that `output.directory` or `--output-dir` sets. The folder name is the start time, plus the label if the session has one: `2026-11-08_100000_standup`.

```
2026-11-08_100000_standup/
├── transcript.txt            # the transcript, one timestamped line each
├── notes.md                  # the latest notes
├── context.txt               # the session context you typed
├── recording.wav             # the audio
├── speakers.json             # speaker turns and names
├── transcript_speakers.txt   # the transcript with speaker labels
├── debug.log                 # the debug log
├── .status.json              # the state of a headless session
└── history/                  # each version of the notes
    ├── notes_001.md
    ├── notes_002.md
    └── notes_003_final.md    # the final notes, after the stop
```

| File | When it is written |
|---|---|
| `transcript.txt`, `notes.md` | Always |
| `context.txt` | `save_context: true` and you typed a context |
| `recording.wav` | `save_audio: true`. Also kept with `save_audio: false` when transcription fails, because the audio is then the only full record |
| `recording_2.wav`, ... | The disk filled or the folder was removed during a recording. The recording goes on in the next file |
| `speakers.json`, `transcript_speakers.txt` | Speaker labels are on, or after `take-note diarize` |
| `debug.log` | `save_debug_log: true` |
| `.status.json` | A `--headless` session. `take-note status` reads it |
| `history/` | `save_notes_history: true` |

`recording.wav` is 16 kHz, mono, 16-bit PCM.

## Transcript lines

```
[00:00:04] the text of the line
```

The time counts from the start of the session. In `transcript_speakers.txt`, the speaker comes after the time:

```
[00:00:04] Andy: the text of the line
```
