# Label speakers

This guide turns on speaker labels, so the transcript says who spoke each line: `S1`, `S2`, or a name you give. Labels are experimental.

The app needs the recording to find speakers, so `save_audio` must stay on (it is on by default).

## Install the speaker files

1. Run `take-note config`.
2. On the **Models** page, press `Download` next to `NeMo Speech runtime` and next to `Nemotron 3 diarization`. Together they are about 112 MB.

## Label a live session

1. In `take-note config`, open the **Audio** page and turn on **Label speakers**. Press `ctrl+s`.
2. Record as usual. About a minute in, a bar under the transcript shows each speaker and their speech time.
3. Name a speaker:
   - With the mouse: click the chip, then click it again.
   - With the keyboard (native build only): press `Tab` until the speaker bar has the focus. Choose the chip with `←` and `→`, and press `Enter`.
4. Type the name and press `Enter`. The name replaces `S1` everywhere in the transcript, and the notes use it.

A click on a chip, or a chip chosen with the arrows, highlights that speaker's lines.

## Label a saved session

Label a session that was recorded without speaker labels:

```bash
take-note diarize ~/Documents/LazyTakeNotes/2026-11-08_100000_standup
```

This writes `speakers.json` and `transcript_speakers.txt` into the session folder. Name the speakers afterwards in `take-note view`.

## Use your own build of the runtime

Set absolute paths in `config.yaml`:

```yaml
diarization:
  binary: /path/to/nemo-speech
  model: /path/to/Nemotron-3-Diarization.q8_0.gguf
```

Or export `LTN_NEMO_BINARY` and `LTN_NEMO_MODEL`. Leave both empty to use the files from Settings.

## See also

- [How speaker labels work](explain_speaker_labels.md): when a pass runs, and why `S1` is not the same person in two meetings.
- [Configuration](ref_configuration.md#diarization): the `diarization` keys.
