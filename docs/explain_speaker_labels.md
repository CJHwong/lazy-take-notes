# How speaker labels work

Speaker labels come from [Nemotron 3 Diarization](https://github.com/NVIDIA/NeMo-Speech.cpp), run by NVIDIA's NeMo-Speech.cpp runtime on your machine. Nothing is sent anywhere.

## A pass reads the whole recording

The runtime does not listen live. It reads a finished audio file and returns speaker turns: who spoke from when to when. So during a recording, the app runs it again and again on a copy of `recording.wav` so far. One run is a pass.

Each pass reads the whole recording, not only the new part. A speaker who comes back after 20 minutes then gets the same number as before. The cost grows with the length of the meeting.

## When a pass runs

| Moment | Pass | Version |
|---|---|---|
| Every `diarization.interval_s` (60 s), if new audio arrived | Yes | Both |
| You pause the recording | Yes, at once | Native only |
| You mute the mic | No: the recording goes on | Both |
| You stop the recording | Yes, the last one, before the final notes | Native only |
| A `--headless` recording ends | Yes, before the final notes | Native only |

Between two passes, a new transcript line gets a label at once only if it is close to a turn that a pass already found. Other new lines get their label at the next pass.

## One label per line

The model can find two people talking at the same time. A transcript line still gets one speaker: the one who talked the longest during that line. The speech model writes mixed speech as one line anyway, so a split would be a guess.

## Labels belong to one session

`S1` is "the first voice heard in this session". In the next meeting, `S1` is whoever speaks first there. The names you give are saved in that session's `speakers.json` only.

Numbers follow the order in which voices first appear. A label already on screen keeps its meaning when a new voice joins.

## Speed

Diarization is fast compared with the meeting. 20 minutes of audio took 3.7 s on Metal, and 61 minutes took 44 s on a CPU.
