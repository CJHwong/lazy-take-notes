# The model cache

The speech models, the speaker-label runtime and model, and the native app itself are large downloads. The app keeps them in one cache, shared by both versions:

| System | Cache folder |
|---|---|
| macOS | `~/Library/Caches/lazy-take-notes` |
| Linux | `~/.cache/lazy-take-notes` |

## Every file is pinned

The app knows each file it may download in advance: its address, its size, and its SHA-256. It checks every download against those values, and it never runs a runtime that failed the check. A version never changes behind your back: a new model or runtime comes only with a new release of the app.

## Downloads survive interruptions

A download resumes where it stopped, after a network break or a cancel. The partial file stays in the cache until the download finishes.

Two downloads of one file never run at the same time, even from two windows. The second one waits, then uses the first one's file. The Models page shows a file that another window downloads as Downloading.

## Models from older versions

Older versions kept whisper models in the `pywhispercpp` cache, and speaker files in `~/.cache/nemotron-diarization`. On first use, the app links or copies those files into the cache and leaves the originals where they are.

When you remove such a model in Settings, the dialog says how much space the original uses. It deletes the original only if you tick the box for it. A removed model stays removed until you press `Download` again, which reuses the original if it still exists.
