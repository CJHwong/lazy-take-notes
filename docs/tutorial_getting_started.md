# Getting started

In this tutorial you install lazy-take-notes, connect it to an AI model, and record your first session. It takes about 15 minutes, most of it for downloads. You need no experience with Ollama, OpenAI or terminal tools.

You need a Mac or a Linux computer with a microphone.

## 1. Install the app

Run this in a terminal:

```bash
curl -fsSL https://raw.githubusercontent.com/CJHwong/lazy-take-notes/main/setup.sh | bash
```

The script first warns that it runs downloaded code as you. Press Enter to continue.

It then does five things:

1. It installs [uv](https://docs.astral.sh/uv/), which brings everything else the app needs, Python included.
2. It asks which AI provider you want: Ollama or an OpenAI-compatible API.
3. It sets up that provider.
4. It adds the `take-note` command.
5. It writes your first `config.yaml`.

If you picked Ollama, continue with step 2. If you picked an OpenAI-compatible API, go to step 3.

## 2. Get the Ollama model

Ollama runs the AI model on your machine, so nothing leaves your computer.

1. Start Ollama, if it does not run yet:

   ```bash
   ollama serve
   ```

2. Download the default model, about 14 GB:

   ```bash
   ollama pull gpt-oss:20b
   ```

3. Check that it answers:

   ```bash
   ollama run gpt-oss:20b "Say hello"
   ```

Go to step 4.

## 3. Set your API key

If you use OpenAI, Groq, Together, Gemini or another OpenAI-compatible API:

1. Get an API key from your provider.
2. Put it in your shell profile (`~/.zshrc` or `~/.bashrc`), so it stays out of config files:

   ```bash
   export OPENAI_API_KEY="sk-your-key-here"
   ```

3. Open a new terminal, so the key is set.
4. Run `take-note config`. On the **Models** page, set the provider to `openai`, enter the base URL of your provider, and pick the models. Press `ctrl+s` to save.

## 4. Record your first session

1. Run:

   ```bash
   take-note record
   ```

2. Pick a template with the arrow keys and Enter. For an English meeting, pick `default_en`.
3. The app checks your setup: the output folder, the microphone, system audio, the speech model, and the AI model. The first run downloads the speech model, so this step takes a few minutes once.
4. Talk for a few minutes. Your words appear in the transcript on the left. After 15 lines and at least a minute, the notes on the right fill in. Press `d` to make notes at once.
5. Press `s` to stop the recording.
6. Press `q`. The app writes the final notes and closes.

macOS asks for permission to use the microphone. Allow it. To record the other side of a call, also turn on system audio by hand: see "No system audio on macOS" below.

## 5. Find your notes

Each session is a folder in `~/Documents/LazyTakeNotes`. Print the notes of the newest session:

```bash
take-note notes
```

Browse all sessions in the app:

```bash
take-note view
```

## Next steps

- Transcribe a file you already have: `take-note transcribe meeting.m4a`. This needs [ffmpeg](https://ffmpeg.org) (`brew install ffmpeg`, or `apt install ffmpeg`).
- [Label who said what](howto_speaker_labels.md).
- [Make a template](howto_templates.md) for your own kind of meeting.
- Press `h` during a recording to see every key.

## If something goes wrong

**"LLM provider not reachable"**

- Ollama on your machine: check that `ollama serve` runs. `ollama list` must answer.
- Ollama cloud models: sign in with `ollama signin`.
- OpenAI-compatible API: check that `echo $OPENAI_API_KEY` prints your key.

**"Model not found"**

- Ollama: run `ollama pull <model-name>`.
- OpenAI-compatible API: check the model names in `take-note config` against your provider's list.

**Nothing happens when I speak**

- Check that the recording is not paused. `Space` resumes it.
- Speak closer to the mic. The app ignores sound that is too quiet to be speech.

**"No input audio devices found"**

- Check that your microphone is connected and not muted.
- On macOS, allow your terminal app in **System Settings > Privacy & Security > Microphone**.
- On Linux, check that PulseAudio or PipeWire runs: `pactl info` must answer.

**No system audio on macOS**

Open **System Settings > Privacy & Security > Screen & System Audio Recording**. Turn on **System Audio Recording Only** for your terminal app. Without the permission, macOS sends silence and reports no error. The app warns you when no system audio arrives.

**The notes are slow, or not good enough**

The default model needs about 14 GB of memory. Try a smaller Ollama model, or a cloud model: sign in with `ollama signin`, then set `gpt-oss:120b-cloud` as the digest model in `take-note config`.
