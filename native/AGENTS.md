# Native Binaries — Agent Context

This directory contains native Swift helper binaries that are compiled once and committed.
Python gateways in `l3_interface_adapters/gateways/` invoke them as subprocesses.

## coreaudio-tap

### What it does

Captures system audio output (all apps on the main display) via ScreenCaptureKit and
writes raw **float32 mono PCM at 16 kHz** to stdout. The Python side
(`CoreAudioTapSource`) reads it in a background thread and feeds it to the transcriber.

### Build

Requires Xcode Command Line Tools and Swift toolchain (macOS only).

```bash
bash scripts/build_native.sh
```

This compiles for both arm64 and x86_64 and links them into a universal binary at
`src/lazy_take_notes/_native/bin/coreaudio-tap`.

After building for the first time, ensure the executable bit is tracked by git:

```bash
git update-index --chmod=+x src/lazy_take_notes/_native/bin/coreaudio-tap
```

### Manual smoke test

These tests must be run on a real macOS machine — a sandboxed shell or CI environment
will not have Screen Recording permission and the binary will exit immediately.

**Step 1 — Check TCC permission is granted**

The binary requires Screen Recording permission for the *terminal app* that launches it
(e.g. Terminal.app or iTerm2). Grant it in:

> System Settings → Privacy & Security → Screen Recording

Add the terminal app to the list, then restart the terminal. You only need to do this once.

**Step 2 — Verify the binary starts and prints the format line**

```bash
timeout 3 src/lazy_take_notes/_native/bin/coreaudio-tap 2>&1 >/dev/null | head -2
```

Expected stderr output (format line printed on first audio callback):

```
coreaudio-tap: 48000 Hz 2ch → 16000 Hz 1ch
```

If you see this, the stream is up and SCKit is delivering audio callbacks.

If the binary exits immediately with no output, Screen Recording permission is missing —
see Step 1.

**Step 3 — Verify bytes are flowing**

Play audio from a GUI application (Zoom, Safari, Music, QuickTime — anything with a
window on the main display). CLI tools like `say` or `afplay` have no windows and are
**not** captured by ScreenCaptureKit.

```bash
# Play something audible in a GUI app, then:
src/lazy_take_notes/_native/bin/coreaudio-tap > /tmp/ltn_pcm.bin &
TAP_PID=$!
sleep 3
kill $TAP_PID

# Check bytes written (should be > 0)
wc -c /tmp/ltn_pcm.bin

# Decode and inspect (requires Python + numpy):
python3 -c "
import numpy as np, sys
arr = np.fromfile('/tmp/ltn_pcm.bin', dtype=np.float32)
print(f'samples: {len(arr)}, duration: {len(arr)/16000:.1f}s, peak: {arr.max():.4f}')
"
```

Expected: `duration: ~3.0s`, `peak > 0.001` when audio is playing.

**Step 4 — Verify error detection (no permission)**

If you want to test the permission-denied path, temporarily remove Screen Recording
permission from your terminal in System Settings, then run:

```bash
src/lazy_take_notes/_native/bin/coreaudio-tap 2>&1 | head -5
```

Expected: error message about Screen Recording + `exit(1)`.

### Key architectural lessons

**ScreenCaptureKit, not Core Audio Taps**

The original plan used `CATapDescription` + `AudioHardwareCreateProcessTap`. This was
abandoned because:
- `kTCCServiceScreenCapture` TCC permission is required for the tap to deliver audio.
- From an unsigned CLI binary (no app bundle, no entitlements), TCC blocks audio delivery
  *silently* — no error, no callback, zero bytes. There is no API to query why.
- SCKit has a proper permission flow (`SCShareableContent.getExcludingDesktopWindows`)
  that either succeeds or returns a clear error. It also works from unsigned CLI binaries
  as long as the *host terminal* has been granted Screen Recording in System Settings.

**Do NOT set `sampleRate` or `channelCount` on `SCStreamConfiguration`**

Setting a non-standard `sampleRate` (e.g. 16000) on `SCStreamConfiguration` silently
disables audio delivery. Always capture at the system native format (48 kHz stereo) and
resample in the audio callback via `AVAudioConverter`.

**A `.screen` output handler must be registered for audio callbacks to fire**

SCKit's audio delivery is internally tied to video frame ticks. If only a `.audio`
handler is registered, audio callbacks never fire. Always register both:

```swift
captureStream.addStreamOutput(handler, type: .audio, sampleHandlerQueue: audioQueue)
captureStream.addStreamOutput(handler, type: .screen, sampleHandlerQueue: audioQueue)
```

The `.screen` callback can be a no-op (ignore video frames via `guard type == .audio`).

**`minimumFrameInterval = CMTime.positiveInfinity` kills audio**

Using `positiveInfinity` to suppress video entirely also suppresses audio callbacks.
Use a real low frame rate instead: `CMTime(value: 1, timescale: 10)` = 10 fps.

**SCKit only captures audio from GUI apps with windows**

`capturesAudio = true` on SCKit delivers audio from apps that have visible windows on the
captured display. CLI programs (`say`, `afplay`, `ffplay`) with no windows are **not**
captured. For the target use case (Zoom calls), Zoom has windows so it is captured.

**Use `CMSampleBufferCopyPCMDataIntoAudioBufferList` for PCM extraction**

Do NOT manually index into `CMBlockBuffer` or try to copy `AudioBufferList` by value.
The system delivers non-interleaved stereo (2 separate channel buffers). The correct API
to extract PCM into an `AVAudioPCMBuffer` for the converter is:

```swift
CMSampleBufferCopyPCMDataIntoAudioBufferList(
    sampleBuffer, at: 0, frameCount: Int32(numFrames),
    into: srcBuf.mutableAudioBufferList)
```

**Stdin-close detection for clean subprocess teardown**

The binary watches stdin for EOF via `DispatchSource.makeReadSource`. When the Python
parent closes `proc.stdin` (or crashes), stdin becomes readable with 0 bytes, and the
binary sets `shouldStop = true` and exits cleanly. This prevents zombie processes.

### Python integration

The Python gateway is at:
`src/lazy_take_notes/l3_interface_adapters/gateways/coreaudio_tap_source.py`

It launches the binary via `subprocess.Popen` with `stdout=PIPE, stderr=PIPE, stdin=PIPE`.
A background reader thread pumps stdout bytes (raw float32) into a `queue.Queue`.
If the process exits with a non-zero code, the error is surfaced on the next `read()` call.

Tests (all mocked — no real binary invoked):
`tests/l3_interface_adapters/test_coreaudio_tap_source.py`
