# Native Binaries — Agent Context

This directory contains native Swift helper binaries that are compiled once and committed.
Python gateways in `l3_interface_adapters/gateways/` invoke them as subprocesses.

## coreaudio-tap

### What it does

Captures system audio output via a Core Audio process tap (`AudioHardwareCreateProcessTap`,
macOS 14.2+) and writes raw **float32 mono PCM at 16 kHz** to stdout. The Python side
(`CoreAudioTapSource`) reads it in a background thread and feeds it to the transcriber.

The tap reads the output mixer directly, so it captures every process, not only apps with
a window on screen. `afplay`, `say`, and `ffplay` are captured.

### Build

Requires Xcode Command Line Tools and Swift toolchain (macOS only).

```bash
bash scripts/build_native.sh
```

This compiles arm64 and x86_64 and links them into a universal binary at
`src/lazy_take_notes/_native/bin/coreaudio-tap`.

The script asks SwiftPM for the product path with `--show-bin-path` and copies each arch
before the next build. Do not hardcode `.build/<arch>-apple-macosx/release`: newer
toolchains share one product directory across `--arch` builds, and lipo then links the
same stale file twice while exiting 0.

After building for the first time, ensure the executable bit is tracked by git:

```bash
git update-index --chmod=+x src/lazy_take_notes/_native/bin/coreaudio-tap
```

### Manual smoke test

These tests must run on a real macOS 14.2+ machine. A sandboxed shell or CI environment
has no audio and no TCC grant.

**Step 1 — Permission**

The tap needs the system audio capture grant. macOS 15 and newer list it under:

> System Settings → Privacy & Security → Screen & System Audio Recording

Grant it to the terminal that launches the binary, then restart the terminal. If the
grant is missing, tap creation fails with an OSStatus on stderr.

**Step 2 — Start and check the format line**

```bash
timeout 3 src/lazy_take_notes/_native/bin/coreaudio-tap 2>&1 >/dev/null | head -3
```

Expected:

```
coreaudio-tap: 48000 Hz 1ch -> 16000 Hz 1ch
coreaudio-tap: capturing
```

**Step 3 — Verify bytes are flowing**

```bash
afplay /System/Library/Sounds/Glass.aiff &
src/lazy_take_notes/_native/bin/coreaudio-tap > /tmp/ltn_pcm.bin 2>/dev/null &
TAP_PID=$!
sleep 3
kill $TAP_PID

python3 -c "
import numpy as np
a = np.fromfile('/tmp/ltn_pcm.bin', dtype=np.float32)
print(f'samples: {len(a)}, duration: {len(a)/16000:.1f}s, peak: {a.max():.4f}')
"
```

Expected: around 3 s of audio at 16 kHz, peak above 0.001 while the sound plays.

### Key architectural lessons

**A process tap, not ScreenCaptureKit**

ScreenCaptureKit paces audio callbacks off the captured display's video frame clock.
That causes two problems, both measured on macOS 26.6.2:

- Delivery rate follows the frame rate. A 30 fps video window gave 50 callbacks per
  second; a static screen gave 4. Audio arrives late when nothing moves on screen.
- After a CoreAudio overload, SCKit keeps the stream alive and callbacks firing while
  filling every buffer with zeros. It reports no error, so the only evidence is silence.

A process tap reads the output mixer directly. Measured on the same machine:

| Check | Result |
| --- | --- |
| 8 s run with audio | 8.4 s of frames captured |
| 20 s run with two output-device switches | 21.0 s of frames, 100-500 ms gap per switch, immediate recovery |
| 12 s run through the Python gateway | 11.5 s of audio at 16 kHz mono after startup |

A device switch permanently kills the SCKit path. The tap recovers within half a second.

**The tap needs an aggregate device**

`AudioHardwareCreateProcessTap` returns a tap object, not a readable device. Wrap it in a
private aggregate device (`kAudioAggregateDeviceTapListKey` with `kAudioSubTapUIDKey`) and
read that with `AudioDeviceCreateIOProcIDWithBlock`.

**Use the mono global tap**

`CATapDescription(monoGlobalTapButExcludeProcesses: [])` gives 48 kHz mono, so the
converter only resamples and never has to downmix. The stereo variant arrives interleaved
and doubles the buffer arithmetic for no benefit.

**Do not set the tap's sample rate**

The tap runs at the output device's rate. Resample to 16 kHz with `AVAudioConverter`.
Asking the system for 16 kHz directly is not supported.

**Cleanup must destroy all three objects**

`AudioDeviceDestroyIOProcID`, `AudioHardwareDestroyAggregateDevice`, and
`AudioHardwareDestroyProcessTap`. A leaked tap keeps the system audio indicator lit after
the app exits.

**Allocation inside the audio callback**

The callback allocates two `AVAudioPCMBuffer`s and runs the converter. That is not
real-time safe, and a long stall could drop samples. It matches the previous SCKit
implementation, and at 48 kHz mono the work is small. If glitches appear under load, move
the conversion to a serial queue and accept one buffer of latency.

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
