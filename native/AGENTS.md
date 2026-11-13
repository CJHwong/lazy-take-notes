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

Requires Xcode Command Line Tools and Swift 5.10 or newer (macOS only).

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

Turn on **System Audio Recording Only** for the app that launches the binary. The
combined screen toggle also works, but it grants more than the tap needs.

macOS does not prompt when a tap is created. With the grant missing, tap creation
succeeds, `AudioDeviceStart` succeeds, and every frame is zeros. No OSStatus is
returned. That is why the worker warns after 15 s with no non-zero sample.

The desktop bundle passes `--ensure-permission`, which raises the dialog through
ScreenCaptureKit before capture starts, because that is the only API that asks.
The grant takes effect without a restart, so the binary retries for 45 s while the
user is in System Settings. A terminal run does not pass the flag and keeps the
silent behaviour.

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

`CATapDescription(monoGlobalTapButExcludeProcesses: [])` supplies mono float32.
The helper reads the tap format and the aggregate nominal rate at each start.
It rebuilds the converter when the device or format changes.

**Do not set the tap's sample rate**

The tap runs at the output device's rate. Resample to 16 kHz with `AVAudioConverter`.
Asking the system for 16 kHz directly is not supported.

**Cleanup must destroy all three objects**

`AudioDeviceDestroyIOProcID`, `AudioHardwareDestroyAggregateDevice`, and
`AudioHardwareDestroyProcessTap`. A leaked tap keeps the system audio indicator lit after
the app exits.

**Bounded callback handoff**

The callback copies samples and capture time into a preallocated Swift ring.
The ring uses Apple's Swift Atomics package, pinned to version 1.3.1.
Keep `Package.resolved` with the source so builds use the recorded dependency revision.
This preserves macOS 14.2 capture support without the newer standard-library `Atomic` type.
It does not allocate, convert samples, or write to a pipe.
The ring has 64 slots, with room for 8192 mono frames per slot.
Full or oversized buffers increment the dropped-frame counter.
The consumer retains each slot until conversion finishes.

The main control loop converts samples and writes stdout without blocking.
It honors each converter input request and preserves the remaining input samples.
Ignoring those requests loses samples when a capture buffer exceeds the requested size.
Pending stdout storage has a 256,000-byte limit, equal to four seconds of output.
This holds a supported three-second gap while a healthy consumer resumes.
If storage fills, the helper reports an error and releases capture resources.
It preserves partial-write suffixes, so later writes cannot corrupt float32 framing.

Capture timestamps preserve gaps up to three seconds, including device rebuild gaps.
A larger gap produces an explicit error instead of an unbounded silence allocation.
Dropped frames and timestamp gaps appear on stderr.

**Device recovery and first-buffer confirmation**

The helper watches the default output, tap format, and aggregate nominal rate.
It waits 300 ms after a change before rebuilding the tap and aggregate.
It reports `capturing` only after receiving a buffer.
Missing buffers trigger a rebuild after three seconds.
Startup stops after three attempts without a delivered buffer.
Zero-valued buffers count as delivery; valid silence does not trigger this watchdog.

**Stdin-close detection for clean subprocess teardown**

The binary watches stdin for EOF via `DispatchSource.makeReadSource`. When the Python
parent closes `proc.stdin` (or crashes), stdin becomes readable with 0 bytes, and the
binary sets an atomic stop flag and exits cleanly. This prevents zombie processes.
Dispatch signal sources handle SIGTERM and SIGINT outside the raw signal handler.
This keeps Swift runtime calls out of the signal context.

### Python integration

The Python gateway is at:
`src/lazy_take_notes/l3_interface_adapters/gateways/coreaudio_tap_source.py`

It launches the binary via `subprocess.Popen` with `stdout=PIPE, stderr=PIPE, stdin=PIPE`.
A background reader thread pumps stdout bytes (raw float32) into a bounded `AudioBuffer`.
If the process exits with a non-zero code, the error is surfaced on the next `read()` call.

Python adapter tests use process fakes and small executable helpers:
`tests/l3_interface_adapters/gateways/test_coreaudio_tap_source.py`

Native support tests exercise the ring, conversion, pipe stalls, and buffer health:

```bash
swift test --package-path native/coreaudio_tap
```

These tests do not replace the manual capture and device-switch checks.
