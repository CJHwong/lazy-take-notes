/// coreaudio-tap: captures system audio output via ScreenCaptureKit (macOS 13+)
/// and writes raw float32 mono PCM at 16 kHz to stdout.
///
/// Build: see scripts/build_native.sh
/// Usage: coreaudio-tap   (stdout = raw float32 PCM, 16000 Hz, mono)
/// Exit:  SIGTERM or stdin closed
///
/// --- Python analogy for the whole program ---
///
/// This is equivalent to a Python script that:
///   1. Registers signal handlers (signal.signal)
///   2. Asks macOS for permission to record the screen/audio
///   3. Starts an audio stream callback (like sounddevice.InputStream)
///   4. In the callback: resamples from 48kHz stereo → 16kHz mono,
///      then writes raw float32 bytes to sys.stdout.buffer
///   5. Loops until SIGTERM or stdin is closed
///   6. Cleanly shuts down the stream
///
/// The data written to stdout is consumed by CoreAudioTapSource.py,
/// which reads it in a background thread and puts numpy arrays on a Queue.

// --- Imports (same as Python's `import`) ---
// These are Apple framework modules, not third-party packages.
import AVFoundation    // AVAudioConverter, AVAudioPCMBuffer, AVAudioFormat
import CoreMedia       // CMSampleBuffer, CMTime
import Foundation      // FileHandle, DispatchSemaphore, DispatchQueue, DispatchSource
import ScreenCaptureKit // SCStream, SCShareableContent, SCStreamConfiguration, SCContentFilter


// MARK: - Signal handling
// (MARK is like a comment divider — Xcode uses it for navigation, like a region header)

// `var` = mutable variable (like Python's regular assignment)
// `let` = immutable binding (like Python's conceptual "constant" — reassignment is a compile error)
var shouldStop = false

// signal() is identical to Python's signal.signal().
// The closure `{ _ in shouldStop = true }` is Swift's lambda syntax.
// Python equivalent: signal.signal(signal.SIGTERM, lambda sig, frame: globals().__setitem__('shouldStop', True))
// The `_` discards the signal number argument we don't need.
signal(SIGTERM) { _ in shouldStop = true }
signal(SIGINT)  { _ in shouldStop = true }

// --- Stdin-close watcher ---
// When the Python parent process dies or calls proc.stdin.close(), the stdin fd
// becomes readable (returns 0 bytes = EOF). We use that as a "please exit" signal
// so we don't become a zombie if the parent crashes without sending SIGTERM.
//
// Python equivalent:
//   def _watch_stdin():
//       while sys.stdin.buffer.read(1): pass   # blocks until EOF
//       global shouldStop; shouldStop = True
//   threading.Thread(target=_watch_stdin, daemon=True).start()
//
// DispatchSource.makeReadSource is a GCD (Grand Central Dispatch) file-descriptor watcher.
// GCD is macOS's built-in thread pool system (analogous to Python's concurrent.futures).
// `.global()` = use a shared background thread pool queue.
let stdinWatcher = DispatchSource.makeReadSource(fileDescriptor: STDIN_FILENO, queue: .global())
stdinWatcher.setEventHandler {
    var buf = [UInt8](repeating: 0, count: 1)  // allocate a 1-byte buffer (like bytearray(1))
    // read() is the POSIX syscall. Returns <= 0 on EOF or error.
    if read(STDIN_FILENO, &buf, 1) <= 0 { shouldStop = true }
}
stdinWatcher.resume()  // sources start suspended — must call resume() to activate


// MARK: - Permission check + content discovery

// --- DispatchSemaphore: bridging async callbacks to synchronous code ---
//
// Swift/macOS APIs are heavily async callback-based (like JavaScript promises or
// Python asyncio, but without async/await syntax here). When we need to WAIT for
// an async result before proceeding, we use a semaphore:
//
// Python equivalent:
//   sema = threading.Semaphore(0)   # starts at 0 (locked)
//   result_holder = [None]
//   def callback(result, error):
//       result_holder[0] = result
//       sema.release()              # unblock the waiter
//   async_api(callback)
//   sema.acquire()                  # blocks until callback fires
//   result = result_holder[0]
//
// DispatchSemaphore(value: 0) starts at 0 (acquire will block immediately).
// .signal() = release(), .wait() = acquire().
let contentSema = DispatchSemaphore(value: 0)

// `var x: Type? = nil` declares an Optional variable (like Python's x: Optional[T] = None).
// `SCShareableContent?` means it can be nil (Python None) or an SCShareableContent object.
var shareableContent: SCShareableContent? = nil

// Ask ScreenCaptureKit what's on screen. This triggers the Screen Recording TCC permission
// dialog if not yet granted. If denied, `result` will be nil and `_` (the error) is non-nil.
//
// `{ result, _ in ... }` is a trailing closure (lambda). The `_` ignores the error parameter.
SCShareableContent.getExcludingDesktopWindows(false, onScreenWindowsOnly: true) { result, _ in
    shareableContent = result
    contentSema.signal()  // unblock the .wait() below
}
contentSema.wait()  // block here until the callback above fires

// `guard let x = optional else { ... }` is Swift's "early exit if nil" pattern.
// Python equivalent:
//   if shareableContent is None or not shareableContent.displays:
//       sys.stderr.write("...\n"); sys.exit(1)
//   content = shareableContent
//   display = content.displays[0]
guard let content = shareableContent, let display = content.displays.first else {
    fputs("""
    coreaudio-tap: Screen Recording permission is required to capture system audio.
    Grant it in: System Settings → Privacy & Security → Screen Recording
    Add the app that launched this process (e.g. Terminal) to the list, then re-run.
    \n
    """, stderr)
    // `fputs(..., stderr)` = Python's `sys.stderr.write(...)` (low-level C stdio)
    exit(1)
}


// MARK: - Stream configuration

// SCStreamConfiguration is a plain data bag (like a Python dataclass or dict of settings).
let streamConfig = SCStreamConfiguration()

// Tell SCKit we want audio. Without this flag, only video frames are delivered.
streamConfig.capturesAudio = true

// Don't feed our own process's audio back to us (avoids feedback loops).
streamConfig.excludesCurrentProcessAudio = true

// IMPORTANT: Do NOT set streamConfig.sampleRate or streamConfig.channelCount here.
// Setting a non-standard sample rate disables audio delivery entirely in SCKit.
// We capture at the system's native format (48kHz stereo) and resample in the callback.

// Audio delivery in SCKit is tied to video frame ticks. Without a video handler registered,
// audio callbacks never fire. Setting a low frame rate (10fps) gives us ~100ms audio chunks
// at minimal CPU cost.
// CMTime(value: 1, timescale: 10) = fraction 1/10 second (Python: fractions.Fraction(1, 10))
streamConfig.minimumFrameInterval = CMTime(value: 1, timescale: 10)

// SCKit requires width/height for video capture config even though we discard video frames.
streamConfig.width  = Int(display.width)
streamConfig.height = Int(display.height)
streamConfig.showsCursor = false  // don't waste time rendering the cursor

// SCContentFilter tells SCKit WHAT to capture: the entire main display,
// excluding no applications and excepting no windows (i.e., capture everything).
// Python analogy: a filter/query object passed to the stream constructor.
let contentFilter = SCContentFilter(
    display: display,
    excludingApplications: [],
    exceptingWindows: [])


// MARK: - Audio output handler

// --- Class declaration ---
// `final class` = a class that cannot be subclassed (like Python's @final from typing).
// `NSObject` = the base class for all Objective-C/macOS objects. Required here because
//   SCStreamOutput and SCStreamDelegate are ObjC protocols.
// `SCStreamOutput` = a protocol (like Python's ABC / typing.Protocol). It requires us to
//   implement `stream(_:didOutputSampleBuffer:of:)`.
// `SCStreamDelegate` = another protocol requiring `stream(_:didStopWithError:)`.
//
// Python analogy:
//   class AudioOutputHandler:
//       def stream_did_output_sample_buffer(self, stream, sample_buffer, output_type): ...
//       def stream_did_stop_with_error(self, stream, error): ...
final class AudioOutputHandler: NSObject, SCStreamOutput, SCStreamDelegate {

    // Instance variables (like Python self.x = None in __init__).
    // `AVAudioConverter?` = Optional — starts as nil, built on first callback.
    private var converter: AVAudioConverter? = nil
    private var inputSampleRate: Double = 0

    // The output format we want: 16kHz, mono, interleaved float32.
    // `let` = set once in init, never changes. `!` = force-unwrap (crash if nil).
    // We know this format is valid so force-unwrapping is safe.
    // Python analogy: self.output_format = {"sample_rate": 16000, "channels": 1, "format": "float32"}
    private let outputFormat = AVAudioFormat(
        commonFormat: .pcmFormatFloat32,
        sampleRate: 16_000,      // underscores in number literals are ignored (style only), like 16_000 in Python
        channels: 1,
        interleaved: true)!

    // --- Audio callback ---
    // This is called by SCKit on the audioQueue thread every ~100ms (at our 10fps setting).
    // Python analogy: the `callback` parameter of sounddevice.InputStream — called from a
    // background thread whenever a new chunk of audio is ready.
    //
    // `sampleBuffer` is a CMSampleBuffer: Apple's container for a chunk of audio or video data.
    // Think of it as a wrapper around a numpy array + metadata (format, timestamp, etc.).
    // `type` tells us whether this is an audio or video frame — we only care about .audio.
    func stream(
        _ stream: SCStream,
        didOutputSampleBuffer sampleBuffer: CMSampleBuffer,
        of type: SCStreamOutputType
    ) {
        // Ignore video frames — we registered a .screen handler only because SCKit requires it
        // for audio delivery to work; we don't process those frames.
        guard type == .audio else { return }

        // How many audio frames (samples) are in this buffer?
        // Python analogy: len(chunk) where chunk is a numpy array
        let numFrames = AVAudioFrameCount(CMSampleBufferGetNumSamples(sampleBuffer))
        guard numFrames > 0 else { return }  // skip empty buffers

        // --- Lazy converter setup (first callback only) ---
        // On the first audio callback we learn the system's native audio format
        // (typically 48kHz non-interleaved stereo float32 on Apple Silicon).
        // We build a resampler once and reuse it for all subsequent callbacks.
        //
        // Python analogy (conceptually):
        //   if self.converter is None:
        //       input_format = detect_format(sample_buffer)
        //       self.converter = scipy.signal.resample  # configured for this rate
        //       self.input_sample_rate = input_format.sample_rate
        if converter == nil, let fmtDesc = CMSampleBufferGetFormatDescription(sampleBuffer) {
            // AVAudioFormat wraps a CoreAudio AudioStreamBasicDescription (ASBD).
            // `cmAudioFormatDescription:` extracts the format from the buffer's metadata.
            let inputFormat = AVAudioFormat(cmAudioFormatDescription: fmtDesc)
            converter = AVAudioConverter(from: inputFormat, to: outputFormat)
            inputSampleRate = inputFormat.sampleRate
            // Log the detected format to stderr so the user can see what the system is doing.
            fputs("coreaudio-tap: \(Int(inputFormat.sampleRate)) Hz \(inputFormat.channelCount)ch → 16000 Hz 1ch\n", stderr)
        }
        // If we haven't built the converter yet (shouldn't happen after the block above), skip.
        guard let conv = converter, inputSampleRate > 0 else { return }

        // --- Copy CMSampleBuffer data into a typed AVAudioPCMBuffer ---
        //
        // CMSampleBuffer is a generic container. We need to unwrap it into an
        // AVAudioPCMBuffer so the converter can read it.
        //
        // Python analogy:
        //   src_array = np.frombuffer(sample_buffer.raw_bytes, dtype=np.float32)
        //   src_array = src_array.reshape(num_frames, num_channels)
        guard let srcBuf = AVAudioPCMBuffer(pcmFormat: conv.inputFormat, frameCapacity: numFrames)
        else { return }
        srcBuf.frameLength = numFrames  // tell the buffer how many frames are valid

        // Copy the actual PCM bytes from the CMSampleBuffer into srcBuf.
        // This handles the non-interleaved layout of the system audio format correctly.
        // `noErr` = 0 (OSStatus success code, like checking returncode == 0)
        let copyStatus = CMSampleBufferCopyPCMDataIntoAudioBufferList(
            sampleBuffer, at: 0, frameCount: Int32(numFrames),
            into: srcBuf.mutableAudioBufferList)
        guard copyStatus == noErr else { return }

        // --- Resample: input rate → 16kHz mono ---
        //
        // Calculate output frame count: output_frames = input_frames * (16000 / input_rate) + 1
        // The +1 is a ceiling guard for rounding.
        // Python analogy: out_len = int(num_frames * 16000 / input_sample_rate) + 1
        let outFrames = AVAudioFrameCount(
            Double(numFrames) * outputFormat.sampleRate / inputSampleRate + 1)
        guard let dstBuf = AVAudioPCMBuffer(pcmFormat: outputFormat, frameCapacity: outFrames)
        else { return }

        // AVAudioConverter.convert uses a "pull" callback model: the converter calls our
        // closure to ask for input data. We provide `srcBuf` once, then signal `.noDataNow`.
        //
        // Python analogy (simplified):
        //   given = False
        //   def provide_input(num_frames, status):
        //       nonlocal given
        //       if given: status = "no_data"; return None
        //       given = True; status = "have_data"; return src_array
        //   dst_array = converter.convert(provide_input)
        var inputConsumed = false
        _ = conv.convert(to: dstBuf, error: nil) { _, status in
            if inputConsumed { status.pointee = .noDataNow; return nil }
            inputConsumed = true
            status.pointee = .haveData
            return srcBuf
        }
        // `status.pointee` = dereferencing a pointer (like `*status` in C).
        // `status` is an UnsafeMutablePointer<AVAudioConverterInputStatus>.

        // Skip if the converter produced no output (e.g., startup transient).
        guard dstBuf.frameLength > 0, let channelData = dstBuf.floatChannelData else { return }

        // --- Write raw float32 bytes to stdout ---
        //
        // `dstBuf.floatChannelData` is an array of Float* pointers, one per channel.
        // Since we have 1 channel (mono), channelData[0] is the pointer to our samples.
        //
        // Python analogy:
        //   arr = np.frombuffer(dst_buf, dtype=np.float32)[:dst_buf.frame_length]
        //   sys.stdout.buffer.write(arr.tobytes())
        let byteCount = Int(dstBuf.frameLength) * MemoryLayout<Float>.size
        // MemoryLayout<Float>.size = 4 (same as np.dtype('float32').itemsize)

        // `withMemoryRebound` temporarily reinterprets the Float* pointer as UInt8* (byte pointer)
        // so we can pass it to `Data(bytes:count:)` — like a safe C-style cast.
        channelData[0].withMemoryRebound(to: UInt8.self, capacity: byteCount) { ptr in
            // Write the raw bytes directly to file descriptor 1 (stdout).
            // Python: sys.stdout.buffer.write(bytes(ptr[:byteCount]))
            FileHandle.standardOutput.write(Data(bytes: ptr, count: byteCount))
        }
    }

    // --- Stream error delegate ---
    // Called if SCKit encounters a fatal error (e.g., permission revoked mid-session).
    // Python analogy: an error callback passed to sounddevice.InputStream.
    func stream(_ stream: SCStream, didStopWithError error: Error) {
        fputs("coreaudio-tap: stream stopped: \(error.localizedDescription)\n", stderr)
        shouldStop = true  // triggers the main loop to exit cleanly
    }
}


// MARK: - Start capture

let audioHandler = AudioOutputHandler()

// Create the stream with our filter (what to capture) and config (how to capture).
// `delegate: audioHandler` registers `didStopWithError` callbacks on audioHandler.
let captureStream = SCStream(filter: contentFilter, configuration: streamConfig, delegate: audioHandler)

// DispatchQueue = a serial or concurrent task queue (like threading.Thread or ThreadPoolExecutor).
// `.userInteractive` QoS = high priority, same as the UI thread — keeps audio latency low.
let audioQueue = DispatchQueue(label: "coreaudio-tap.audio", qos: .userInteractive)

// Register our handler for BOTH audio and video output types.
// CRITICAL: SCKit only delivers audio callbacks when a .screen handler is also registered.
// We add both here; the audio handler ignores video frames via `guard type == .audio`.
do {
    try captureStream.addStreamOutput(audioHandler, type: .audio, sampleHandlerQueue: audioQueue)
    // `.screen` handler registration is required — audio delivery is tied to video frame ticks.
    try captureStream.addStreamOutput(audioHandler, type: .screen, sampleHandlerQueue: audioQueue)
} catch {
    fputs("coreaudio-tap: addStreamOutput failed: \(error)\n", stderr)
    exit(1)
}

// Start capture (async — use semaphore to wait for completion/error).
// Python analogy: stream.start(); wait_for_start_event.wait()
let startSema = DispatchSemaphore(value: 0)
var startError: Error? = nil
captureStream.startCapture { error in
    startError = error
    startSema.signal()
}
startSema.wait()

if let err = startError {
    fputs("coreaudio-tap: startCapture failed: \(err)\n", stderr)
    exit(1)
}


// MARK: - Run loop

// This is the main thread event loop. Unlike Python's while True: time.sleep(0.05),
// RunLoop.current.run(until:) also processes pending macOS events (timers, GCD callbacks,
// I/O sources like stdinWatcher). This keeps the stdin-close watcher and signal handlers alive.
//
// Python analogy (approximate):
//   while not shouldStop:
//       time.sleep(0.05)   # but also drains event queues
while !shouldStop {
    RunLoop.current.run(until: Date(timeIntervalSinceNow: 0.05))
    // Date(timeIntervalSinceNow: 0.05) = datetime.now() + timedelta(seconds=0.05)
}


// MARK: - Cleanup

// Stop the capture stream gracefully (also async — wait for completion).
// Python analogy: stream.stop(); wait_for_stop_event.wait()
let stopSema = DispatchSemaphore(value: 0)
captureStream.stopCapture { _ in stopSema.signal() }
stopSema.wait()
// Program exits here (falls off the end of main.swift = implicit exit(0)).
