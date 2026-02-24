/// coreaudio-tap: captures system audio output via ScreenCaptureKit (macOS 13+)
/// and writes raw float32 mono PCM at 16 kHz to stdout.
///
/// Build: see scripts/build_native.sh
/// Usage: coreaudio-tap   (stdout = raw float32 PCM, 16000 Hz, mono)
/// Exit:  SIGTERM or stdin closed
///
/// Recovery: when the audio stream goes silent (e.g. CoreAudio I/O overload
/// in another process causes SCKit to deliver zero-filled buffers), the binary
/// automatically tears down and recreates the SCStream, reconnecting to the
/// still-alive system audio output.  Up to 10 consecutive restarts are attempted;
/// the counter resets when real audio resumes.  Detection requires ~15 seconds of
/// continuous zeros to avoid false positives on legitimate meeting silence.
///
/// Testing: send SIGUSR1 to inject zero audio (simulates dead stream).
/// The restart logic will detect the zeros and restart the stream, clearing the
/// injection flag so real audio resumes — verifying the full recovery path.

import Accelerate
import AVFoundation
import CoreMedia
import Foundation
import ScreenCaptureKit


// MARK: - Global state

var shouldStop = false
var shouldRestart = false
var restartCount = 0
let maxRestarts = 10

/// Testing hook: SIGUSR1 sets this to true, causing the audio callback to zero
/// out the output buffer.  Cleared automatically after a successful restart.
var forceZeros = false

// 64 KB buffered stdout — eliminates per-callback syscalls; flushed on exit.
setvbuf(stdout, nil, _IOFBF, 65536)

signal(SIGTERM) { _ in shouldStop = true }
signal(SIGINT)  { _ in shouldStop = true }
signal(SIGUSR1) { _ in forceZeros = true }

// Stdin-close watcher — detect parent process death.
let stdinWatcher = DispatchSource.makeReadSource(fileDescriptor: STDIN_FILENO, queue: .global())
stdinWatcher.setEventHandler {
    var buf = [UInt8](repeating: 0, count: 1)
    if read(STDIN_FILENO, &buf, 1) <= 0 { shouldStop = true }
}
stdinWatcher.resume()


// MARK: - Permission check

let contentSema = DispatchSemaphore(value: 0)
var shareableContent: SCShareableContent? = nil
SCShareableContent.getExcludingDesktopWindows(false, onScreenWindowsOnly: true) { result, error in
    if let error = error {
        fputs("coreaudio-tap: SCShareableContent query failed: \(error.localizedDescription)\n", stderr)
    }
    shareableContent = result
    contentSema.signal()
}
if contentSema.wait(timeout: .now() + 5.0) == .timedOut {
    fputs("coreaudio-tap: SCShareableContent query timed out after 5 s\n", stderr)
    exit(1)
}

guard shareableContent != nil, shareableContent!.displays.first != nil else {
    fputs("""
    coreaudio-tap: Screen Recording permission is required to capture system audio.
    Grant it in: System Settings → Privacy & Security → Screen Recording
    Add the app that launched this process (e.g. Terminal) to the list, then re-run.
    \n
    """, stderr)
    exit(1)
}


// MARK: - Audio output handler

final class AudioOutputHandler: NSObject, SCStreamOutput, SCStreamDelegate {

    private var converter: AVAudioConverter? = nil
    private var inputSampleRate: Double = 0

    private let outputFormat = AVAudioFormat(
        commonFormat: .pcmFormatFloat32,
        sampleRate: 16_000,
        channels: 1,
        interleaved: true)!

    // Dead-audio detection: count consecutive zero-filled callbacks.
    // At ~10 fps (minimumFrameInterval = 1/10s), 150 callbacks ≈ 15 seconds.
    private var consecutiveZeroCallbacks = 0
    private let zeroThreshold = 150
    /// True once we've seen real audio — prevents false restarts on startup silence.
    private var hadAudio = false

    /// Reset converter state so the next callback re-detects the system format.
    /// Must dispatch onto the audio queue to serialize with audio callbacks.
    func resetForRestart(on queue: DispatchQueue) {
        queue.sync {
            converter = nil
            inputSampleRate = 0
            consecutiveZeroCallbacks = 0
        }
    }

    func stream(
        _ stream: SCStream,
        didOutputSampleBuffer sampleBuffer: CMSampleBuffer,
        of type: SCStreamOutputType
    ) {
        guard type == .audio else { return }

        let numFrames = AVAudioFrameCount(CMSampleBufferGetNumSamples(sampleBuffer))
        guard numFrames > 0 else { return }

        // Lazy converter setup on first audio callback.
        if converter == nil, let fmtDesc = CMSampleBufferGetFormatDescription(sampleBuffer) {
            let inputFormat = AVAudioFormat(cmAudioFormatDescription: fmtDesc)
            guard let conv = AVAudioConverter(from: inputFormat, to: outputFormat) else {
                fputs("coreaudio-tap: AVAudioConverter init failed for \(inputFormat)\n", stderr)
                return
            }
            converter = conv
            inputSampleRate = inputFormat.sampleRate
            fputs("coreaudio-tap: \(Int(inputFormat.sampleRate)) Hz \(inputFormat.channelCount)ch → 16000 Hz 1ch\n", stderr)
        }
        guard let conv = converter, inputSampleRate > 0 else { return }

        // Copy CMSampleBuffer → AVAudioPCMBuffer for the converter.
        guard let srcBuf = AVAudioPCMBuffer(pcmFormat: conv.inputFormat, frameCapacity: numFrames)
        else { return }
        srcBuf.frameLength = numFrames

        let copyStatus = CMSampleBufferCopyPCMDataIntoAudioBufferList(
            sampleBuffer, at: 0, frameCount: Int32(numFrames),
            into: srcBuf.mutableAudioBufferList)
        guard copyStatus == noErr else { return }

        // Resample: native rate → 16 kHz mono.
        let outFrames = AVAudioFrameCount(
            Double(numFrames) * outputFormat.sampleRate / inputSampleRate + 1)
        guard let dstBuf = AVAudioPCMBuffer(pcmFormat: outputFormat, frameCapacity: outFrames)
        else { return }

        var inputConsumed = false
        var convertError: NSError? = nil
        let convertStatus = conv.convert(to: dstBuf, error: &convertError) { _, status in
            if inputConsumed { status.pointee = .noDataNow; return nil }
            inputConsumed = true
            status.pointee = .haveData
            return srcBuf
        }
        if convertStatus == .error {
            fputs("coreaudio-tap: convert failed: \(convertError?.localizedDescription ?? "unknown")\n", stderr)
            return
        }

        guard dstBuf.frameLength > 0, let channelData = dstBuf.floatChannelData else { return }

        // Testing: inject zeros to simulate dead SCKit stream (toggle via SIGUSR1).
        if forceZeros {
            let count = Int(dstBuf.frameLength)
            for i in 0..<count { channelData[0][i] = 0 }
        }

        // Write raw float32 bytes to stdout (uses the 64 KB setvbuf buffer).
        let byteCount = Int(dstBuf.frameLength) * MemoryLayout<Float>.size
        fwrite(channelData[0], 1, byteCount, stdout)

        // --- Dead-audio detection ---
        // SIMD peak absolute amplitude via Accelerate.
        var maxAmp: Float = 0
        let frameCount = vDSP_Length(dstBuf.frameLength)
        vDSP_maxmgv(channelData[0], 1, &maxAmp, frameCount)

        if maxAmp < 1e-7 {
            consecutiveZeroCallbacks += 1
            if hadAudio && consecutiveZeroCallbacks >= zeroThreshold {
                if restartCount < maxRestarts {
                    fputs("coreaudio-tap: dead audio (\(consecutiveZeroCallbacks) zero callbacks) — restart \(restartCount + 1)/\(maxRestarts)\n", stderr)
                    shouldRestart = true
                }
                consecutiveZeroCallbacks = 0  // always reset to prevent unbounded growth
            }
        } else {
            if consecutiveZeroCallbacks > 0 {
                fputs("coreaudio-tap: audio recovered after \(consecutiveZeroCallbacks) zero callbacks\n", stderr)
            }
            consecutiveZeroCallbacks = 0
            hadAudio = true
            restartCount = 0  // real audio flowing — reset budget for future failures
        }
    }

    func stream(_ stream: SCStream, didStopWithError error: Error) {
        fputs("coreaudio-tap: stream stopped: \(error.localizedDescription)\n", stderr)
        shouldStop = true
    }
}


// MARK: - Stream factory

/// Create a new SCStream, register outputs, and start capture.
/// Returns the running stream, or nil on failure.
func createCaptureStream(
    handler: AudioOutputHandler,
    queue: DispatchQueue
) -> SCStream? {
    // Query current display layout (may have changed since last start).
    let sema = DispatchSemaphore(value: 0)
    var content: SCShareableContent? = nil
    SCShareableContent.getExcludingDesktopWindows(false, onScreenWindowsOnly: true) { result, error in
        if let error = error {
            fputs("coreaudio-tap: SCShareableContent query failed: \(error.localizedDescription)\n", stderr)
        }
        content = result
        sema.signal()
    }
    if sema.wait(timeout: .now() + 5.0) == .timedOut {
        fputs("coreaudio-tap: SCShareableContent query timed out in createCaptureStream\n", stderr)
        return nil
    }

    guard let c = content, let display = c.displays.first else {
        fputs("coreaudio-tap: no display found during stream creation\n", stderr)
        return nil
    }

    let config = SCStreamConfiguration()
    config.capturesAudio = true
    config.excludesCurrentProcessAudio = true
    // IMPORTANT: Do NOT set sampleRate or channelCount — see AGENTS.md.
    config.minimumFrameInterval = CMTime(value: 1, timescale: 10)
    config.width = Int(display.width)
    config.height = Int(display.height)
    config.showsCursor = false

    let filter = SCContentFilter(
        display: display,
        excludingApplications: [],
        exceptingWindows: [])

    let stream = SCStream(filter: filter, configuration: config, delegate: handler)

    do {
        try stream.addStreamOutput(handler, type: .audio, sampleHandlerQueue: queue)
        try stream.addStreamOutput(handler, type: .screen, sampleHandlerQueue: queue)
    } catch {
        fputs("coreaudio-tap: addStreamOutput failed: \(error)\n", stderr)
        return nil
    }

    let startSema = DispatchSemaphore(value: 0)
    var startErr: Error? = nil
    stream.startCapture { err in
        startErr = err
        startSema.signal()
    }
    if startSema.wait(timeout: .now() + 5.0) == .timedOut {
        fputs("coreaudio-tap: startCapture timed out after 5 s\n", stderr)
        return nil
    }

    if let err = startErr {
        fputs("coreaudio-tap: startCapture failed: \(err)\n", stderr)
        return nil
    }

    return stream
}


// MARK: - Start capture

let audioHandler = AudioOutputHandler()
let audioQueue = DispatchQueue(label: "coreaudio-tap.audio", qos: .userInteractive)

guard var captureStream = createCaptureStream(handler: audioHandler, queue: audioQueue) else {
    fputs("coreaudio-tap: initial stream creation failed\n", stderr)
    exit(1)
}


// MARK: - Run loop

while !shouldStop {
    RunLoop.current.run(until: Date(timeIntervalSinceNow: 0.05))

    // Handle stream restart request from the audio callback.
    if shouldRestart && !shouldStop {
        shouldRestart = false
        restartCount += 1
        fputs("coreaudio-tap: restarting stream (attempt \(restartCount)/\(maxRestarts))\n", stderr)

        // Tear down the dead stream (proceed even if stop hangs).
        let stopSema = DispatchSemaphore(value: 0)
        captureStream.stopCapture { _ in stopSema.signal() }
        if stopSema.wait(timeout: .now() + 3.0) == .timedOut {
            fputs("coreaudio-tap: stopCapture timed out during restart — proceeding\n", stderr)
        }

        // Create a fresh stream — reconnects to the (still alive) system audio.
        if let newStream = createCaptureStream(handler: audioHandler, queue: audioQueue) {
            captureStream = newStream
            audioHandler.resetForRestart(on: audioQueue)
            forceZeros = false  // clear test injection after successful restart
            fputs("coreaudio-tap: stream restarted successfully\n", stderr)
        } else {
            fputs("coreaudio-tap: restart failed — stopping\n", stderr)
            shouldStop = true
        }
    }
}


// MARK: - Cleanup

stdinWatcher.cancel()

let finalStopSema = DispatchSemaphore(value: 0)
captureStream.stopCapture { _ in finalStopSema.signal() }
if finalStopSema.wait(timeout: .now() + 3.0) == .timedOut {
    fputs("coreaudio-tap: final stopCapture timed out — exiting anyway\n", stderr)
}

fflush(stdout)
