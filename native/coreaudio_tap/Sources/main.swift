/// coreaudio-tap: captures system audio output via a Core Audio process tap
/// (macOS 14.2+) and writes raw float32 mono PCM at 16 kHz to stdout.
///
/// Build: see scripts/build_native.sh
/// Usage: coreaudio-tap   (stdout = raw float32 PCM, 16000 Hz, mono)
/// Exit:  SIGTERM, SIGINT, or stdin closed
///
/// Why a process tap and not ScreenCaptureKit: ScreenCaptureKit paces audio
/// callbacks off the captured display's video frame clock, so capture stalls
/// when nothing moves on screen, and it can silently start delivering
/// zero-filled buffers after a CoreAudio overload with no error reported.
/// A process tap reads the output mixer directly. It is independent of the
/// display. It still fails silently without the audio capture grant: macOS
/// hands back zeros and reports no error, so the caller watches for the first
/// non-zero sample instead of trusting the API status.
///
/// Measured on macOS 26.6.2: an 8 s run delivered 8.4 s of frames, and two
/// output-device switches mid-run cost 100-500 ms each with immediate recovery.

import AudioToolbox
import AVFoundation
import CoreAudio
import Foundation
import ScreenCaptureKit

let sampleRate = 16_000.0

var shouldStop = false

// Unbuffered stdout: every converted chunk goes straight to the pipe, so the
// Python side's level meter tracks the room instead of a stale buffer.
setvbuf(stdout, nil, _IONBF, 0)

signal(SIGTERM) { _ in shouldStop = true }
signal(SIGINT) { _ in shouldStop = true }

func fail(_ message: String, _ status: OSStatus? = nil) -> Never {
    let suffix = status.map { " (OSStatus \($0))" } ?? ""
    FileHandle.standardError.write("coreaudio-tap: \(message)\(suffix)\n".data(using: .utf8)!)
    exit(1)
}

func note(_ message: String) {
    FileHandle.standardError.write("coreaudio-tap: \(message)\n".data(using: .utf8)!)
}

// Stdin-close watcher: detects parent process death so the tap never outlives
// the app and keeps holding the tap.
let stdinWatcher = DispatchSource.makeReadSource(fileDescriptor: STDIN_FILENO, queue: .global())
stdinWatcher.setEventHandler {
    var buffer = [UInt8](repeating: 0, count: 1)
    if read(STDIN_FILENO, &buffer, 1) <= 0 { shouldStop = true }
}
stdinWatcher.resume()

/// Raise the system audio permission dialog, which a process tap never does.
///
/// The tap checks the grant silently: missing grant, no error, all zeros.
/// ScreenCaptureKit is the only API that asks the user, so the desktop bundle
/// calls it once before capturing. The loop keeps asking while the user is in
/// System Settings, because a grant takes effect without restarting the process.
@available(macOS 14.2, *)
func ensurePermission() {
    final class Flag {
        private let lock = NSLock()
        private var value = false
        func set() {
            lock.lock()
            value = true
            lock.unlock()
        }
        func get() -> Bool {
            lock.lock()
            defer { lock.unlock() }
            return value
        }
    }

    let deadline = Date().addingTimeInterval(45)
    while Date() < deadline {
        let granted = Flag()
        let semaphore = DispatchSemaphore(value: 0)
        SCShareableContent.getExcludingDesktopWindows(false, onScreenWindowsOnly: true) { content, _ in
            if content != nil { granted.set() }
            semaphore.signal()
        }
        _ = semaphore.wait(timeout: .now() + 10)
        if granted.get() {
            note("permission granted")
            return
        }
        note(
            "waiting for permission: enable this app in System Settings > "
                + "Privacy & Security > Screen & System Audio Recording")
        Thread.sleep(forTimeInterval: 3)
    }
    note("permission not granted; system audio will be silent")
}

@available(macOS 14.2, *)
func capture() {
    // MARK: - Input format

    let tapDescription = CATapDescription(monoGlobalTapButExcludeProcesses: [])
    tapDescription.isPrivate = true
    tapDescription.muteBehavior = .unmuted
    tapDescription.name = "coreaudio-tap"

    var tapID = AudioObjectID(kAudioObjectUnknown)
    var status = AudioHardwareCreateProcessTap(tapDescription, &tapID)
    if status != noErr { fail("AudioHardwareCreateProcessTap failed", status) }

    var tapFormat = AudioStreamBasicDescription()
    var tapFormatSize = UInt32(MemoryLayout<AudioStreamBasicDescription>.size)
    var formatAddress = AudioObjectPropertyAddress(
        mSelector: kAudioTapPropertyFormat,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain)
    status = AudioObjectGetPropertyData(tapID, &formatAddress, 0, nil, &tapFormatSize, &tapFormat)
    if status != noErr { fail("could not read the tap format", status) }

    guard let inputFormat = AVAudioFormat(streamDescription: &tapFormat) else {
        fail("unsupported tap format")
    }
    guard let outputFormat = AVAudioFormat(
        commonFormat: .pcmFormatFloat32, sampleRate: sampleRate, channels: 1, interleaved: false)
    else {
        fail("could not build the 16 kHz mono output format")
    }
    guard let converter = AVAudioConverter(from: inputFormat, to: outputFormat) else {
        fail("could not build the 48 kHz to 16 kHz converter")
    }
    note("\(Int(inputFormat.sampleRate)) Hz \(inputFormat.channelCount)ch -> \(Int(sampleRate)) Hz 1ch")

    // MARK: - Aggregate device

    let aggregateDescription: [String: Any] = [
        kAudioAggregateDeviceNameKey as String: "coreaudio-tap",
        kAudioAggregateDeviceUIDKey as String: UUID().uuidString,
        kAudioAggregateDeviceIsPrivateKey as String: true,
        kAudioAggregateDeviceIsStackedKey as String: false,
        kAudioAggregateDeviceTapListKey as String: [
            [
                kAudioSubTapUIDKey as String: tapDescription.uuid.uuidString,
                kAudioSubTapDriftCompensationKey as String: true,
            ]
        ],
    ]

    var aggregateID = AudioObjectID(kAudioObjectUnknown)
    status = AudioHardwareCreateAggregateDevice(aggregateDescription as CFDictionary, &aggregateID)
    if status != noErr { fail("AudioHardwareCreateAggregateDevice failed", status) }

    // MARK: - Audio callback

    var procID: AudioDeviceIOProcID?
    status = AudioDeviceCreateIOProcIDWithBlock(&procID, aggregateID, nil) {
        _, inInputData, _, _, _ in
        let buffers = UnsafeMutableAudioBufferListPointer(UnsafeMutablePointer(mutating: inInputData))
        guard let first = buffers.first, let source = first.mData else { return }

        let bytesPerFrame = MemoryLayout<Float>.size * Int(inputFormat.channelCount)
        let frames = AVAudioFrameCount(Int(first.mDataByteSize) / bytesPerFrame)
        guard frames > 0,
              let input = AVAudioPCMBuffer(pcmFormat: inputFormat, frameCapacity: frames),
              let inputData = input.floatChannelData?[0]
        else { return }

        input.frameLength = frames
        memcpy(inputData, source, Int(first.mDataByteSize))

        let capacity = AVAudioFrameCount(Double(frames) * sampleRate / inputFormat.sampleRate) + 1024
        guard let output = AVAudioPCMBuffer(pcmFormat: outputFormat, frameCapacity: capacity) else { return }

        var conversionError: NSError?
        var supplied = false
        converter.convert(to: output, error: &conversionError) { _, outStatus in
            if supplied {
                outStatus.pointee = .noDataNow
                return nil
            }
            supplied = true
            outStatus.pointee = .haveData
            return input
        }
        if conversionError != nil { return }

        let produced = Int(output.frameLength)
        if produced > 0, let samples = output.floatChannelData?[0] {
            fwrite(samples, MemoryLayout<Float>.size, produced, stdout)
        }
    }
    guard status == noErr, let proc = procID else { fail("AudioDeviceCreateIOProcIDWithBlock failed", status) }

    status = AudioDeviceStart(aggregateID, proc)
    if status != noErr { fail("AudioDeviceStart failed", status) }
    note("capturing")

    while !shouldStop {
        RunLoop.current.run(until: Date(timeIntervalSinceNow: 0.05))
    }

    stdinWatcher.cancel()
    AudioDeviceStop(aggregateID, proc)
    AudioDeviceDestroyIOProcID(aggregateID, proc)
    AudioHardwareDestroyAggregateDevice(aggregateID)
    AudioHardwareDestroyProcessTap(tapID)
    fflush(stdout)
}

if #available(macOS 14.2, *) {
    if CommandLine.arguments.contains("--ensure-permission") {
        ensurePermission()
    }
    capture()
} else {
    fail("Core Audio process taps require macOS 14.2 or newer")
}
