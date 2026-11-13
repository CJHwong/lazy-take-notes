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

import AVFoundation
import Atomics
import AudioRing
import AudioToolbox
import CaptureSupport
import CoreAudio
import Foundation
import ScreenCaptureKit

let sampleRate = 16_000.0
let stopRequested = ManagedAtomic(false)

func requestStop() { stopRequested.store(true, ordering: .relaxed) }
func shouldStop() -> Bool { stopRequested.load(ordering: .relaxed) }

// Dispatch handles signals outside the raw signal context, where Swift runtime calls are safe.
let stopSignals = [SIGTERM, SIGINT].map { number in
  signal(number, SIG_IGN)
  let source = DispatchSource.makeSignalSource(signal: number, queue: .global())
  source.setEventHandler { requestStop() }
  source.resume()
  return source
}
signal(SIGPIPE, SIG_IGN)

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
  if read(STDIN_FILENO, &buffer, 1) <= 0 { requestStop() }
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
  while Date() < deadline && !shouldStop() {
    let granted = Flag()
    let semaphore = DispatchSemaphore(value: 0)
    SCShareableContent.getExcludingDesktopWindows(false, onScreenWindowsOnly: true) { content, _ in
      if content != nil { granted.set() }
      semaphore.signal()
    }
    _ = waitForCompletion(semaphore, timeout: 10, shouldStop: shouldStop)
    if shouldStop() { return }
    if granted.get() {
      note("permission granted")
      return
    }
    note(
      "waiting for permission: enable this app in System Settings > "
        + "Privacy & Security > Screen & System Audio Recording")
    _ = waitForCompletion(DispatchSemaphore(value: 0), timeout: 3, shouldStop: shouldStop)
  }
  note("permission not granted; system audio will be silent")
}

func address(_ selector: AudioObjectPropertySelector) -> AudioObjectPropertyAddress {
  AudioObjectPropertyAddress(
    mSelector: selector, mScope: kAudioObjectPropertyScopeGlobal,
    mElement: kAudioObjectPropertyElementMain)
}

func check(_ status: OSStatus, _ operation: String) throws {
  if status != noErr { throw CaptureFailure("\(operation) failed (OSStatus \(status))") }
}

@available(macOS 14.2, *)
final class CaptureSession {
  private let ring: AudioRing
  private var tapID = AudioObjectID(kAudioObjectUnknown)
  private var aggregateID = AudioObjectID(kAudioObjectUnknown)
  private var procID: AudioDeviceIOProcID?
  private var callbackStopped = true
  private var listeners:
    [(AudioObjectID, AudioObjectPropertyAddress, AudioObjectPropertyListenerBlock)] = []
  private var converter: PCMConverter?
  private var inputRate = 48_000.0
  private var tapRate = 48_000.0
  private var reportedDrops: UInt64 = 0
  private(set) var rebuildAt: Double?
  private(set) var health = CaptureHealth(startedAt: ProcessInfo.processInfo.systemUptime)
  private(set) var announced = false
  private let maximumFrames = 8192

  init() throws {
    guard let ring = AudioRing(capacity: 64, maximumFrames: maximumFrames)
    else { throw CaptureFailure("could not allocate the audio ring") }
    self.ring = ring
  }

  deinit {
    stop()
    // A failed teardown must not free storage that an active callback can still use.
    guard callbackStopped, procID == nil, aggregateID == kAudioObjectUnknown,
      tapID == kAudioObjectUnknown
    else { fail("could not release capture hardware safely") }
  }

  func start() throws {
    let description = CATapDescription(monoGlobalTapButExcludeProcesses: [])
    description.isPrivate = true
    description.muteBehavior = .unmuted
    description.name = "coreaudio-tap"
    try check(AudioHardwareCreateProcessTap(description, &tapID), "AudioHardwareCreateProcessTap")
    let format = try readTapFormat()
    guard format.mFormatID == kAudioFormatLinearPCM,
      format.mFormatFlags & kAudioFormatFlagIsFloat != 0,
      format.mBitsPerChannel == 32, format.mChannelsPerFrame == 1
    else { throw CaptureFailure("unsupported tap format; expected float32 mono") }
    tapRate = format.mSampleRate
    try createAggregate(tapUID: description.uuid.uuidString)
    inputRate = nominalRate() ?? tapRate
    if inputRate != tapRate { note("tap reports \(tapRate) Hz; aggregate runs at \(inputRate) Hz") }
    converter = try PCMConverter(sampleRate: inputRate, maximumFrames: maximumFrames)
    note("\(Int(inputRate)) Hz 1ch -> 16000 Hz 1ch")
    try startCallback()
    try watchChanges()
  }

  private func readTapFormat() throws -> AudioStreamBasicDescription {
    var format = AudioStreamBasicDescription()
    var size = UInt32(MemoryLayout<AudioStreamBasicDescription>.size)
    var property = address(kAudioTapPropertyFormat)
    try check(
      AudioObjectGetPropertyData(tapID, &property, 0, nil, &size, &format), "read tap format")
    return format
  }

  private func nominalRate() -> Double? {
    var rate = Double(0)
    var size = UInt32(MemoryLayout<Double>.size)
    var property = address(kAudioDevicePropertyNominalSampleRate)
    guard AudioObjectGetPropertyData(aggregateID, &property, 0, nil, &size, &rate) == noErr,
      rate > 0
    else { return nil }
    return rate
  }

  private func createAggregate(tapUID: String) throws {
    let description: [String: Any] = [
      kAudioAggregateDeviceNameKey as String: "coreaudio-tap",
      kAudioAggregateDeviceUIDKey as String: UUID().uuidString,
      kAudioAggregateDeviceIsPrivateKey as String: true,
      kAudioAggregateDeviceIsStackedKey as String: false,
      kAudioAggregateDeviceTapListKey as String: [
        [
          kAudioSubTapUIDKey as String: tapUID,
          kAudioSubTapDriftCompensationKey as String: true,
        ]
      ],
    ]
    try check(
      AudioHardwareCreateAggregateDevice(description as CFDictionary, &aggregateID),
      "AudioHardwareCreateAggregateDevice")
  }

  private func startCallback() throws {
    let ring = self.ring
    try check(
      AudioDeviceCreateIOProcIDWithBlock(&procID, aggregateID, nil) { _, input, timestamp, _, _ in
        let buffers = UnsafeMutableAudioBufferListPointer(UnsafeMutablePointer(mutating: input))
        guard let first = buffers.first, let samples = first.mData,
          first.mNumberChannels == 1, first.mDataByteSize > 0
        else { return }
        let count = first.mDataByteSize / UInt32(MemoryLayout<Float>.size)
        let hostTime =
          timestamp.pointee.mFlags.contains(.hostTimeValid)
          ? timestamp.pointee.mHostTime : mach_absolute_time()
        _ = ring.enqueue(
          samples.assumingMemoryBound(to: Float.self), count: Int(count), hostTime: hostTime)
      }, "AudioDeviceCreateIOProcIDWithBlock")
    guard let procID else { throw CaptureFailure("audio callback identifier is missing") }
    callbackStopped = false
    try check(AudioDeviceStart(aggregateID, procID), "AudioDeviceStart")
  }

  private func watchChanges() throws {
    try watch(AudioObjectID(kAudioObjectSystemObject), kAudioHardwarePropertyDefaultOutputDevice) {
      true
    }
    try watch(tapID, kAudioTapPropertyFormat) { [weak self] in
      guard let self, let format = try? self.readTapFormat() else { return true }
      return format.mSampleRate != self.tapRate || format.mChannelsPerFrame != 1
    }
    try watch(aggregateID, kAudioDevicePropertyNominalSampleRate) { [weak self] in
      guard let self else { return false }
      return self.nominalRate() != self.inputRate
    }
  }

  private func watch(
    _ object: AudioObjectID, _ selector: AudioObjectPropertySelector,
    changed: @escaping () -> Bool
  ) throws {
    var property = address(selector)
    guard AudioObjectHasProperty(object, &property) else { return }
    let listener: AudioObjectPropertyListenerBlock = { [weak self] _, _ in
      if changed() { self?.rebuildAt = ProcessInfo.processInfo.systemUptime + 0.3 }
    }
    try check(
      AudioObjectAddPropertyListenerBlock(object, &property, .main, listener), "add device listener"
    )
    listeners.append((object, property, listener))
  }

  func consume(into output: PCMOutput, expectedHostTime: inout Double?) throws {
    guard let converter else { return }
    while let frame = ring.peek() {
      try preserveGap(before: frame, output: output, expectedHostTime: &expectedHostTime)
      let converted = try converter.convert(frame.samples, count: frame.count)
      try output.append(converted)
      health.received(at: ProcessInfo.processInfo.systemUptime)
      ring.release()
    }
    let dropped = ring.droppedFrames
    if dropped != reportedDrops {
      note(
        "capture overflow: \(dropped - reportedDrops) frames dropped; gap follows capture timestamps"
      )
      reportedDrops = dropped
    }
  }

  private func preserveGap(
    before frame: AudioRing.Frame, output: PCMOutput, expectedHostTime: inout Double?
  ) throws {
    let hostTime = Double(AudioConvertHostTimeToNanos(frame.hostTime)) / 1_000_000_000
    defer { expectedHostTime = hostTime + Double(frame.count) / inputRate }
    guard let expectedHostTime, hostTime - expectedHostTime > 0.1 else { return }
    let gap = hostTime - expectedHostTime
    note("capture gap: \(String(format: "%.3f", gap)) seconds")
    // A large discontinuity needs recovery instead of an unbounded silence allocation.
    guard gap <= 3 else { throw CaptureFailure("capture timestamp gap exceeds three seconds") }
    var remaining = Int(gap * sampleRate)
    while remaining > 0 {
      let count = min(remaining, 1600)
      try output.append([Float](repeating: 0, count: count))
      remaining -= count
    }
  }

  func announce() {
    announced = true
    note("capturing")
  }

  func stop() {
    for (object, storedProperty, listener) in listeners {
      var property = storedProperty
      let status = AudioObjectRemovePropertyListenerBlock(object, &property, .main, listener)
      if status != noErr { note("remove device listener failed (OSStatus \(status))") }
    }
    listeners.removeAll()
    stopCallback()
    destroyDevices()
  }

  private func stopCallback() {
    guard let procID else { return }
    let stopStatus = AudioDeviceStop(aggregateID, procID)
    if stopStatus != noErr { note("AudioDeviceStop failed (OSStatus \(stopStatus))") }
    let destroyStatus = AudioDeviceDestroyIOProcID(aggregateID, procID)
    if destroyStatus != noErr {
      note("AudioDeviceDestroyIOProcID failed (OSStatus \(destroyStatus))")
    }
    if stopStatus == noErr || destroyStatus == noErr { callbackStopped = true }
    if destroyStatus == noErr { self.procID = nil }
  }

  private func destroyDevices() {
    if aggregateID != kAudioObjectUnknown {
      let status = AudioHardwareDestroyAggregateDevice(aggregateID)
      if status != noErr { note("AudioHardwareDestroyAggregateDevice failed (OSStatus \(status))") }
      if status == noErr {
        aggregateID = AudioObjectID(kAudioObjectUnknown)
        procID = nil
        callbackStopped = true
      }
    }
    if tapID != kAudioObjectUnknown {
      let status = AudioHardwareDestroyProcessTap(tapID)
      if status != noErr { note("AudioHardwareDestroyProcessTap failed (OSStatus \(status))") }
      if status == noErr { tapID = AudioObjectID(kAudioObjectUnknown) }
    }
  }
}

@available(macOS 14.2, *)
final class CaptureRunner {
  private let output: PCMOutput
  private var session: CaptureSession?
  private var attempts = 0
  private var retryAt = Double(0)
  private var expectedHostTime: Double?

  init() throws { output = try PCMOutput(descriptor: STDOUT_FILENO) }

  func run() throws {
    defer {
      session?.stop()
      stdinWatcher.cancel()
    }
    while !shouldStop() {
      try step()
      RunLoop.current.run(until: Date(timeIntervalSinceNow: 0.005))
    }
    session?.stop()
    session = nil
    try output.flush()
    if output.pendingBytes > 0 { throw CaptureFailure("stdout stalled during shutdown") }
  }

  private func step() throws {
    let now = ProcessInfo.processInfo.systemUptime
    if session == nil && now >= retryAt { try start(at: now) }
    guard let session else { return }
    try session.consume(into: output, expectedHostTime: &expectedHostTime)
    if session.health.confirmed && !session.announced {
      session.announce()
      attempts = 0
    }
    if session.health.expired(at: now) { try rebuild(at: now, reason: "no fresh audio buffers") }
    if let deadline = session.rebuildAt, now >= deadline {
      try rebuild(at: now, reason: "device or format changed")
    }
    try output.flush()
  }

  private func start(at time: Double) throws {
    guard attempts < 3 else {
      throw CaptureFailure("capture did not deliver buffers after three attempts")
    }
    attempts += 1
    let fresh = try CaptureSession()
    do {
      try fresh.start()
      session = fresh
    } catch {
      fresh.stop()
      note("capture start attempt \(attempts) failed: \(error)")
      retryAt = time + 0.3
    }
  }

  private func rebuild(at time: Double, reason: String) throws {
    note("rebuilding capture: \(reason)")
    session?.stop()
    session = nil
    retryAt = time + 0.3
  }
}

if #available(macOS 14.2, *) {
  if CommandLine.arguments.contains("--ensure-permission") {
    ensurePermission()
  }
  if !shouldStop() {
    do { try CaptureRunner().run() } catch { fail(String(describing: error)) }
  }
  stdinWatcher.cancel()
  for source in stopSignals { source.cancel() }
} else {
  fail("Core Audio process taps require macOS 14.2 or newer")
}
