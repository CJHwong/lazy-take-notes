import AVFoundation
import Darwin
import Foundation

public struct CaptureFailure: Error, CustomStringConvertible {
  public let description: String
  public init(_ description: String) { self.description = description }
}

public func waitForCompletion(
  _ semaphore: DispatchSemaphore, timeout: Double,
  shouldStop: () -> Bool
) -> Bool {
  let deadline = ProcessInfo.processInfo.systemUptime + timeout
  while !shouldStop() && ProcessInfo.processInfo.systemUptime < deadline {
    if semaphore.wait(timeout: .now() + 0.05) == .success { return true }
  }
  return false
}

public struct CaptureHealth {
  private var lastFrameAt: Double
  public private(set) var confirmed = false

  public init(startedAt: Double) { lastFrameAt = startedAt }
  public mutating func received(at time: Double) {
    lastFrameAt = time
    confirmed = true
  }
  public func expired(at time: Double) -> Bool { time - lastFrameAt > 3 }
}

public final class PCMConverter {
  private let converter: AVAudioConverter
  private let input: AVAudioPCMBuffer
  private let output: AVAudioPCMBuffer

  public init(sampleRate: Double, maximumFrames: Int) throws {
    guard sampleRate > 0,
      let inputFormat = AVAudioFormat(
        commonFormat: .pcmFormatFloat32, sampleRate: sampleRate,
        channels: 1, interleaved: false),
      let outputFormat = AVAudioFormat(
        commonFormat: .pcmFormatFloat32, sampleRate: 16_000,
        channels: 1, interleaved: false),
      let input = AVAudioPCMBuffer(
        pcmFormat: inputFormat, frameCapacity: AVAudioFrameCount(maximumFrames)),
      let output = AVAudioPCMBuffer(
        pcmFormat: outputFormat,
        frameCapacity: AVAudioFrameCount(Double(maximumFrames) * 16_000 / sampleRate) + 1024),
      let converter = AVAudioConverter(from: inputFormat, to: outputFormat)
    else { throw CaptureFailure("could not build the mono converter") }
    self.converter = converter
    converter.primeMethod = .none
    self.input = input
    self.output = output
  }

  public func convert(_ samples: UnsafePointer<Float>, count: Int) throws -> [Float] {
    guard count <= Int(input.frameCapacity), let inputSamples = input.floatChannelData?[0]
    else { throw CaptureFailure("input buffer exceeds converter capacity") }
    var consumed = 0
    var result: [Float] = []
    var status: AVAudioConverterOutputStatus
    repeat {
      var conversionError: NSError?
      output.frameLength = 0
      status = converter.convert(to: output, error: &conversionError) { requested, inputStatus in
        guard consumed < count else {
          inputStatus.pointee = .noDataNow
          return nil
        }
        // The converter can request less than one capture buffer. Preserve the remainder.
        let supplied = min(Int(requested), count - consumed)
        self.input.frameLength = AVAudioFrameCount(supplied)
        memcpy(inputSamples, samples.advanced(by: consumed), supplied * MemoryLayout<Float>.size)
        consumed += supplied
        inputStatus.pointee = .haveData
        return self.input
      }
      guard status != .error, conversionError == nil, let converted = output.floatChannelData?[0]
      else {
        throw CaptureFailure("sample conversion failed: \(String(describing: conversionError))")
      }
      result.append(
        contentsOf: UnsafeBufferPointer(start: converted, count: Int(output.frameLength)))
    } while status == .haveData
    return result
  }
}

public final class PCMOutput {
  private let descriptor: Int32
  private let maximumPendingBytes: Int
  private var pending = Data()
  private var offset = 0
  public private(set) var droppedFrames = 0
  public var pendingBytes: Int { pending.count - offset }

  public init(descriptor: Int32, maximumPendingBytes: Int = 256_000) throws {
    self.descriptor = descriptor
    self.maximumPendingBytes = maximumPendingBytes
    let flags = fcntl(descriptor, F_GETFL)
    guard flags >= 0, fcntl(descriptor, F_SETFL, flags | O_NONBLOCK) == 0
    else { throw CaptureFailure("could not make stdout nonblocking: \(errno)") }
  }

  public func append(_ samples: [Float]) throws {
    try flush()
    let byteCount = samples.count * MemoryLayout<Float>.size
    guard pendingBytes + byteCount <= maximumPendingBytes else {
      droppedFrames += samples.count
      throw CaptureFailure("stdout stalled; bounded output buffer exhausted")
    }
    if offset > 0 {
      pending.removeFirst(offset)
      offset = 0
    }
    samples.withUnsafeBytes { pending.append(contentsOf: $0) }
    try flush()
  }

  public func flush() throws {
    while offset < pending.count {
      let written = pending.withUnsafeBytes { bytes in
        Darwin.write(descriptor, bytes.baseAddress!.advanced(by: offset), bytes.count - offset)
      }
      if written > 0 {
        offset += written
        continue
      }
      if written < 0 && errno == EINTR { continue }
      if written < 0 && (errno == EAGAIN || errno == EWOULDBLOCK) { return }
      throw CaptureFailure("stdout write failed: \(errno)")
    }
    pending.removeAll(keepingCapacity: true)
    offset = 0
  }
}
