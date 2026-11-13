import AVFoundation
import AudioRing
import CaptureSupport
import Darwin
import XCTest

final class CaptureSupportTests: XCTestCase {
  func testPermissionWaitStopsBeforeItsDeadline() {
    let started = ProcessInfo.processInfo.systemUptime
    let result = waitForCompletion(DispatchSemaphore(value: 0), timeout: 10) {
      ProcessInfo.processInfo.systemUptime - started > 0.02
    }
    XCTAssertFalse(result)
    XCTAssertLessThan(ProcessInfo.processInfo.systemUptime - started, 0.5)
    let completed = DispatchSemaphore(value: 0)
    completed.signal()
    XCTAssertTrue(waitForCompletion(completed, timeout: 1, shouldStop: { false }))
  }

  func testRingKeepsReadSlotUntilRelease() throws {
    let ring = try XCTUnwrap(AudioRing(capacity: 2, maximumFrames: 4))
    var first: [Float] = [1, 2, 3, 4]
    var second: [Float] = [5, 6, 7, 8]
    XCTAssertTrue(ring.enqueue(&first, count: 4, hostTime: 100))
    XCTAssertTrue(ring.enqueue(&second, count: 4, hostTime: 200))
    let entry = try XCTUnwrap(ring.peek())
    XCTAssertEqual(entry.hostTime, 100)
    XCTAssertEqual(entry.count, 4)
    XCTAssertFalse(ring.enqueue(&second, count: 4, hostTime: 300))
    XCTAssertEqual(Array(UnsafeBufferPointer(start: entry.samples, count: 4)), first)
    ring.release()
    XCTAssertTrue(ring.enqueue(&first, count: 4, hostTime: 300))
    XCTAssertEqual(try XCTUnwrap(ring.peek()).hostTime, 200)
    XCTAssertEqual(ring.droppedFrames, 4)
  }

  func testRingRejectsOversizedBuffersAndWraps() throws {
    let ring = try XCTUnwrap(AudioRing(capacity: 2, maximumFrames: 4))
    var samples = [Float](repeating: 0.25, count: 5)
    XCTAssertFalse(ring.enqueue(&samples, count: 5, hostTime: 0))
    XCTAssertNil(ring.peek())
    for timestamp in 1...100 {
      XCTAssertTrue(ring.enqueue(&samples, count: 4, hostTime: UInt64(timestamp)))
      XCTAssertEqual(try XCTUnwrap(ring.peek()).hostTime, UInt64(timestamp))
      ring.release()
    }
    XCTAssertEqual(ring.droppedFrames, 5)
  }

  func testRingRejectsInvalidCapacityAndEmptyBuffers() throws {
    XCTAssertNil(AudioRing(capacity: 0, maximumFrames: 4))
    XCTAssertNil(AudioRing(capacity: -1, maximumFrames: 4))
    XCTAssertNil(AudioRing(capacity: 4, maximumFrames: 0))
    XCTAssertNil(AudioRing(capacity: 4, maximumFrames: -1))
    XCTAssertNil(AudioRing(capacity: Int.max, maximumFrames: 2))
    XCTAssertNil(AudioRing(capacity: 1, maximumFrames: Int.max))
    let ring = try XCTUnwrap(AudioRing(capacity: 1, maximumFrames: 4))
    var samples: [Float] = [1, 2, 3, 4]
    XCTAssertFalse(ring.enqueue(&samples, count: 0, hostTime: 100))
    XCTAssertFalse(ring.enqueue(&samples, count: -1, hostTime: 100))
    XCTAssertNil(ring.peek())
    XCTAssertEqual(ring.droppedFrames, 0)
    ring.release()
    XCTAssertTrue(ring.enqueue(&samples, count: 4, hostTime: 200))
    XCTAssertEqual(try XCTUnwrap(ring.peek()).hostTime, 200)
  }

  func testRingSupportsNonPowerOfTwoCapacity() throws {
    let ring = try XCTUnwrap(AudioRing(capacity: 3, maximumFrames: 1))
    var sample: Float = 0.5
    for timestamp in 0..<100 {
      XCTAssertTrue(ring.enqueue(&sample, count: 1, hostTime: UInt64(timestamp)))
      XCTAssertEqual(try XCTUnwrap(ring.peek()).hostTime, UInt64(timestamp))
      ring.release()
    }
  }

  func testConverterPreservesDurationAndFiniteSamples() throws {
    let converter = try PCMConverter(sampleRate: 48_000, maximumFrames: 4800)
    var samples = (0..<4800).map { Float(sin(Double($0) * 2 * .pi * 440 / 48_000)) }
    var output: [Float] = []
    for _ in 0..<10 {
      let converted = try converter.convert(&samples, count: samples.count)
      output += converted
    }
    XCTAssertLessThan(abs(output.count - 16_000), 100)
    XCTAssertTrue(output.allSatisfy(\.isFinite))
    XCTAssertGreaterThan(output.map { abs($0) }.max() ?? 0, 0.5)
  }

  func testConverterKeepsSmallAndNonIntegerRateChunks() throws {
    for rate in [16_000.0, 44_100.0, 96_000.0] {
      let frames = Int(rate / 100)
      let converter = try PCMConverter(sampleRate: rate, maximumFrames: frames)
      var samples = [Float](repeating: 0.25, count: frames)
      var produced = 0
      for _ in 0..<100 { produced += try converter.convert(&samples, count: frames).count }
      XCTAssertLessThan(abs(produced - 16_000), 100)
    }
  }

  func testRingConcurrentProducerCannotOverwriteConsumer() throws {
    for capacity in [1, 3, 64] { try checkConcurrentRing(capacity: capacity) }
  }

  private func checkConcurrentRing(capacity: Int) throws {
    let ring = try XCTUnwrap(AudioRing(capacity: capacity, maximumFrames: 4))
    for sequence in 0..<capacity {
      var samples = [Float](repeating: Float(sequence), count: 4)
      XCTAssertTrue(ring.enqueue(&samples, count: 4, hostTime: UInt64(sequence)))
    }
    var overflow = [Float](repeating: -1, count: 4)
    XCTAssertFalse(ring.enqueue(&overflow, count: 4, hostTime: UInt64(capacity)))
    XCTAssertEqual(ring.droppedFrames, 4)
    let producer = DispatchGroup()
    producer.enter()
    // Dispatch transfers the producer role after the initial full-ring check.
    DispatchQueue.global().async {
      for sequence in capacity..<10_000 {
        var samples = [Float](repeating: Float(sequence), count: 4)
        while !ring.enqueue(&samples, count: 4, hostTime: UInt64(sequence)) { sched_yield() }
      }
      producer.leave()
    }
    for sequence in 0..<10_000 {
      var entry = ring.peek()
      while entry == nil {
        sched_yield()
        entry = ring.peek()
      }
      let frame = try XCTUnwrap(entry)
      XCTAssertEqual(frame.hostTime, UInt64(sequence), "capacity \(capacity)")
      XCTAssertEqual(
        Array(UnsafeBufferPointer(start: frame.samples, count: 4)),
        [Float](repeating: Float(sequence), count: 4), "capacity \(capacity)")
      ring.release()
    }
    XCTAssertEqual(producer.wait(timeout: .now() + 1), .success)
  }

  func testClosedPipeReportsFailure() throws {
    var descriptors: [Int32] = [0, 0]
    XCTAssertEqual(pipe(&descriptors), 0)
    let previous = signal(SIGPIPE, SIG_IGN)
    defer {
      close(descriptors[1])
      signal(SIGPIPE, previous)
    }
    close(descriptors[0])
    let output = try PCMOutput(descriptor: descriptors[1])
    XCTAssertThrowsError(try output.append([0.25, 0.5]))
  }

  func testSupportedGapFitsBeforeHealthyConsumerRuns() throws {
    var descriptors: [Int32] = [0, 0]
    XCTAssertEqual(pipe(&descriptors), 0)
    defer {
      close(descriptors[0])
      close(descriptors[1])
    }
    let output = try PCMOutput(descriptor: descriptors[1])
    let silence = [Float](repeating: 0, count: 1600)
    for _ in 0..<30 { try output.append(silence) }
    try output.append([Float](repeating: 0.25, count: 1600))
    XCTAssertEqual(output.droppedFrames, 0)
    XCTAssertLessThanOrEqual(output.pendingBytes, 256_000)
    XCTAssertEqual(fcntl(descriptors[0], F_SETFL, O_NONBLOCK), 0)
    var received = [UInt8](repeating: 0, count: 65_536)
    var total = 0
    for _ in 0..<100 {
      let count = read(descriptors[0], &received, received.count)
      if count > 0 { total += count }
      try output.flush()
    }
    XCTAssertEqual(total, 31 * 1600 * MemoryLayout<Float>.size)
    XCTAssertEqual(output.pendingBytes, 0)
  }

  func testPipeStallRemainsBoundedAndFlushesWithoutCorruption() throws {
    var descriptors: [Int32] = [0, 0]
    XCTAssertEqual(pipe(&descriptors), 0)
    defer {
      close(descriptors[0])
      close(descriptors[1])
    }
    let output = try PCMOutput(descriptor: descriptors[1], maximumPendingBytes: 16_000)
    let samples = [Float](repeating: 0.25, count: 1600)
    var rejected = false
    for _ in 0..<100 {
      do { try output.append(samples) } catch {
        rejected = true
        break
      }
    }
    XCTAssertTrue(rejected)
    XCTAssertLessThanOrEqual(output.pendingBytes, 16_000)
    XCTAssertGreaterThan(output.droppedFrames, 0)
    XCTAssertEqual(output.pendingBytes % MemoryLayout<Float>.size, 0)
    XCTAssertEqual(fcntl(descriptors[0], F_SETFL, O_NONBLOCK), 0)
    var received = [UInt8](repeating: 0, count: 65_536)
    while read(descriptors[0], &received, received.count) > 0 {}
    try output.flush()
    let count = read(descriptors[0], &received, received.count)
    XCTAssertGreaterThan(count, 0)
    let values = received.prefix(count).withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    XCTAssertTrue(values.allSatisfy { $0 == 0.25 })
  }

  func testHealthRequiresFramesAndDoesNotInspectAmplitude() {
    var health = CaptureHealth(startedAt: 10)
    XCTAssertFalse(health.confirmed)
    XCTAssertFalse(health.expired(at: 12))
    XCTAssertTrue(health.expired(at: 13.1))
    health.received(at: 13.2)
    XCTAssertTrue(health.confirmed)
    XCTAssertFalse(health.expired(at: 16))
    XCTAssertTrue(health.expired(at: 16.3))
  }
}
