import Atomics

/// One producer calls enqueue. One consumer calls peek and release.
/// Acquire/release publication protects the raw sample and header storage.
/// The caller stops both roles before releasing its last ring reference.
public final class AudioRing: @unchecked Sendable {
  public struct Frame {
    /// The consumer keeps the ring alive and finishes reading before release.
    public let samples: UnsafePointer<Float>
    public let count: Int
    public let hostTime: UInt64
  }

  private struct Header {
    var count = 0
    var hostTime: UInt64 = 0
  }

  private let capacity: Int
  private let maximumFrames: Int
  private let cycleLength: UInt64
  private let sampleStorage: UnsafeMutablePointer<Float>
  private let headers: UnsafeMutablePointer<Header>
  private let sampleCount: Int
  private let head = ManagedAtomic<UInt64>(0)
  private let tail = ManagedAtomic<UInt64>(0)
  private let dropped = ManagedAtomic<UInt64>(0)

  public init?(capacity: Int, maximumFrames: Int) {
    guard capacity > 0, maximumFrames > 0 else { return nil }
    let (sampleCount, countOverflow) = capacity.multipliedReportingOverflow(by: maximumFrames)
    let (_, byteOverflow) = sampleCount.multipliedReportingOverflow(by: MemoryLayout<Float>.stride)
    let (_, headerOverflow) = capacity.multipliedReportingOverflow(by: MemoryLayout<Header>.stride)
    guard !countOverflow, !byteOverflow, !headerOverflow else { return nil }
    self.capacity = capacity
    self.maximumFrames = maximumFrames
    self.sampleCount = sampleCount
    cycleLength = UInt64(capacity) * 2
    sampleStorage = .allocate(capacity: sampleCount)
    sampleStorage.initialize(repeating: 0, count: sampleCount)
    headers = .allocate(capacity: capacity)
    headers.initialize(repeating: Header(), count: capacity)
  }

  deinit {
    sampleStorage.deinitialize(count: sampleCount)
    sampleStorage.deallocate()
    headers.deinitialize(count: capacity)
    headers.deallocate()
  }

  public var droppedFrames: UInt64 { dropped.load(ordering: .relaxed) }

  /// The producer never allocates or waits. Full and oversized buffers count as drops.
  public func enqueue(_ samples: UnsafePointer<Float>, count: Int, hostTime: UInt64) -> Bool {
    guard count > 0 else { return false }
    let writePosition = head.load(ordering: .relaxed)
    let readPosition = tail.load(ordering: .acquiring)
    let occupied =
      writePosition >= readPosition
      ? writePosition - readPosition : cycleLength - readPosition + writePosition
    guard count <= maximumFrames, occupied < UInt64(capacity) else {
      dropped.wrappingIncrement(by: UInt64(count), ordering: .relaxed)
      return false
    }
    let index = Int(writePosition % UInt64(capacity))
    sampleStorage.advanced(by: index * maximumFrames).update(from: samples, count: count)
    headers[index] = Header(count: count, hostTime: hostTime)
    head.store(advance(writePosition), ordering: .releasing)
    return true
  }

  /// The slot remains reserved until this consumer calls release.
  public func peek() -> Frame? {
    let readPosition = tail.load(ordering: .relaxed)
    guard readPosition != head.load(ordering: .acquiring) else { return nil }
    let index = Int(readPosition % UInt64(capacity))
    let header = headers[index]
    return Frame(
      samples: UnsafePointer(sampleStorage.advanced(by: index * maximumFrames)),
      count: header.count, hostTime: header.hostTime)
  }

  public func release() {
    let readPosition = tail.load(ordering: .relaxed)
    guard readPosition != head.load(ordering: .acquiring) else { return }
    tail.store(advance(readPosition), ordering: .releasing)
  }

  private func advance(_ position: UInt64) -> UInt64 {
    // A bounded cycle also supports capacities that do not divide UInt64's range.
    position + 1 == cycleLength ? 0 : position + 1
  }
}
