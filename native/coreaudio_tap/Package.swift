// swift-tools-version: 5.10
import PackageDescription

let package = Package(
  name: "coreaudio-tap",
  platforms: [.macOS(.v13)],
  dependencies: [
    .package(url: "https://github.com/apple/swift-atomics.git", exact: "1.3.1")
  ],
  targets: [
    .target(
      name: "AudioRing", dependencies: [.product(name: "Atomics", package: "swift-atomics")],
      path: "AudioRing"),
    .target(name: "CaptureSupport", path: "CaptureSupport"),
    .executableTarget(
      name: "coreaudio-tap",
      dependencies: [
        "AudioRing", "CaptureSupport", .product(name: "Atomics", package: "swift-atomics"),
      ],
      path: "Sources",
      linkerSettings: [
        .linkedFramework("AVFoundation"),
        .linkedFramework("CoreMedia"),
        .linkedFramework("ScreenCaptureKit"),
      ]
    ),
    .testTarget(
      name: "CaptureSupportTests", dependencies: ["CaptureSupport", "AudioRing"], path: "Tests"),
  ]
)
