//! Audio capture: the microphone, the macOS system audio tap, the mix of both, and the session WAV.

pub mod buffer;
#[cfg(target_os = "macos")]
pub mod coreaudio_tap;
pub mod mic_boost;
pub mod microphone;
pub mod mixed;
#[cfg(test)]
mod mixed_tests;
#[cfg(target_os = "linux")]
pub mod pulse_monitor;
#[cfg(all(test, target_os = "linux"))]
mod pulse_monitor_tests;
pub mod rate;
pub mod session_folder;
pub mod wav;
