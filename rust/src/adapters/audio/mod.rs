//! Audio capture: the microphone, the macOS system audio tap, the mix of both, and the session WAV.

pub mod buffer;
pub mod coreaudio_tap;
pub mod mic_boost;
pub mod microphone;
pub mod mixed;
#[cfg(test)]
mod mixed_tests;
pub mod rate;
pub mod session_folder;
pub mod wav;
