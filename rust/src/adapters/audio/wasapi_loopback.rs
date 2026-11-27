//! The system audio on Windows: WASAPI loopback of the default output.
//!
//! The loopback client is the next step. Until it lands, preflight reports the
//! failed system audio check and the app records the microphone only.

use std::time::Duration;

use crate::ports::audio_source::{AudioError, AudioSource};

pub struct WasapiLoopbackSource;

impl WasapiLoopbackSource {
    pub fn new() -> Self {
        Self
    }
}

impl Default for WasapiLoopbackSource {
    fn default() -> Self {
        Self::new()
    }
}

impl AudioSource for WasapiLoopbackSource {
    fn open(&self, _sample_rate: u32, _channels: u16) -> Result<(), AudioError> {
        Err(AudioError(
            "WASAPI loopback capture is not implemented yet".into(),
        ))
    }

    fn read(&self, _timeout: Duration) -> Result<Option<Vec<f32>>, AudioError> {
        Ok(None)
    }

    fn close(&self) -> Result<(), AudioError> {
        Ok(())
    }

    fn continuous_capture(&self) -> bool {
        true
    }
}
