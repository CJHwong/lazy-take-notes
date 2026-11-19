use std::time::Duration;

use thiserror::Error;

#[derive(Debug, Error)]
#[error("{0}")]
pub struct AudioError(pub String);

/// A mono float32 audio input stream.
pub trait AudioSource: Send {
    fn open(&mut self, sample_rate: u32, channels: u16) -> Result<(), AudioError>;

    /// The next block of samples, or None when nothing arrived within `timeout`.
    fn read(&mut self, timeout: Duration) -> Result<Option<Vec<f32>>, AudioError>;

    /// Discard buffered audio. A source with no buffer does nothing.
    fn drain(&mut self) {}

    fn close(&mut self);

    /// Silence the microphone part of the stream. A source without a mic ignores it.
    fn set_mic_muted(&mut self, _muted: bool) {}
}
