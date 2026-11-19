use std::time::Duration;

use thiserror::Error;

#[derive(Debug, Clone, Error, PartialEq, Eq)]
#[error("{0}")]
pub struct AudioError(pub String);

/// A mono float32 audio input stream.
///
/// Every method takes `&self`: the mixer reads a source on one thread while it
/// reopens, drains or closes it on another, and a reader stuck inside `read`
/// must never block `close`. Each source synchronizes itself.
pub trait AudioSource: Send + Sync {
    fn open(&self, sample_rate: u32, channels: u16) -> Result<(), AudioError>;

    /// The next block of samples, or None when nothing arrived within `timeout`.
    fn read(&self, timeout: Duration) -> Result<Option<Vec<f32>>, AudioError>;

    /// Discard buffered audio. A source with no buffer does nothing.
    fn drain(&self) {}

    fn close(&self) -> Result<(), AudioError>;

    /// Silence the microphone part of the stream. A source without a mic ignores it.
    fn set_mic_muted(&self, _muted: bool) {}

    /// The stream ended and will deliver nothing more until it is opened again.
    fn exhausted(&self) -> bool {
        false
    }

    /// The device delivers buffers all the time, silent or not, so a gap in delivery means it is stuck.
    fn continuous_capture(&self) -> bool {
        false
    }

    /// The source waits for the user to grant capture access.
    fn awaiting_permission(&self) -> bool {
        false
    }

    /// Whether a non-zero system sample arrived. A source that cannot tell counts as seen.
    fn system_signal_seen(&self) -> bool {
        true
    }

    /// Set when the configured device is missing and another one records.
    fn device_notice(&self) -> String {
        String::new()
    }

    /// Samples dropped to keep capture delay bounded.
    fn dropped_frames(&self) -> u64 {
        0
    }
}
