use std::path::Path;
use std::time::Duration;

use crate::domain::speaker_turn::SpeakerTurn;

/// Splits a 16 kHz mono WAV recording into anonymous speaker turns.
pub trait Diarizer: Send + Sync {
    /// Fails when the backend is missing, fails, or runs past `timeout`.
    fn diarize(&self, audio_path: &Path, timeout: Duration) -> Result<Vec<SpeakerTurn>, String>;
}
