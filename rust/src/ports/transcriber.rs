use thiserror::Error;

use crate::domain::transcript::TranscriptSegment;

#[derive(Debug, Error)]
#[error("{0}")]
pub struct TranscribeError(pub String);

/// Speech to text. Segment times are seconds into the audio it was given.
pub trait Transcriber: Send {
    fn load_model(&mut self, model_path: &str) -> Result<(), TranscribeError>;

    fn transcribe(
        &mut self,
        audio: &[f32],
        language: &str,
        hints: &[String],
    ) -> Result<Vec<TranscriptSegment>, TranscribeError>;

    fn close(&mut self) {}
}
