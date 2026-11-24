/// One stretch of speech by one anonymous speaker, in seconds from the recording start.
///
/// The speaker number is local to one diarization run. It is not an identity.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SpeakerTurn {
    pub start: f64,
    pub end: f64,
    pub speaker: u32,
}
