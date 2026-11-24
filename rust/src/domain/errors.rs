use thiserror::Error;

/// A whisper model name that does not resolve to a local file.
#[derive(Debug, Error)]
#[error("{0}")]
pub struct ModelResolutionError(pub String);
