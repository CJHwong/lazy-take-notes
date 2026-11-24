//! Pure domain types. No I/O, no framework types.

pub mod chat;
pub mod config;
pub mod digest_state;
pub mod errors;
pub mod model_asset;
pub mod session_files;
pub mod session_status;
pub mod speaker_turn;
pub mod template;
pub mod transcript;

/// Every audio buffer in the app runs at this rate, mono.
pub const SAMPLE_RATE: usize = 16_000;
