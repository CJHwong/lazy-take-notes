//! Business rules. They reach the outside world only through `crate::ports`.

pub mod compact;
pub mod digest;
pub mod label;
pub mod prompt_builder;
pub mod pyformat;
pub mod quick_action;
pub mod speaker_attribution;
pub mod template_builder;
pub mod template_diff;
pub mod template_validator;
pub mod transcribe_audio;
