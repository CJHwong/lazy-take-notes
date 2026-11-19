//! The runtime around a session: worker threads, their events, and headless mode.

pub mod cli;
pub mod diarize_command;
pub mod download_progress;
pub mod events;
pub mod preflight;
pub mod workers;
