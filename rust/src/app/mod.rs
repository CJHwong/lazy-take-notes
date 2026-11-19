//! The runtime around a session: worker threads, their events, and headless mode.

pub mod cli;
pub mod container;
pub mod diarize_command;
pub mod download_progress;
pub mod events;
pub mod headless;
pub mod headless_run;
pub mod keep_awake;
pub mod logging;
pub mod preflight;
pub mod shell;
pub mod status_command;
pub mod tui_run;
pub mod workers;
