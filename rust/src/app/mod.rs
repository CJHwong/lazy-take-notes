//! The runtime around a session: worker threads, their events, and headless mode.

pub mod container;
pub mod diarize_command;
pub mod download_progress;
pub mod events;
pub mod headless;
pub mod headless_run;
pub mod keep_awake;
pub mod logging;
pub mod logo;
pub mod modes;
pub mod notices;
pub mod preflight;
pub mod preflight_runtime;
pub mod saved_session;
pub mod session;
pub mod session_commands;
pub mod session_list;
pub mod session_start;
pub mod settings;
pub mod status_command;
pub mod tasks;
pub mod template_builder;
pub mod template_catalog;
pub mod theme;
pub mod workers;
