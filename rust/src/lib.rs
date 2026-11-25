//! lazy-take-notes: live transcription with a rolling AI digest.

pub mod adapters;
pub mod app;
pub mod cli;
pub mod config;
pub mod domain;
#[cfg(feature = "gui")]
pub mod gui;
pub mod ports;
pub mod tui;
pub mod usecases;
