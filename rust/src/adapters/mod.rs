//! Gateways to the file system and other outside systems. They implement `crate::ports`.

pub mod assets;
pub mod audio;
pub mod audio_file;
pub mod config_file;
pub mod file_persistence;
pub mod llm;
pub mod model_catalog;
pub mod nemo_diarizer;
pub mod os_error;
pub mod paths;
pub mod pyrepr;
pub mod session_controller;
pub mod session_reader;
pub mod session_status;
pub mod templates;
pub mod whisper;
pub mod yaml_io;
