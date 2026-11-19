//! Gateways to the file system and other outside systems. They implement `crate::ports`.

pub mod config_file;
pub mod file_persistence;
pub mod model_catalog;
pub mod paths;
pub mod session_reader;
pub mod session_status;
pub mod templates;
pub mod yaml_io;
