//! Status record of a headless session, persisted so the CLI can check in.

use serde::{Deserialize, Serialize};

pub const STARTING: &str = "starting";
pub const LOADING_MODEL: &str = "loading_model";
pub const DOWNLOADING: &str = "downloading";
pub const RECORDING: &str = "recording";
pub const DIGESTING: &str = "digesting";
pub const STOPPED: &str = "stopped";
pub const ERROR: &str = "error";

/// States that need a live process. A dead PID in one of them means the session crashed.
pub const LIVE_STATES: [&str; 5] = [STARTING, LOADING_MODEL, DOWNLOADING, RECORDING, DIGESTING];

/// Snapshot of a headless session's progress. The field order is the JSON key order.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionStatus {
    pub state: String,
    pub pid: u32,
    pub started_at: String,
    pub updated_at: String,
    #[serde(default)]
    pub segment_count: u64,
    #[serde(default)]
    pub digest_count: u64,
    #[serde(default)]
    pub error: String,
}

impl SessionStatus {
    pub fn is_live_state(&self) -> bool {
        LIVE_STATES.contains(&self.state.as_str())
    }
}
