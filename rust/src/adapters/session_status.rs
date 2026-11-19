//! The status file of a headless session, and the process check behind it.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use crate::domain::session_status::SessionStatus;

pub const STATUS_FILE: &str = ".status.json";

/// Write the status atomically: a reader never sees half a file.
pub fn write_status(session_dir: &Path, status: &SessionStatus) -> io::Result<PathBuf> {
    let path = session_dir.join(STATUS_FILE);
    let temp = session_dir.join(format!("{STATUS_FILE}.tmp"));
    fs::write(
        &temp,
        serde_json::to_string(status).map_err(io::Error::other)?,
    )?;
    fs::rename(&temp, &path)?;
    Ok(path)
}

/// The session's status, or None when it is missing or unreadable.
pub fn read_status(session_dir: &Path) -> Option<SessionStatus> {
    serde_json::from_str(&fs::read_to_string(session_dir.join(STATUS_FILE)).ok()?).ok()
}

/// True when `pid` names a running process, including one owned by another user.
pub fn is_pid_alive(pid: u32) -> bool {
    let Ok(pid) = i32::try_from(pid) else {
        return false;
    };
    if pid <= 0 {
        return false;
    }
    // SAFETY: signal 0 only checks that the process exists; it sends nothing.
    let result = unsafe { libc::kill(pid, 0) };
    result == 0 || io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_round_trips_and_reads_the_python_file() {
        let dir = tempfile::tempdir().unwrap();
        let status = SessionStatus {
            state: "recording".into(),
            pid: 42,
            started_at: "a".into(),
            updated_at: "b".into(),
            segment_count: 3,
            digest_count: 1,
            error: String::new(),
        };
        write_status(dir.path(), &status).unwrap();
        assert_eq!(read_status(dir.path()), Some(status));
        let python = r#"{"state": "stopped", "pid": 7, "started_at": "x", "updated_at": "y", "segment_count": 0, "digest_count": 0, "error": ""}"#;
        fs::write(dir.path().join(STATUS_FILE), python).unwrap();
        assert_eq!(read_status(dir.path()).unwrap().state, "stopped");
        fs::write(
            dir.path().join(STATUS_FILE),
            r#"{"state": "x", "pid": "7"}"#,
        )
        .unwrap();
        assert_eq!(read_status(dir.path()), None);
    }

    #[test]
    fn this_process_is_alive_and_pid_zero_is_not() {
        assert!(is_pid_alive(std::process::id()));
        assert!(!is_pid_alive(0));
        assert!(!is_pid_alive(u32::MAX));
    }
}
