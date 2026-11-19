//! Keep the Mac awake while a live recording runs.
//!
//! `caffeinate -w <pid>` ties the assertion to this process, so it ends even
//! when the process is killed without a clean shutdown.

use std::os::unix::process::CommandExt;
use std::process::{Child, Command, Stdio};

/// Holds the machine awake until dropped.
pub struct KeepAwake(Option<Child>);

pub fn keep_awake() -> KeepAwake {
    let child = Command::new("caffeinate")
        .args(["-di", "-w", &std::process::id().to_string()])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        // Its own process group: a terminal Ctrl-C must not end it before the
        // final digest. It ends with this process, or when dropped.
        .process_group(0)
        .spawn()
        .ok();
    KeepAwake(child)
}

impl Drop for KeepAwake {
    fn drop(&mut self) {
        let Some(mut child) = self.0.take() else {
            return;
        };
        if let Ok(pid) = i32::try_from(child.id()) {
            // SAFETY: the pid is our own live child; SIGTERM as Python's terminate().
            unsafe { libc::kill(pid, libc::SIGTERM) };
        }
        let _ = child.wait();
    }
}
