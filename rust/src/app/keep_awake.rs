//! Keep the machine awake while a live recording runs.
//!
//! `caffeinate -w <pid>` ties the assertion to this process, so it ends even
//! when the process is killed without a clean shutdown. On Linux,
//! `systemd-inhibit` blocks idle and sleep for the length of the child. On
//! Windows, `SetThreadExecutionState` holds the display and the system awake.

use tracing::debug;

#[cfg(unix)]
use std::os::unix::process::CommandExt;
#[cfg(unix)]
use std::process::{Child, Command, Stdio};
#[cfg(windows)]
use windows_sys::Win32::System::Power::{
    ES_CONTINUOUS, ES_DISPLAY_REQUIRED, ES_SYSTEM_REQUIRED, SetThreadExecutionState,
};

/// Holds the machine awake until dropped.
pub struct KeepAwake {
    #[cfg(unix)]
    child: Option<Child>,
}

#[cfg(unix)]
fn command() -> Command {
    #[cfg(target_os = "macos")]
    {
        let mut command = Command::new("caffeinate");
        command.args(["-di", "-w", &std::process::id().to_string()]);
        command
    }
    #[cfg(target_os = "linux")]
    {
        let mut command = Command::new("systemd-inhibit");
        command.args([
            "--what=idle:sleep",
            "--who=lazy-take-notes",
            "--why=Recording",
            "--mode=block",
            "sleep",
            "infinity",
        ]);
        command
    }
}

pub fn keep_awake() -> KeepAwake {
    #[cfg(unix)]
    {
        let child = command()
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            // Its own process group: a terminal Ctrl-C must not end it before the
            // final digest. It ends with this process, or when dropped.
            .process_group(0)
            .spawn()
            .ok();
        if child.is_none() {
            debug!("no idle inhibitor found; the machine may sleep during a recording");
        }
        KeepAwake { child }
    }
    #[cfg(windows)]
    {
        let flags = ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED;
        // SAFETY: the flags are valid EXECUTION_STATE bits.
        if unsafe { SetThreadExecutionState(flags) } == 0 {
            debug!("no idle inhibitor found; the machine may sleep during a recording");
        }
        KeepAwake {}
    }
}

impl Drop for KeepAwake {
    fn drop(&mut self) {
        #[cfg(unix)]
        {
            let Some(mut child) = self.child.take() else {
                return;
            };
            if let Ok(pid) = i32::try_from(child.id()) {
                // SAFETY: the pid is our own live child; SIGTERM as Python's terminate().
                unsafe { libc::kill(pid, libc::SIGTERM) };
            }
            let _ = child.wait();
        }
        #[cfg(windows)]
        {
            // SAFETY: clearing the flags ends the assertion.
            unsafe { SetThreadExecutionState(ES_CONTINUOUS) };
        }
    }
}
