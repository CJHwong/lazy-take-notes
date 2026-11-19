//! An exclusive lock on a file, across processes and across open files in one process.
//!
//! Two downloads of one asset write the same `.part` file, and both then fail
//! their digest check. The kiosk runs the preflight and Settings in one process
//! with two stores, and a second terminal can run another session, so a thread
//! lock is not enough. `flock()` locks the open file, so two opens in one process
//! also conflict; POSIX `lockf()` would not. The Python version uses the same lock
//! files, so the two versions also exclude each other.

use std::fs::{self, File, OpenOptions};
use std::io;
use std::os::fd::AsRawFd;
use std::os::unix::fs::OpenOptionsExt;
use std::path::Path;
use std::thread;
use std::time::Duration;

pub const POLL: Duration = Duration::from_millis(200);

/// A held lock. Dropping it unlocks.
#[derive(Debug)]
pub struct LockGuard {
    file: File,
}

impl Drop for LockGuard {
    fn drop(&mut self) {
        // SAFETY: the descriptor belongs to `self.file`, which is still open.
        unsafe { libc::flock(self.file.as_raw_fd(), libc::LOCK_UN) };
    }
}

fn open(path: &Path) -> io::Result<File> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .mode(0o644)
        .open(path)
}

fn try_lock(file: &File) -> io::Result<bool> {
    // SAFETY: flock on a valid open descriptor.
    if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } == 0 {
        return Ok(true);
    }
    let error = io::Error::last_os_error();
    if error.raw_os_error() == Some(libc::EWOULDBLOCK) {
        Ok(false)
    } else {
        Err(error)
    }
}

/// Wait for the lock on `path`. While another holder has it, call `on_wait` every `POLL`.
/// Fails with `Interrupted` when `cancelled` turns true during the wait.
pub fn exclusive(
    path: &Path,
    cancelled: &dyn Fn() -> bool,
    on_wait: &mut dyn FnMut(),
) -> io::Result<LockGuard> {
    let file = open(path)?;
    while !try_lock(&file)? {
        if cancelled() {
            let name = path.file_name().unwrap_or_default().to_string_lossy();
            return Err(io::Error::new(
                io::ErrorKind::Interrupted,
                format!("cancelled while waiting for {name}"),
            ));
        }
        on_wait();
        thread::sleep(POLL);
    }
    Ok(LockGuard { file })
}

/// The lock on `path` if it is free right now, else None. Never waits.
pub fn try_exclusive(path: &Path) -> io::Result<Option<LockGuard>> {
    let file = open(path)?;
    Ok(try_lock(&file)?.then_some(LockGuard { file }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_second_open_in_the_same_process_is_locked_out_until_drop() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("locks/a.lock");
        let held = try_exclusive(&path).unwrap().expect("free");
        assert!(try_exclusive(&path).unwrap().is_none());
        let waits = std::cell::Cell::new(0);
        let error = exclusive(&path, &|| waits.get() > 0, &mut || {
            waits.set(waits.get() + 1)
        })
        .unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::Interrupted);
        drop(held);
        assert!(try_exclusive(&path).unwrap().is_some());
    }

    #[test]
    fn exclusive_waits_for_the_holder() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("b.lock");
        let held = try_exclusive(&path).unwrap().unwrap();
        let releaser = thread::spawn(move || {
            thread::sleep(Duration::from_millis(300));
            drop(held);
        });
        let mut waits = 0;
        exclusive(&path, &|| false, &mut || waits += 1).unwrap();
        releaser.join().unwrap();
        assert!(waits >= 1);
    }
}
