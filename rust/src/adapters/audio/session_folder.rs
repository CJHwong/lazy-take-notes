//! The session folder, shared by the app and the WAV writer.
//!
//! The user can rename the folder while it records. The writer checks that its
//! file is still at its path, to catch Move to Trash, so a rename it does not
//! know about looks like a removed file and starts a second recording. A rename
//! through this object holds the writer's lock and moves the writer's path too.

use std::path::{Path, PathBuf};
use std::sync::{Condvar, Mutex, MutexGuard};
use std::time::Duration;

use super::wav::OsError;

/// The writer holds the lock for one chunk. Longer means the disk is stalled.
pub const RENAME_WAIT: Duration = Duration::from_secs(2);

const EBUSY: i32 = 16;

struct State {
    path: PathBuf,
    held: bool,
}

pub struct SessionFolder {
    state: Mutex<State>,
    released: Condvar,
}

/// Exclusive use of the folder. The path cannot change while this lives.
pub struct FolderGuard<'a> {
    folder: &'a SessionFolder,
}

impl FolderGuard<'_> {
    pub fn path(&self) -> PathBuf {
        self.folder.lock_state().path.clone()
    }
}

impl Drop for FolderGuard<'_> {
    fn drop(&mut self) {
        self.folder.lock_state().held = false;
        self.folder.released.notify_one();
    }
}

impl SessionFolder {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            state: Mutex::new(State {
                path: path.into(),
                held: false,
            }),
            released: Condvar::new(),
        }
    }

    pub fn path(&self) -> PathBuf {
        self.lock_state().path.clone()
    }

    fn lock_state(&self) -> MutexGuard<'_, State> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Wait for the folder as long as it takes.
    pub fn hold(&self) -> FolderGuard<'_> {
        let state = self.lock_state();
        let mut state = self
            .released
            .wait_while(state, |s| s.held)
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state.held = true;
        FolderGuard { folder: self }
    }

    fn try_hold(&self, wait: Duration) -> Option<FolderGuard<'_>> {
        let state = self.lock_state();
        let (mut state, _) = self
            .released
            .wait_timeout_while(state, wait, |s| s.held)
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if state.held {
            return None;
        }
        state.held = true;
        Some(FolderGuard { folder: self })
    }

    /// Rename the folder. Called on the UI thread, so a stalled disk must not freeze the screen.
    pub fn rename(&self, new_path: &Path) -> Result<(), OsError> {
        let Some(_guard) = self.try_hold(RENAME_WAIT) else {
            return Err(OsError::new(
                EBUSY,
                "the disk is busy writing the recording; try again",
                None,
            ));
        };
        let old_path = self.path();
        std::fs::rename(&old_path, new_path)
            .map_err(|error| OsError::from_io(&error, Some(&old_path), Some(new_path)))?;
        self.lock_state().path = new_path.to_path_buf();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_rename_moves_the_folder_and_the_path() {
        let root = tempfile::tempdir().unwrap();
        let folder = SessionFolder::new(root.path().join("a"));
        std::fs::create_dir(folder.path()).unwrap();
        folder.rename(&root.path().join("b")).unwrap();
        assert_eq!(folder.path(), root.path().join("b"));
        assert!(root.path().join("b").is_dir());
    }

    #[test]
    fn a_rename_gives_up_while_the_writer_holds_the_folder() {
        let folder = SessionFolder::new("/nowhere");
        let _writer = folder.hold();
        let error = folder.try_hold(Duration::from_millis(20)).map(|_| ());
        assert!(error.is_none());
        let message = OsError::new(
            EBUSY,
            "the disk is busy writing the recording; try again",
            None,
        )
        .to_string();
        assert_eq!(
            message,
            "[Errno 16] the disk is busy writing the recording; try again"
        );
    }

    #[test]
    fn a_failed_rename_reads_like_python() {
        let root = tempfile::tempdir().unwrap();
        let folder = SessionFolder::new(root.path().join("missing"));
        let error = folder
            .rename(&root.path().join("b"))
            .unwrap_err()
            .to_string();
        let expected = format!(
            "[Errno 2] No such file or directory: '{}' -> '{}'",
            root.path().join("missing").display(),
            root.path().join("b").display()
        );
        assert_eq!(error, expected);
    }
}
