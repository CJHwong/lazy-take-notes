//! File names of session artifacts, with the legacy name each one replaced.

use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SessionFile {
    pub name: &'static str,
    pub legacy: &'static str,
}

impl SessionFile {
    /// The current or legacy file in `directory`, whichever exists first.
    pub fn resolve(&self, directory: &Path) -> Option<PathBuf> {
        [self.name, self.legacy]
            .into_iter()
            .map(|name| directory.join(name))
            .find(|path| path.exists())
    }
}

pub const TRANSCRIPT: SessionFile = SessionFile {
    name: "transcript.txt",
    legacy: "transcript_raw.txt",
};
pub const NOTES: SessionFile = SessionFile {
    name: "notes.md",
    legacy: "digest.md",
};
pub const CONTEXT: SessionFile = SessionFile {
    name: "context.txt",
    legacy: "session_context.txt",
};
pub const DEBUG_LOG: SessionFile = SessionFile {
    name: "debug.log",
    legacy: "ltn_debug.log",
};
/// The transcript with speaker labels, written by a labeled session and by `diarize`.
pub const SPEAKER_TRANSCRIPT: &str = "transcript_speakers.txt";

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("ltn-session-files-{name}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn resolve_prefers_the_current_name() {
        let dir = temp_dir("current");
        std::fs::write(dir.join("notes.md"), "new").unwrap();
        std::fs::write(dir.join("digest.md"), "old").unwrap();
        assert_eq!(NOTES.resolve(&dir), Some(dir.join("notes.md")));
    }

    #[test]
    fn resolve_falls_back_to_the_legacy_name() {
        let dir = temp_dir("legacy");
        std::fs::write(dir.join("transcript_raw.txt"), "old").unwrap();
        assert_eq!(
            TRANSCRIPT.resolve(&dir),
            Some(dir.join("transcript_raw.txt"))
        );
        assert_eq!(CONTEXT.resolve(&dir), None);
    }
}
