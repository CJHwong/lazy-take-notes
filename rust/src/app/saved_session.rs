//! A saved session, read from its folder: the transcript, the notes, the
//! speakers, and a new label. The TUI view screen and the GUI reader both
//! show it, so the file rules live here once.

use std::path::{Path, PathBuf};

use crate::adapters::os_error::OsError;
use crate::adapters::saved_speakers::{self, SavedSpeakers};
use crate::app::container::safe_label;
use crate::app::session::fit_name;
use crate::domain::session_files::{NOTES, SPEAKER_TRANSCRIPT, TRANSCRIPT};

/// Python's `UnicodeDecodeError` text for the first bad byte.
pub fn utf8_error(error: &std::string::FromUtf8Error) -> String {
    let at = error.utf8_error().valid_up_to();
    let byte = error.as_bytes()[at];
    format!("'utf-8' codec can't decode byte 0x{byte:02x} in position {at}: invalid start byte")
}

/// The stripped text of `path`. The error names the file and says why.
pub fn read_saved(path: &Path) -> Result<String, String> {
    let failure = match std::fs::read(path) {
        Ok(bytes) => match String::from_utf8(bytes) {
            Ok(text) => return Ok(text.trim().to_string()),
            Err(e) => utf8_error(&e),
        },
        Err(e) => OsError::from_io(&e, Some(path), None).to_string(),
    };
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    Err(format!("Cannot read {name}: {failure}"))
}

/// The labeled transcript when the session has one, else the plain one.
pub fn transcript_path(dir: &Path) -> Option<PathBuf> {
    let labeled = dir.join(SPEAKER_TRANSCRIPT);
    match labeled.is_file() {
        true => Some(labeled),
        false => TRANSCRIPT.resolve(dir),
    }
}

/// The non-empty lines of a saved transcript.
pub fn transcript_lines(text: &str) -> Vec<String> {
    text.lines()
        .filter(|line| !line.trim().is_empty())
        .map(str::to_string)
        .collect()
}

/// A saved line `[hh:mm:ss] text` as its timestamp (with brackets) and its text.
pub fn split_line(text: &str) -> (Option<&str>, &str) {
    match text.starts_with('[') {
        true => match text.find("] ") {
            Some(at) => (Some(&text[..=at]), &text[at + 1..]),
            None => (None, text),
        },
        false => (None, text),
    }
}

/// True when `text` is a line of the speaker shown as `label`.
pub fn spoken_by(text: &str, label: &str) -> bool {
    text.find("] ")
        .is_some_and(|at| text[at + 2..].starts_with(&format!("{label}: ")))
}

/// True when the folder holds a recording (`recording*.wav`).
pub fn has_recording(dir: &Path) -> bool {
    std::fs::read_dir(dir).is_ok_and(|entries| {
        entries.filter_map(Result::ok).any(|entry| {
            let name = entry.file_name().to_string_lossy().into_owned();
            name.starts_with("recording") && name.ends_with(".wav")
        })
    })
}

/// Everything a reader shows. `errors` holds one line per file that could not be read.
#[derive(Debug, Default)]
pub struct SavedSession {
    pub transcript: Vec<String>,
    pub notes: String,
    pub speakers: Option<SavedSpeakers>,
    /// The session has no transcript file, only its audio.
    pub audio_only: bool,
    /// The note type the session ran with, when it was saved.
    pub template: Option<String>,
    pub errors: Vec<String>,
}

impl SavedSession {
    pub fn load(dir: &Path) -> Self {
        let mut session = Self::default();
        match saved_speakers::load(dir) {
            Ok(speakers) => session.speakers = speakers,
            Err(message) => session.errors.push(message),
        }
        session.reload_transcript(dir);
        session.template = crate::adapters::session_reader::template_name(dir);
        session.audio_only = TRANSCRIPT.resolve(dir).is_none() && has_recording(dir);
        if let Some(path) = NOTES.resolve(dir) {
            match read_saved(&path) {
                Ok(text) => session.notes = text,
                Err(message) => session.errors.push(message),
            }
        }
        session
    }

    /// Read the transcript again, after a speaker rename rewrote it.
    pub fn reload_transcript(&mut self, dir: &Path) {
        self.transcript.clear();
        let Some(path) = transcript_path(dir) else {
            return;
        };
        match read_saved(&path) {
            Ok(text) => self.transcript = transcript_lines(&text),
            Err(message) => self.errors.push(message),
        }
    }
}

/// Move the session folder to the name for `label`, as a live session does:
/// the 17-character timestamp, `_`, then the safe label. Ok(None): the label
/// is empty or gives the same name, so nothing moved.
pub fn rename_folder(dir: &Path, label: &str) -> Result<Option<PathBuf>, String> {
    let safe = fit_name(&safe_label(label.trim()));
    if safe.is_empty() {
        return Ok(None);
    }
    let current = dir
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_default();
    let prefix: String = current.chars().take(17).collect();
    let new_name = format!("{prefix}_{safe}");
    let new_path = dir
        .parent()
        .map_or_else(|| PathBuf::from(&new_name), |parent| parent.join(&new_name));
    if new_path == dir {
        return Ok(None);
    }
    if new_path.exists() {
        return Err(format!("Session folder {new_name} already exists"));
    }
    std::fs::rename(dir, &new_path).map_err(|e| {
        let text = OsError::from_io(&e, Some(dir), Some(&new_path));
        format!("Could not rename the session folder: {text}")
    })?;
    Ok(Some(new_path))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn session_dir() -> (tempfile::TempDir, PathBuf) {
        let root = tempfile::tempdir().unwrap();
        let dir = root.path().join("2026-02-21_143052_standup");
        std::fs::create_dir(&dir).unwrap();
        (root, dir)
    }

    #[test]
    fn loads_the_plain_transcript_and_the_notes() {
        let (_root, dir) = session_dir();
        std::fs::write(
            dir.join("transcript.txt"),
            "[00:00:01] hi\n\n[00:00:02] there\n",
        )
        .unwrap();
        std::fs::write(dir.join("notes.md"), "  # Notes\n").unwrap();
        let session = SavedSession::load(&dir);
        assert_eq!(session.transcript, ["[00:00:01] hi", "[00:00:02] there"]);
        assert_eq!(session.notes, "# Notes");
        assert!(!session.audio_only);
        assert!(session.errors.is_empty());
    }

    #[test]
    fn prefers_the_labeled_transcript() {
        let (_root, dir) = session_dir();
        std::fs::write(dir.join("transcript.txt"), "[00:00:01] hi").unwrap();
        std::fs::write(dir.join(SPEAKER_TRANSCRIPT), "[00:00:01] S0: hi").unwrap();
        assert_eq!(SavedSession::load(&dir).transcript, ["[00:00:01] S0: hi"]);
    }

    #[test]
    fn a_folder_with_only_audio_is_audio_only() {
        let (_root, dir) = session_dir();
        std::fs::write(dir.join("recording.wav"), b"").unwrap();
        let session = SavedSession::load(&dir);
        assert!(session.audio_only);
        assert!(session.transcript.is_empty());
        assert!(session.notes.is_empty());
    }

    #[test]
    fn a_bad_file_is_an_error_line() {
        let (_root, dir) = session_dir();
        std::fs::write(dir.join("notes.md"), [b'a', 0xff]).unwrap();
        let session = SavedSession::load(&dir);
        assert_eq!(
            session.errors,
            [
                "Cannot read notes.md: 'utf-8' codec can't decode byte 0xff in position 1: invalid start byte"
            ]
        );
    }

    #[test]
    fn splits_the_timestamp_from_the_text() {
        assert_eq!(split_line("[00:00:01] hi"), (Some("[00:00:01]"), " hi"));
        assert_eq!(split_line("no stamp"), (None, "no stamp"));
        assert_eq!(split_line("[broken"), (None, "[broken"));
    }

    #[test]
    fn spoken_by_matches_the_label_after_the_stamp() {
        assert!(spoken_by("[00:00:01] Ann: hi", "Ann"));
        assert!(!spoken_by("[00:00:01] Anna: hi", "Ann"));
        assert!(!spoken_by("Ann: hi", "Ann"));
    }

    #[test]
    fn rename_keeps_the_timestamp_and_moves_the_folder() {
        let (root, dir) = session_dir();
        let moved = rename_folder(&dir, "team sync!").unwrap().unwrap();
        assert_eq!(moved, root.path().join("2026-02-21_143052_team_sync_"));
        assert!(moved.is_dir());
        assert!(!dir.exists());
    }

    #[test]
    fn rename_to_nothing_or_the_same_name_moves_nothing() {
        let (_root, dir) = session_dir();
        assert_eq!(rename_folder(&dir, "  "), Ok(None));
        assert_eq!(rename_folder(&dir, "standup"), Ok(None));
        assert!(dir.is_dir());
    }

    #[test]
    fn rename_onto_an_existing_folder_is_refused() {
        let (root, dir) = session_dir();
        std::fs::create_dir(root.path().join("2026-02-21_143052_taken")).unwrap();
        assert_eq!(
            rename_folder(&dir, "taken"),
            Err("Session folder 2026-02-21_143052_taken already exists".into())
        );
        assert!(dir.is_dir());
    }
}
