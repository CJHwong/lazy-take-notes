//! Read-only access to saved session folders.
//!
//! Sessions are timestamp-prefixed subfolders, so a lexical sort of the names is
//! chronological and the newest session is the current one.

use std::fs;
use std::path::{Path, PathBuf};

use crate::domain::session_files::{NOTES, SPEAKER_TRANSCRIPT, TEMPLATE, TRANSCRIPT};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionInfo {
    pub dir: PathBuf,
    pub name: String,
    pub has_notes: bool,
    pub has_transcript: bool,
    /// The note type the session ran with. Sessions before it was saved have none.
    pub template: Option<String>,
}

/// The note type a session ran with, from its `template.txt`.
pub fn template_name(dir: &Path) -> Option<String> {
    let text = fs::read_to_string(dir.join(TEMPLATE)).ok()?;
    let name = text.trim();
    (!name.is_empty()).then(|| name.to_string())
}

/// Sessions under `base_dir`, newest first. A session is a folder with a transcript
/// or a recording: a failed transcription keeps only its audio, and the user must
/// still find it.
pub fn list_sessions(base_dir: &Path) -> Vec<SessionInfo> {
    let Ok(entries) = fs::read_dir(base_dir) else {
        return Vec::new();
    };
    let mut dirs: Vec<PathBuf> = entries
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .collect();
    dirs.sort_by(|a, b| b.file_name().cmp(&a.file_name()));
    dirs.into_iter()
        .filter_map(|dir| {
            let has_transcript = TRANSCRIPT.resolve(&dir).is_some();
            if !has_transcript && !has_recording(&dir) {
                return None;
            }
            Some(SessionInfo {
                name: dir.file_name()?.to_string_lossy().into_owned(),
                has_notes: NOTES.resolve(&dir).is_some(),
                has_transcript,
                template: template_name(&dir),
                dir,
            })
        })
        .collect()
}

fn has_recording(dir: &Path) -> bool {
    let Ok(entries) = fs::read_dir(dir) else {
        return false;
    };
    entries.filter_map(Result::ok).any(|entry| {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        name.starts_with("recording") && name.ends_with(".wav")
    })
}

/// The newest session with a transcript.
pub fn latest_session(base_dir: &Path) -> Option<SessionInfo> {
    list_sessions(base_dir)
        .into_iter()
        .find(|session| session.has_transcript)
}

pub fn read_transcript(session_dir: &Path) -> Option<String> {
    fs::read_to_string(TRANSCRIPT.resolve(session_dir)?).ok()
}

/// The transcript with speaker labels when the session has one, else the plain one.
pub fn read_labeled_transcript(session_dir: &Path) -> Option<String> {
    let labeled = session_dir.join(SPEAKER_TRANSCRIPT);
    if labeled.is_file() {
        return fs::read_to_string(labeled).ok();
    }
    read_transcript(session_dir)
}

pub fn read_notes(session_dir: &Path) -> Option<String> {
    fs::read_to_string(NOTES.resolve(session_dir)?).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lists_sessions_newest_first_including_audio_only_ones() {
        let dir = tempfile::tempdir().unwrap();
        let make = |name: &str, files: &[&str]| {
            let session = dir.path().join(name);
            fs::create_dir_all(&session).unwrap();
            for file in files {
                fs::write(session.join(file), "x").unwrap();
            }
        };
        make("2026-01-01_090000", &["transcript_raw.txt", "digest.md"]);
        make("2026-01-02_090000", &["recording_2.wav"]);
        make("2026-01-03_090000", &["notes.md"]);
        make("2026-01-04_090000", &["transcript.txt"]);
        let names: Vec<(String, bool, bool)> = list_sessions(dir.path())
            .into_iter()
            .map(|s| (s.name, s.has_notes, s.has_transcript))
            .collect();
        assert_eq!(
            names,
            [
                ("2026-01-04_090000".into(), false, true),
                ("2026-01-02_090000".into(), false, false),
                ("2026-01-01_090000".into(), true, true),
            ]
        );
        assert_eq!(
            latest_session(dir.path()).unwrap().name,
            "2026-01-04_090000"
        );
        assert!(list_sessions(&dir.path().join("missing")).is_empty());
    }

    #[test]
    fn a_session_names_its_note_type_once_saved() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(template_name(dir.path()), None, "older sessions have none");
        fs::write(dir.path().join("template.txt"), "  \n").unwrap();
        assert_eq!(
            template_name(dir.path()),
            None,
            "a blank file names nothing"
        );
        fs::write(dir.path().join("template.txt"), "Lecture Notes\n").unwrap();
        assert_eq!(template_name(dir.path()).as_deref(), Some("Lecture Notes"));
    }

    #[test]
    fn the_labeled_transcript_wins_when_present() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("transcript.txt"), "plain").unwrap();
        assert_eq!(
            read_labeled_transcript(dir.path()).as_deref(),
            Some("plain")
        );
        fs::write(dir.path().join(SPEAKER_TRANSCRIPT), "S1: hi").unwrap();
        assert_eq!(
            read_labeled_transcript(dir.path()).as_deref(),
            Some("S1: hi")
        );
        assert_eq!(read_notes(dir.path()), None);
    }
}
