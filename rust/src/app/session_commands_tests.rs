use std::fs;
use std::path::Path;

use super::*;

fn session(base: &Path, name: &str, files: &[&str]) {
    let dir = base.join(name);
    fs::create_dir_all(&dir).unwrap();
    for file in files {
        fs::write(dir.join(file), format!("{name} {file}\n")).unwrap();
    }
}

#[test]
fn ls_marks_each_session_newest_first() {
    let base = tempfile::tempdir().unwrap();
    session(base.path(), "2026-01-01_a", &["transcript.txt", "notes.md"]);
    session(base.path(), "2026-01-02_b", &["transcript.txt"]);
    session(base.path(), "2026-01-03_c", &["recording.wav"]);
    let listing = list(base.path(), false, false);
    assert_eq!(
        listing.stdout,
        [
            "audio  2026-01-03_c",
            "   -  2026-01-02_b",
            "notes  2026-01-01_a"
        ]
    );
    assert!(listing.stderr.is_none());
}

#[test]
fn ls_shows_the_newest_twenty_and_counts_the_rest() {
    let base = tempfile::tempdir().unwrap();
    for day in 1..=22 {
        session(
            base.path(),
            &format!("2026-01-{day:02}"),
            &["transcript.txt"],
        );
    }
    let listing = list(base.path(), false, false);
    assert_eq!(listing.stdout.len(), LS_DEFAULT_LIMIT);
    assert_eq!(listing.stdout[0], "   -  2026-01-22");
    assert_eq!(
        listing.stderr.as_deref(),
        Some("... 2 more (use -a/--all to show every session)")
    );
    assert_eq!(list(base.path(), false, true).stdout.len(), 22);
}

#[test]
fn ls_without_sessions_says_where_it_looked() {
    let base = tempfile::tempdir().unwrap();
    let listing = list(base.path(), false, false);
    assert_eq!(
        listing.stdout,
        [format!("No sessions found in {}.", base.path().display())]
    );
    assert_eq!(list(base.path(), true, false).stdout, ["[]"]);
}

#[test]
fn ls_json_matches_python_json_dumps() {
    let base = tempfile::tempdir().unwrap();
    session(base.path(), "2026-01-01_會議 \"q\"", &["transcript.txt"]);
    let dir = base.path().join("2026-01-01_會議 \"q\"");
    let expected = format!(
        "[{{\"name\": \"2026-01-01_\\u6703\\u8b70 \\\"q\\\"\", \"dir\": {}, \"has_notes\": false, \"has_transcript\": true}}]",
        ascii_json_string(&dir.display().to_string())
    );
    assert_eq!(list(base.path(), true, false).stdout, [expected]);
    assert_eq!(ascii_json_string("a😀\n"), "\"a\\ud83d\\ude00\\n\"");
}

#[test]
fn transcript_prefers_the_labeled_one_and_defaults_to_the_newest() {
    let base = tempfile::tempdir().unwrap();
    session(base.path(), "2026-01-01_old", &["transcript.txt"]);
    session(
        base.path(),
        "2026-01-02_new",
        &["transcript.txt", "transcript_speakers.txt"],
    );
    session(base.path(), "2026-01-03_audio", &["recording.wav"]);
    assert_eq!(
        transcript(base.path(), None).unwrap(),
        "2026-01-02_new transcript_speakers.txt\n"
    );
    assert_eq!(
        transcript(base.path(), Some("old")).unwrap(),
        "2026-01-01_old transcript.txt\n"
    );
}

#[test]
fn notes_and_names_fail_with_python_messages() {
    let base = tempfile::tempdir().unwrap();
    assert_eq!(
        notes(base.path(), None).unwrap_err(),
        format!("No sessions found in {}.", base.path().display())
    );
    session(base.path(), "2026-01-01_standup", &["transcript.txt"]);
    session(
        base.path(),
        "2026-01-02_standup",
        &["transcript.txt", "notes.md"],
    );
    assert_eq!(
        notes(base.path(), Some("2026-01-01")).unwrap_err(),
        "Session '2026-01-01_standup' has no notes."
    );
    assert_eq!(
        notes(base.path(), Some("retro")).unwrap_err(),
        "No session matching 'retro'."
    );
    assert_eq!(
        notes(base.path(), Some("standup")).unwrap_err(),
        "'standup' matches multiple sessions: 2026-01-02_standup, 2026-01-01_standup"
    );
    assert_eq!(
        notes(base.path(), Some("2026-01-02_standup")).unwrap(),
        "2026-01-02_standup notes.md\n"
    );
}
