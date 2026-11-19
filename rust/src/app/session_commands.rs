//! `ls`, `transcript` and `notes`: print saved sessions for scripts and pipes.

use std::fmt::Write as _;
use std::path::Path;

use crate::adapters::session_reader::{
    SessionInfo, latest_session, list_sessions, read_labeled_transcript, read_notes,
};
use crate::app::status_command::match_by_name;

/// `ls` shows this many sessions unless `--all` is given.
pub const LS_DEFAULT_LIMIT: usize = 20;

/// What `ls` writes: lines for stdout, and an optional note for stderr.
pub struct Listing {
    pub stdout: Vec<String>,
    pub stderr: Option<String>,
}

pub fn list(base_dir: &Path, as_json: bool, show_all: bool) -> Listing {
    let sessions = list_sessions(base_dir);
    if as_json {
        return Listing {
            stdout: vec![sessions_json(&sessions)],
            stderr: None,
        };
    }
    if sessions.is_empty() {
        return Listing {
            stdout: vec![format!("No sessions found in {}.", base_dir.display())],
            stderr: None,
        };
    }
    let shown = if show_all {
        sessions.len()
    } else {
        sessions.len().min(LS_DEFAULT_LIMIT)
    };
    let stdout = sessions[..shown]
        .iter()
        .map(|session| format!("{}  {}", marker(session), session.name))
        .collect();
    let hidden = sessions.len() - shown;
    let stderr =
        (hidden > 0).then(|| format!("... {hidden} more (use -a/--all to show every session)"));
    Listing { stdout, stderr }
}

fn marker(session: &SessionInfo) -> &'static str {
    if session.has_notes {
        "notes"
    } else if session.has_transcript {
        "   -"
    } else {
        "audio"
    }
}

/// The session `name` picks, or the newest one with a transcript.
pub fn resolve_session(base_dir: &Path, name: Option<&str>) -> Result<SessionInfo, String> {
    let Some(name) = name else {
        return latest_session(base_dir)
            .ok_or_else(|| format!("No sessions found in {}.", base_dir.display()));
    };
    match_by_name(list_sessions(base_dir), name, "session", |session| {
        session.name.clone()
    })
}

/// The labeled transcript, else the plain one, of the session `name` picks.
pub fn transcript(base_dir: &Path, name: Option<&str>) -> Result<String, String> {
    session_text(base_dir, name, read_labeled_transcript, "transcript")
}

pub fn notes(base_dir: &Path, name: Option<&str>) -> Result<String, String> {
    session_text(base_dir, name, read_notes, "notes")
}

fn session_text(
    base_dir: &Path,
    name: Option<&str>,
    reader: fn(&Path) -> Option<String>,
    kind: &str,
) -> Result<String, String> {
    let info = resolve_session(base_dir, name)?;
    reader(&info.dir).ok_or_else(|| {
        format!(
            "Session {} has no {kind}.",
            crate::adapters::pyrepr::py_repr_json(&serde_json::Value::String(info.name.clone()))
        )
    })
}

/// The list as Python's `json.dumps` writes it: `", "` and `": "` separators,
/// and every character past ASCII as a `\uXXXX` escape.
fn sessions_json(sessions: &[SessionInfo]) -> String {
    let items: Vec<String> = sessions
        .iter()
        .map(|session| {
            format!(
                "{{\"name\": {}, \"dir\": {}, \"has_notes\": {}, \"has_transcript\": {}}}",
                ascii_json_string(&session.name),
                ascii_json_string(&session.dir.display().to_string()),
                session.has_notes,
                session.has_transcript
            )
        })
        .collect();
    format!("[{}]", items.join(", "))
}

fn ascii_json_string(text: &str) -> String {
    let mut out = String::from("\"");
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            ' '..='~' => out.push(c),
            _ => {
                let mut units = [0u16; 2];
                for unit in c.encode_utf16(&mut units) {
                    let _ = write!(out, "\\u{unit:04x}");
                }
            }
        }
    }
    out.push('"');
    out
}

#[cfg(test)]
#[path = "session_commands_tests.rs"]
mod tests;
