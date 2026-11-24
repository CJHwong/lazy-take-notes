//! `lazy-take-notes status [SESSION]`: show a headless session's status.

use std::path::{Path, PathBuf};

use serde_json::Value as JsonValue;

use crate::adapters::pyrepr::py_repr_json;
use crate::adapters::session_status::{STATUS_FILE, is_pid_alive, read_status};

fn quoted(text: &str) -> String {
    py_repr_json(&JsonValue::String(text.into()))
}

fn name_of(path: &Path) -> String {
    path.file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_default()
}

/// The newest session folder with a status file, or the one `name` picks:
/// an exact name first, then the only name that contains it.
pub fn resolve_status_dir(base_dir: &Path, name: Option<&str>) -> Result<PathBuf, String> {
    let mut candidates: Vec<PathBuf> = std::fs::read_dir(base_dir)
        .map(|entries| {
            entries
                .filter_map(Result::ok)
                .map(|entry| entry.path())
                .filter(|path| path.is_dir() && path.join(STATUS_FILE).exists())
                .collect()
        })
        .unwrap_or_default();
    candidates.sort();
    candidates.reverse();
    if candidates.is_empty() {
        return Err(format!(
            "No headless session found in {}.",
            base_dir.display()
        ));
    }
    let Some(name) = name else {
        return Ok(candidates.swap_remove(0));
    };
    match_by_name(candidates, name, "headless session", |path| name_of(path))
}

/// The item whose name is `name`, else the only one whose name contains it, as
/// Python's `_match_by_name`. `noun` names the kind in the "no match" error.
pub fn match_by_name<T>(
    items: Vec<T>,
    name: &str,
    noun: &str,
    name_of: impl Fn(&T) -> String,
) -> Result<T, String> {
    if let Some(index) = items.iter().position(|item| name_of(item) == name) {
        return Ok(items.into_iter().nth(index).expect("index is in range"));
    }
    let mut matches: Vec<T> = items
        .into_iter()
        .filter(|item| name_of(item).contains(name))
        .collect();
    match matches.len() {
        1 => Ok(matches.remove(0)),
        0 => Err(format!("No {noun} matching {}.", quoted(name))),
        _ => {
            let names: Vec<String> = matches.iter().map(&name_of).collect();
            Err(format!(
                "{} matches multiple sessions: {}",
                quoted(name),
                names.join(", ")
            ))
        }
    }
}

/// The lines `status` prints.
pub fn status_lines(session_dir: &Path) -> Result<Vec<String>, String> {
    let Some(state) = read_status(session_dir) else {
        return Err(format!(
            "Status file for {} is missing or unreadable.",
            quoted(&name_of(session_dir))
        ));
    };
    let display = if state.is_live_state() && !is_pid_alive(state.pid) {
        format!(
            "{} (crashed — process {} not running)",
            state.state, state.pid
        )
    } else {
        state.state.clone()
    };
    let mut lines = vec![
        format!("session:  {}", name_of(session_dir)),
        format!("state:    {display}"),
        format!("pid:      {}", state.pid),
        format!("segments: {}", state.segment_count),
        format!("digests:  {}", state.digest_count),
        format!("started:  {}", state.started_at),
        format!("updated:  {}", state.updated_at),
    ];
    if !state.error.is_empty() {
        lines.push(format!("error:    {}", state.error));
    }
    Ok(lines)
}
