//! The words a session screen shows, the same in the terminal and the window.

use crate::app::session::SessionRuntime;
use crate::domain::transcript::TranscriptSegment;

/// The header text. Textual reads `[` plus a letter, `#`, `/` or `@` as a
/// markup tag and drops it, so a locale like ` [en]` never shows.
pub fn header_text(name: &str, description: &str, locale: &str, label: &str) -> String {
    let mut text = format!("  lazy-take-notes | {name}");
    if !description.is_empty() {
        text.push_str(&format!(" - {description}"));
    }
    let tag_like = locale
        .chars()
        .next()
        .is_some_and(|c| c.is_alphabetic() || matches!(c, '#' | '/' | '@'));
    // The space before the tag stays.
    match (locale.is_empty(), tag_like) {
        (true, _) => {}
        (false, true) => text.push(' '),
        (false, false) => text.push_str(&format!(" [{locale}]")),
    }
    if !label.is_empty() {
        text.push_str(&format!(" \u{2014} {label}"));
    }
    text
}

/// The digest panel's text when the digest model is missing.
pub fn missing_models_notice(missing: &[String]) -> String {
    let pulls: Vec<String> = missing
        .iter()
        .map(|model| format!("`ollama pull {model}`"))
        .collect();
    format!(
        "**LLM model unavailable**\n\nDigests are disabled. To enable:\n\n{}\n\nThen restart.",
        pulls.join("\n\n")
    )
}

/// A transcript line's text after its time: `label: text`, or the text alone.
pub fn spoken_text(segment: &TranscriptSegment, label: Option<&str>) -> String {
    match label {
        Some(label) => format!("{label}: {}", segment.text),
        None => segment.text.clone(),
    }
}

/// The help text: the template, the status bar legend, then the mode's keys.
pub fn help_markdown(rt: &SessionRuntime, rows: &[&str]) -> String {
    let metadata = &rt.controller.template().metadata;
    let mut lines: Vec<String> = Vec::new();
    if !metadata.name.is_empty() {
        lines.push(format!("**Template:** {}\n", metadata.name));
    }
    if !metadata.description.is_empty() {
        lines.push(format!("**Description:** {}\n", metadata.description));
    }
    if !metadata.locale.is_empty() {
        lines.push(format!("**Locale:** {}\n", metadata.locale));
    }
    if !lines.is_empty() {
        lines.push(String::new());
    }
    let min_lines = rt.controller.config().digest.min_lines;
    lines.extend([
        "### Status Bar".to_string(),
        "| Indicator | Meaning |".into(),
        "|-----------|---------|".into(),
        "| `\u{25cf} Rec` `\u{275a}\u{275a} Paused` `\u{25a0} Stopped` `\u{25cb} Idle` | Recording state |".into(),
        format!("| `buf N/{min_lines}` | Lines buffered toward next digest (fires at {min_lines}) |"),
        "| `00:00:00` | Recording time, pauses excluded |".into(),
        "| `last Xs ago` | Time elapsed since the last digest |".into(),
        "| `\u{2581}\u{2582}\u{2584}\u{2588}\u{2584}\u{2582}` | Mic input level \u{2014} flat means silence detected |".into(),
        "| `\u{27f3} Transcribing\u{2026}` | Speech-to-text in progress |".into(),
        "| `\u{27f3} Digesting\u{2026}` | LLM digest cycle in progress |".into(),
        String::new(),
        "### Keybindings".into(),
    ]);
    lines.extend(rows.iter().map(|row| row.to_string()));
    lines.join("\n")
}

pub const RECORD_HELP_ROWS: [&str; 13] = [
    "| Key | Action |",
    "|-----|--------|",
    "| `Space` | Pause / Resume |",
    "| `s` | Stop recording |",
    "| `d` | Force digest now |",
    "| `m` | Mute / Unmute mic |",
    "| `c` | Copy focused panel |",
    "| `Tab` | Switch panel focus |",
    "| `←` `→` `Enter` | On the speaker bar: choose a speaker, name it |",
    "| `l` | Rename session |",
    "| `o` | Open session directory |",
    "| `h` | Toggle this help |",
    "| `q` | Quit |",
];

pub const QUIT_HINT: &str =
    "Press s to stop and save, then q for the final notes. ctrl+q saves and quits without them.";

pub fn record_hints(state: &str) -> &'static str {
    match state {
        "recording" => {
            "[Space] pause  [s] stop  [d] digest  [m] mute mic  [c] copy  [l] label  [h] help"
        }
        "paused" => "[Space] resume  [s] stop  [c] copy  [l] label  [h] help",
        _ => "[c] copy  [l] label  [o] open  [h] help  [q] quit",
    }
}

pub const FILE_HELP_ROWS: [&str; 11] = [
    "| Key | Action |",
    "|-----|--------|",
    "| `s` | Stop transcription |",
    "| `d` | Force digest now |",
    "| `c` | Copy focused panel |",
    "| `Tab` | Switch panel focus |",
    "| `←` `→` `Enter` | On the speaker bar: choose a speaker, name it |",
    "| `l` | Rename session |",
    "| `o` | Open session directory |",
    "| `h` | Toggle this help |",
    "| `q` | Quit |",
];

pub fn file_hints(state: &str) -> &'static str {
    match state {
        "recording" => "[s] stop  [d] digest  [c] copy  [l] label  [h] help",
        _ => "[c] copy  [l] label  [o] open  [h] help  [q] quit",
    }
}
