//! Pick a saved session to view.

use std::path::{Path, PathBuf};

use crossterm::event::KeyEvent;
use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};

use crate::adapters::session_reader::{SessionInfo, list_sessions};
use crate::domain::session_files::{NOTES, TRANSCRIPT};
use crate::tui::host::{Action, Ctx, MouseInput, Screen};
use crate::tui::screens::picker::{Picker, PickerKey};

/// The file text, or a Markdown note when it cannot be read.
fn read_or_note(path: &Path) -> String {
    match std::fs::read(path) {
        Ok(bytes) => match String::from_utf8(bytes) {
            Ok(text) => text,
            Err(e) => format!("*Cannot read {}: {}*", file_name(path), utf8_error(&e)),
        },
        Err(e) => format!(
            "*Cannot read {}: {}*",
            file_name(path),
            crate::adapters::os_error::OsError::from_io(&e, Some(path), None)
        ),
    }
}

fn file_name(path: &Path) -> String {
    path.file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_default()
}

/// Python's `UnicodeDecodeError` text for the first bad byte.
pub fn utf8_error(error: &std::string::FromUtf8Error) -> String {
    let at = error.utf8_error().valid_up_to();
    let byte = error.as_bytes()[at];
    format!("'utf-8' codec can't decode byte 0x{byte:02x} in position {at}: invalid start byte")
}

/// The preview of one session, in Markdown.
pub fn session_preview(session_dir: &Path) -> String {
    let mut lines = vec![format!("## {}", file_name(session_dir)), String::new()];
    match TRANSCRIPT.resolve(session_dir) {
        Some(path) => {
            let text = read_or_note(&path);
            let all: Vec<&str> = text.trim().lines().collect();
            if all.is_empty() {
                lines.push("*Empty transcript*".into());
            } else {
                lines.push("### Transcript (first 10 lines)".into());
                lines.push("```text".into());
                lines.extend(all.iter().take(10).map(|line| (*line).to_string()));
                lines.push("```".into());
                if all.len() > 10 {
                    lines.push(format!("*...{} more lines*", all.len() - 10));
                }
            }
        }
        None => lines.push("*No transcript. The audio is in this folder.*".into()),
    }
    if let Some(path) = NOTES.resolve(session_dir) {
        lines.extend(["".into(), "---".into(), "".into(), "### Digest".into()]);
        let digest = read_or_note(&path);
        let digest = digest.trim();
        lines.push(if digest.is_empty() {
            "*Empty digest*".into()
        } else {
            digest.to_string()
        });
    }
    lines.join("\n")
}

fn row(session: &SessionInfo) -> Line<'static> {
    let dim = Style::default().add_modifier(Modifier::DIM);
    let badge = if !session.has_transcript {
        Span::styled("audio only", dim)
    } else if session.has_notes {
        Span::styled("\u{2713} digest", Style::default().fg(Color::Green))
    } else {
        Span::styled("no digest", dim)
    };
    Line::from(vec![Span::raw(format!("{}  ", session.name)), badge])
}

pub struct SessionPicker {
    picker: Picker,
    sessions: Vec<SessionInfo>,
    shown: Vec<usize>,
}

impl SessionPicker {
    pub fn new(sessions_dir: &Path) -> Self {
        let mut picker = Self {
            picker: Picker::new("Filter sessions...", 48),
            sessions: list_sessions(sessions_dir),
            shown: Vec::new(),
        };
        picker.rebuild("");
        picker
    }

    fn rebuild(&mut self, query: &str) {
        self.shown = (0..self.sessions.len())
            .filter(|index| {
                query.is_empty() || self.sessions[*index].name.to_lowercase().contains(query)
            })
            .collect();
        self.picker.list.reset(self.shown.len());
        self.show_preview();
    }

    fn current(&self) -> Option<&SessionInfo> {
        let index = *self.shown.get(self.picker.list.index?)?;
        self.sessions.get(index)
    }

    fn show_preview(&mut self) {
        let text = match self.current() {
            Some(session) => session_preview(&session.dir),
            None => "*No sessions found*".into(),
        };
        self.picker.preview.set_markdown(&text);
    }

    fn after(&mut self, outcome: PickerKey) -> Action {
        match outcome {
            PickerKey::Cancel => Action::Pop(None),
            PickerKey::Select => match self.current() {
                Some(session) => Action::Pop(Some(Box::new(session.dir.clone()))),
                None => Action::None,
            },
            PickerKey::Filter(query) => {
                self.rebuild(&query);
                Action::None
            }
            PickerKey::Moved => {
                self.show_preview();
                Action::None
            }
            PickerKey::Used | PickerKey::Unused => Action::None,
        }
    }
}

impl Screen for SessionPicker {
    fn render(&mut self, frame: &mut Frame, area: Rect, ctx: &Ctx) {
        let header = format!("  Select a session ({} found)", self.sessions.len());
        let rows: Vec<Line<'static>> = self
            .shown
            .iter()
            .map(|index| row(&self.sessions[*index]))
            .collect();
        self.picker.render(
            frame,
            area,
            &header,
            "[Enter] Select  [↑/↓] Navigate  [Esc] Cancel",
            &rows,
            ctx,
        );
    }

    fn key(&mut self, key: &KeyEvent, name: &str, _ctx: &mut Ctx) -> Action {
        let outcome = self.picker.key(key, name);
        self.after(outcome)
    }

    fn paste(&mut self, text: &str, _ctx: &mut Ctx) -> Action {
        let outcome = self.picker.paste(text);
        self.after(outcome)
    }

    fn mouse(&mut self, input: &MouseInput, _ctx: &mut Ctx) -> Action {
        let outcome = self.picker.mouse(input);
        self.after(outcome)
    }
}

/// The session folder a picker popped with.
pub fn picked_dir(result: Option<crate::tui::host::Payload>) -> Option<PathBuf> {
    result
        .and_then(|payload| payload.downcast::<PathBuf>().ok())
        .map(|dir| *dir)
}
