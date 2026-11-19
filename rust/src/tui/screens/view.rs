//! Read-only view of a saved session: transcript and digest side by side.

use std::path::{Path, PathBuf};
use std::time::Duration;

use crossterm::event::KeyEvent;
use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::Style;
use ratatui::text::Line;
use tracing::error;

use crate::adapters::os_error::OsError;
use crate::domain::session_files::{NOTES, TRANSCRIPT};
use crate::tui::host::{Action, Ctx, Payload, Screen};
use crate::tui::keys::{Binding, SCREEN_BASE, action_for, bind};
use crate::tui::screens::session_picker::{SessionPicker, picked_dir, utf8_error};
use crate::tui::system::{copy_to_clipboard, open_folder};
use crate::tui::theme::Theme;
use crate::tui::widgets::scroll::ScrollView;
use crate::tui::widgets::status_bar::StatusBar;
use crate::tui::widgets::{bar, panel, split_bars};

pub const VIEW_BINDINGS: [Binding; 6] = [
    bind("tab", "app.focus_next", "Switch Panel").hidden(),
    SCREEN_BASE[1],
    SCREEN_BASE[2],
    SCREEN_BASE[3],
    bind("q", "quit_app", "Quit").priority(),
    bind("o", "open_session_dir", "Open").hidden(),
];

pub const TRANSCRIPT_PANEL_BINDINGS: [Binding; 11] = [
    bind("up", "scroll_up", "Scroll Up").hidden(),
    bind("down", "scroll_down", "Scroll Down").hidden(),
    bind("left", "scroll_left", "Scroll Left").hidden(),
    bind("right", "scroll_right", "Scroll Right").hidden(),
    bind("home", "scroll_home", "Scroll Home").hidden(),
    bind("end", "scroll_end", "Scroll End").hidden(),
    bind("pageup", "page_up", "Page Up").hidden(),
    bind("pagedown", "page_down", "Page Down").hidden(),
    bind("ctrl+pageup", "page_left", "Page Left").hidden(),
    bind("ctrl+pagedown", "page_right", "Page Right").hidden(),
    bind("c", "copy_content", "Copy").hidden(),
];

pub const DIGEST_PANEL_BINDINGS: [Binding; 7] = [
    bind("c", "copy_content", "Copy").hidden(),
    bind("up", "scroll_up", "Scroll up").hidden(),
    bind("down", "scroll_down", "Scroll down").hidden(),
    bind("pageup", "page_up", "Page up").hidden(),
    bind("pagedown", "page_down", "Page down").hidden(),
    bind("home", "scroll_home", "Home").hidden(),
    bind("end", "scroll_end", "End").hidden(),
];

/// The transcript panel: a log with copyable plain lines.
pub struct TranscriptPanel {
    pub view: ScrollView,
    pub all_text: Vec<String>,
}

impl Default for TranscriptPanel {
    fn default() -> Self {
        Self {
            view: ScrollView::log(),
            all_text: Vec::new(),
        }
    }
}

impl TranscriptPanel {
    pub fn render(&mut self, frame: &mut Frame, area: Rect, focused: bool, theme: &Theme) {
        let background = if focused {
            theme.tinted("surface", "foreground", 0.05)
        } else {
            theme.color("surface")
        };
        let border = if focused { "accent" } else { "primary" };
        let inner = panel(frame, area, "Transcript", border, background, theme);
        let style = Style::default()
            .bg(background)
            .fg(theme.on("foreground", "surface"));
        self.view.render(frame, inner, style, theme);
    }

    /// Copy every line, with the session context when the screen has a locked one.
    pub fn copy(&self, context: &str, ctx: &mut Ctx) {
        if self.all_text.is_empty() {
            ctx.warn("No transcript to copy", Duration::from_secs(2));
            return;
        }
        let mut text = self.all_text.join("\n");
        if !context.is_empty() {
            text.push_str(&format!("\n\n---\nSession Context:\n{context}"));
        }
        copied(copy_to_clipboard(&text), "Transcript copied", ctx);
    }
}

/// The digest panel: rendered Markdown with copyable source.
pub struct DigestPanel {
    pub view: ScrollView,
    pub markdown: String,
}

impl Default for DigestPanel {
    fn default() -> Self {
        Self {
            view: ScrollView::markdown(),
            markdown: String::new(),
        }
    }
}

impl DigestPanel {
    pub fn update_digest(&mut self, markdown: &str) {
        self.markdown = markdown.to_string();
        self.view.set_markdown(markdown);
    }

    pub fn render(&mut self, frame: &mut Frame, area: Rect, focused: bool, theme: &Theme) {
        let background = theme.color("background");
        let border = if focused { "accent" } else { "secondary" };
        let inner = panel(frame, area, "Digest", border, background, theme);
        let style = Style::default()
            .bg(background)
            .fg(theme.color("foreground"));
        self.view.render(frame, inner, style, theme);
    }

    pub fn copy(&self, context: &str, ctx: &mut Ctx) {
        if self.markdown.is_empty() {
            ctx.warn("No digest to copy", Duration::from_secs(2));
            return;
        }
        let mut text = self.markdown.clone();
        if !context.is_empty() {
            text.push_str(&format!("\n\n---\n\n**Session Context**\n\n{context}"));
        }
        copied(copy_to_clipboard(&text), "Digest copied", ctx);
    }
}

fn copied(result: Result<(), String>, done: &str, ctx: &mut Ctx) {
    match result {
        Ok(()) => ctx.notify_with(
            done,
            crate::tui::host::Severity::Information,
            Duration::from_secs(2),
            "",
        ),
        Err(e) => {
            error!(target: "ltn.app", "Cannot copy to the clipboard: {e}");
            ctx.error(
                format!("Cannot copy to the clipboard: {e}"),
                Duration::from_secs(8),
            );
        }
    }
}

/// Open the session folder, or say why not.
pub fn open_session_dir(dir: &Path, ctx: &mut Ctx) {
    if let Err(e) = open_folder(dir) {
        error!(target: "ltn.view", "Cannot open {}: {e}", dir.display());
        ctx.error(
            format!("Cannot open the session folder: {e}"),
            crate::tui::host::NOTIFICATION_TIMEOUT,
        );
    }
}

/// The header row: `$primary`, bold, centered.
pub fn header(frame: &mut Frame, area: Rect, text: &str, theme: &Theme) {
    bar(frame, area, text, "primary", "text", true, theme);
}

pub struct ViewScreen {
    session_dir: PathBuf,
    transcript: TranscriptPanel,
    digest: DigestPanel,
    status: StatusBar,
    digest_focused: bool,
}

impl ViewScreen {
    pub fn new(session_dir: PathBuf) -> Self {
        let mut status = StatusBar::default();
        status.mode_label = "View".into();
        status.keybinding_hints = "[c] copy  [o] open  [Tab] switch  [q] back".into();
        status.set_stopped(true);
        Self {
            session_dir,
            transcript: TranscriptPanel::default(),
            digest: DigestPanel::default(),
            status,
            digest_focused: false,
        }
    }

    /// The stripped file text, or "" after a toast when it cannot be read.
    fn read_saved(&self, path: Option<PathBuf>, ctx: &mut Ctx) -> String {
        let Some(path) = path else {
            return String::new();
        };
        let failure = match std::fs::read(&path) {
            Ok(bytes) => match String::from_utf8(bytes) {
                Ok(text) => return text.trim().to_string(),
                Err(e) => utf8_error(&e),
            },
            Err(e) => OsError::from_io(&e, Some(&path), None).to_string(),
        };
        error!(target: "ltn.view", "Cannot read {}: {failure}", path.display());
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        ctx.error(
            format!("Cannot read {name}: {failure}"),
            Duration::from_secs(8),
        );
        String::new()
    }

    fn has_recording(&self) -> bool {
        std::fs::read_dir(&self.session_dir).is_ok_and(|entries| {
            entries.filter_map(Result::ok).any(|entry| {
                let name = entry.file_name().to_string_lossy().into_owned();
                name.starts_with("recording") && name.ends_with(".wav")
            })
        })
    }
}

impl Screen for ViewScreen {
    fn mounted(&mut self, ctx: &mut Ctx) -> Action {
        let text = self.read_saved(TRANSCRIPT.resolve(&self.session_dir), ctx);
        // The saved lines already carry their timestamps; write them as they are.
        for line in text.lines().filter(|line| !line.trim().is_empty()) {
            self.transcript.all_text.push(line.to_string());
            self.transcript.view.write(Line::from(line.to_string()));
        }
        if TRANSCRIPT.resolve(&self.session_dir).is_none() && self.has_recording() {
            // A failed transcription keeps only the audio.
            self.transcript.view.write(Line::from(
                "No transcript. The audio is saved in this folder: press o to open it.",
            ));
        }
        let digest = self.read_saved(NOTES.resolve(&self.session_dir), ctx);
        if !digest.is_empty() {
            self.digest.update_digest(&digest);
        }
        Action::None
    }

    fn render(&mut self, frame: &mut Frame, area: Rect, ctx: &Ctx) {
        let theme = &ctx.theme;
        let (top, body, bottom) = split_bars(area, 1, self.status.height());
        let name = self
            .session_dir
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        header(
            frame,
            top,
            &format!("  lazy-take-notes | View: {name}"),
            theme,
        );
        let left = body.width / 2;
        self.transcript.render(
            frame,
            Rect::new(body.x, body.y, left, body.height),
            !self.digest_focused,
            theme,
        );
        self.digest.render(
            frame,
            Rect::new(body.x + left, body.y, body.width - left, body.height),
            self.digest_focused,
            theme,
        );
        self.status.render(frame, bottom, theme);
    }

    fn key(&mut self, _key: &KeyEvent, name: &str, ctx: &mut Ctx) -> Action {
        let panel_table: &[Binding] = if self.digest_focused {
            &DIGEST_PANEL_BINDINGS
        } else {
            &TRANSCRIPT_PANEL_BINDINGS
        };
        // `q` is a priority binding: it wins over the focused panel.
        if action_for(&VIEW_BINDINGS, name) == Some("quit_app") {
            return Action::Pop(None);
        }
        match action_for(panel_table, name) {
            Some("copy_content") if self.digest_focused => self.digest.copy("", ctx),
            Some("copy_content") => self.transcript.copy("", ctx),
            Some(_) if self.digest_focused => {
                self.digest.view.key(name);
            }
            Some(_) => {
                self.transcript.view.key(name);
            }
            None => match action_for(&VIEW_BINDINGS, name) {
                Some("open_session_dir") => open_session_dir(&self.session_dir, ctx),
                Some("app.focus_next" | "app.focus_previous") => {
                    self.digest_focused = !self.digest_focused
                }
                _ => {}
            },
        }
        Action::None
    }
}

/// `lazy-take-notes view`: pick a session, view it, and pick again until Esc.
#[derive(Default)]
pub struct ViewLoop {
    viewing: bool,
    base_dir: PathBuf,
}

impl ViewLoop {
    pub fn new(base_dir: PathBuf) -> Self {
        Self {
            viewing: false,
            base_dir,
        }
    }
}

impl Screen for ViewLoop {
    fn mounted(&mut self, _ctx: &mut Ctx) -> Action {
        Action::Push(Box::new(SessionPicker::new(&self.base_dir)))
    }

    fn resumed(&mut self, result: Option<Payload>, _ctx: &mut Ctx) -> Action {
        if self.viewing {
            self.viewing = false;
            return Action::Push(Box::new(SessionPicker::new(&self.base_dir)));
        }
        match picked_dir(result) {
            Some(dir) => {
                self.viewing = true;
                Action::Push(Box::new(ViewScreen::new(dir)))
            }
            None => Action::Pop(None),
        }
    }

    fn render(&mut self, _frame: &mut Frame, _area: Rect, _ctx: &Ctx) {}

    fn key(&mut self, _key: &KeyEvent, _name: &str, _ctx: &mut Ctx) -> Action {
        Action::None
    }
}
