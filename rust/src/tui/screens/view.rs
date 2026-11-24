//! Read-only view of a saved session: transcript and digest side by side.

use std::path::{Path, PathBuf};
use std::time::Duration;

use crossterm::event::KeyEvent;
use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use tracing::error;

use crate::adapters::os_error::OsError;
use crate::adapters::saved_speakers::{self, SavedSpeakers};
use crate::domain::session_files::{NOTES, SPEAKER_TRANSCRIPT, TRANSCRIPT};
use crate::tui::host::{Action, Ctx, MouseInput, MouseKind, Payload, Screen};
use crate::tui::keys::{Binding, SCREEN_BASE, action_for, bind};
use crate::tui::screens::modals::LabelModal;
use crate::tui::screens::session::{
    BarKey, render_speaker_bar, select_on_entry, speaker_bar_key, wrap_chips,
};
use crate::tui::screens::session_picker::{SessionPicker, picked_dir, utf8_error};
use crate::tui::system::{copy_to_clipboard, open_folder};
use crate::tui::theme::Theme;
use crate::tui::widgets::scroll::ScrollView;
use crate::tui::widgets::status_bar::StatusBar;
use crate::tui::widgets::{bar, split_bars};

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
    /// The speaker label whose lines have a highlighted background.
    pub highlight: Option<String>,
    /// What the drawn lines show: (label, background, line count).
    styled: Option<(Option<String>, Color, usize)>,
}

impl Default for TranscriptPanel {
    fn default() -> Self {
        Self {
            view: {
                // One column of room inside the border; the scrollbar sits on its edge.
                let mut view = ScrollView::log();
                view.padding = 1;
                view
            },
            all_text: Vec::new(),
            highlight: None,
            styled: None,
        }
    }
}

/// One saved line, `[hh:mm:ss] text`: the timestamp dim, then the text. `background`
/// marks a line of the selected speaker.
pub fn transcript_line(text: &str, background: Option<Color>) -> Line<'static> {
    let line = match text.starts_with('[') {
        true => match text.find("] ") {
            Some(at) => Line::from(vec![
                Span::styled(
                    text[..=at].to_string(),
                    Style::default().add_modifier(Modifier::DIM),
                ),
                Span::raw(text[at + 1..].to_string()),
            ]),
            None => Line::from(text.to_string()),
        },
        false => Line::from(text.to_string()),
    };
    match background {
        Some(color) => line.style(Style::default().bg(color)),
        None => line,
    }
}

/// True when `text` is a line of the speaker shown as `label`.
fn spoken_by(text: &str, label: &str) -> bool {
    text.find("] ")
        .is_some_and(|at| text[at + 2..].starts_with(&format!("{label}: ")))
}

impl TranscriptPanel {
    pub fn render(&mut self, frame: &mut Frame, area: Rect, focused: bool, theme: &Theme) {
        let background = theme.color("background");
        let border = crate::tui::widgets::border_color(focused, theme);
        let inner =
            crate::tui::widgets::rounded(frame, area, "Transcript", border, background, theme);
        let style = Style::default()
            .bg(background)
            .fg(theme.color("foreground"));
        self.restyle(theme);
        self.view.render(frame, inner, style, theme);
    }

    /// Drop every line; the next render styles the new ones from scratch.
    pub fn clear(&mut self) {
        self.view.clear();
        self.all_text.clear();
        self.styled = None;
    }

    /// Redraw the lines when the highlighted speaker, the theme or the lines changed.
    fn restyle(&mut self, theme: &Theme) {
        let marked = theme.tinted("background", "primary", 0.35);
        let wanted = (self.highlight.clone(), marked, self.all_text.len());
        let was_marked = self
            .styled
            .as_ref()
            .is_some_and(|(label, _, _)| label.is_some());
        if self.styled.as_ref() == Some(&wanted) || (self.highlight.is_none() && !was_marked) {
            return;
        }
        let lines = self
            .all_text
            .iter()
            .map(|text| {
                let mine = self
                    .highlight
                    .as_deref()
                    .is_some_and(|label| spoken_by(text, label));
                transcript_line(text, mine.then_some(marked))
            })
            .collect();
        self.view.restyle_lines(lines);
        self.styled = Some(wanted);
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
        let border = crate::tui::widgets::border_color(focused, theme);
        let inner = crate::tui::widgets::rounded(frame, area, "Digest", border, background, theme);
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

/// Which pane of the view screen has the focus, in Tab order.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Pane {
    Transcript,
    /// The speaker chips, when the session has speakers.
    Speakers,
    Digest,
}

pub struct ViewScreen {
    session_dir: PathBuf,
    /// None: the session has no speakers.json.
    speakers: Option<SavedSpeakers>,
    /// The speaker whose name prompt is open.
    awaiting_speaker: Option<u32>,
    /// The chip clicked once: its lines are highlighted, a second click renames.
    selected_speaker: Option<u32>,
    transcript: TranscriptPanel,
    digest: DigestPanel,
    status: StatusBar,
    focus: Pane,
}

impl ViewScreen {
    pub fn new(session_dir: PathBuf) -> Self {
        let mut status = StatusBar::default();
        status.mode_label = "View".into();
        status.keybinding_hints = "[c] copy  [o] open  [Tab] switch  [q] back".into();
        status.set_stopped(true);
        Self {
            session_dir,
            speakers: None,
            awaiting_speaker: None,
            selected_speaker: None,
            transcript: TranscriptPanel::default(),
            digest: DigestPanel::default(),
            status,
            focus: Pane::Transcript,
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

    /// The labeled transcript when the session has one, else the plain one.
    fn load_transcript(&mut self, ctx: &mut Ctx) {
        self.transcript.clear();
        let labeled = self.session_dir.join(SPEAKER_TRANSCRIPT);
        let path = match labeled.is_file() {
            true => Some(labeled),
            false => TRANSCRIPT.resolve(&self.session_dir),
        };
        let text = self.read_saved(path, ctx);
        // The saved lines already carry their timestamps; write them as they are.
        for line in text.lines().filter(|line| !line.trim().is_empty()) {
            self.transcript.all_text.push(line.to_string());
            self.transcript.view.write(transcript_line(line, None));
        }
    }

    fn load_speakers(&mut self, ctx: &mut Ctx) {
        match saved_speakers::load(&self.session_dir) {
            Ok(speakers) => self.speakers = speakers,
            Err(message) => {
                error!(target: "ltn.view", "{message}");
                ctx.warn(message, Duration::from_secs(8));
            }
        }
    }

    fn rename_speaker(&mut self, speaker: u32, name: &str, ctx: &mut Ctx) {
        match saved_speakers::rename(&self.session_dir, speaker, name) {
            Ok(speakers) => {
                self.speakers = Some(speakers);
                self.load_transcript(ctx);
                self.sync_highlight();
            }
            Err(message) => {
                error!(target: "ltn.view", "Cannot rename S{speaker}: {message}");
                ctx.error(
                    format!("Cannot rename S{speaker}: {message}"),
                    Duration::from_secs(8),
                );
            }
        }
    }

    /// Highlight the selected speaker's lines under the label they carry now.
    fn ask_speaker_name(&mut self, speaker: u32) -> Action {
        let current = self
            .speakers
            .as_ref()
            .and_then(|saved| saved.names.get(&speaker).cloned())
            .unwrap_or_default();
        self.awaiting_speaker = Some(speaker);
        Action::Push(Box::new(LabelModal::new(
            &current,
            &format!("Name for S{speaker}"),
            "Speaker name...",
        )))
    }

    fn speaker_order(&self) -> Vec<u32> {
        self.speakers
            .as_ref()
            .map(|saved| saved.chips().iter().map(|chip| chip.0).collect())
            .unwrap_or_default()
    }

    /// Tab order: transcript, speakers (when the session has any), digest.
    fn move_focus(&mut self, step: isize) {
        let speakers = self.speaker_order();
        let mut order = vec![Pane::Transcript];
        if !speakers.is_empty() {
            order.push(Pane::Speakers);
        }
        order.push(Pane::Digest);
        let at = order
            .iter()
            .position(|pane| *pane == self.focus)
            .unwrap_or(0) as isize;
        self.focus = order[(at + step).rem_euclid(order.len() as isize) as usize];
        if self.focus == Pane::Speakers {
            select_on_entry(&speakers, &mut self.selected_speaker);
            self.sync_highlight();
        }
    }

    fn sync_highlight(&mut self) {
        self.transcript.highlight = self
            .selected_speaker
            .zip(self.speakers.as_ref())
            .map(|(speaker, saved)| saved.label(speaker));
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
        self.load_speakers(ctx);
        self.load_transcript(ctx);
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
        let chips = self
            .speakers
            .as_ref()
            .map(SavedSpeakers::chips)
            .unwrap_or_default();
        let chip_rows = wrap_chips(&chips, left);
        let bar_rows = (chip_rows.len() as u16).min(body.height / 2);
        if bar_rows > 0 {
            let bar = Rect::new(body.x, body.bottom() - bar_rows, left, bar_rows);
            let focused = self.focus == Pane::Speakers;
            render_speaker_bar(frame, bar, &chip_rows, self.selected_speaker, focused, ctx);
        }
        let transcript_area = Rect::new(body.x, body.y, left, body.height - bar_rows);
        let digest_area = Rect::new(body.x + left, body.y, body.width - left, body.height);
        self.transcript.render(
            frame,
            transcript_area,
            self.focus == Pane::Transcript,
            theme,
        );
        self.digest
            .render(frame, digest_area, self.focus == Pane::Digest, theme);
        ctx.hit(transcript_area, "transcript", 0);
        ctx.hit(digest_area, "digest", 0);
        self.status.render(frame, bottom, theme);
        crate::tui::widgets::hint_targets(frame, bottom, ctx);
    }

    /// The speaker name prompt closed: Escape or an empty name changes nothing.
    fn resumed(&mut self, result: Option<Payload>, ctx: &mut Ctx) -> Action {
        let Some(speaker) = self.awaiting_speaker.take() else {
            return Action::None;
        };
        if let Some(name) = result.and_then(|payload| payload.downcast::<String>().ok()) {
            self.rename_speaker(speaker, &name, ctx);
        }
        Action::None
    }

    /// A click focuses the panel under it; the wheel scrolls that panel.
    fn mouse(&mut self, input: &MouseInput, _ctx: &mut Ctx) -> Action {
        if let (Some(index), MouseKind::Click) = (input.on("speaker"), input.kind) {
            let speaker = index as u32;
            // A click puts the focus on the bar, so Enter renames the chip too.
            self.focus = Pane::Speakers;
            // The first click shows the speaker's lines; a second click on it renames.
            if self.selected_speaker != Some(speaker) {
                self.selected_speaker = Some(speaker);
                self.sync_highlight();
                return Action::None;
            }
            return self.ask_speaker_name(speaker);
        }
        let digest = input.on("digest").is_some();
        if !digest && input.on("transcript").is_none() {
            return Action::None;
        }
        let view = match digest {
            true => &mut self.digest.view,
            false => &mut self.transcript.view,
        };
        match input.kind {
            MouseKind::Click => {
                self.focus = match digest {
                    true => Pane::Digest,
                    false => Pane::Transcript,
                };
                if !digest {
                    // A click in the transcript drops the highlight.
                    self.selected_speaker = None;
                    self.sync_highlight();
                }
            }
            MouseKind::WheelUp => view.scroll_by(-1),
            MouseKind::WheelDown => view.scroll_by(1),
        }
        Action::None
    }

    fn key(&mut self, _key: &KeyEvent, name: &str, ctx: &mut Ctx) -> Action {
        // `q` is a priority binding: it wins over the focused panel.
        if action_for(&VIEW_BINDINGS, name) == Some("quit_app") {
            return Action::Pop(None);
        }
        if self.focus == Pane::Speakers {
            let speakers = self.speaker_order();
            match speaker_bar_key(name, &speakers, &mut self.selected_speaker) {
                Some(BarKey::Rename(speaker)) => return self.ask_speaker_name(speaker),
                Some(BarKey::Selected) => {
                    self.sync_highlight();
                    return Action::None;
                }
                None => {}
            }
        }
        let digest_focused = self.focus == Pane::Digest;
        let panel_table: &[Binding] = match digest_focused {
            true => &DIGEST_PANEL_BINDINGS,
            false => &TRANSCRIPT_PANEL_BINDINGS,
        };
        match action_for(panel_table, name) {
            Some(_) if self.focus == Pane::Speakers => {}
            Some("copy_content") if digest_focused => self.digest.copy("", ctx),
            Some("copy_content") => self.transcript.copy("", ctx),
            Some(_) if digest_focused => {
                self.digest.view.key(name);
            }
            Some(_) => {
                self.transcript.view.key(name);
            }
            None => match action_for(&VIEW_BINDINGS, name) {
                Some("open_session_dir") => open_session_dir(&self.session_dir, ctx),
                Some("app.focus_next") => self.move_focus(1),
                Some("app.focus_previous") => self.move_focus(-1),
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
