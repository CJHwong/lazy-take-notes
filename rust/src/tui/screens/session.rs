//! What the transcribe and record screens share (Python's `BaseApp`): the
//! panels, the speaker bar, help and the modals. The session itself, with its
//! digests, quick actions, label and quit flow, is `app::session`. This view
//! forwards input to it and draws what it holds.

use std::path::PathBuf;

use crossterm::event::KeyEvent;
use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};

use crate::adapters::session_controller::SessionController;
use crate::app::session::{SessionRuntime, Step, format_spoken_time};
use crate::domain::transcript::{TranscriptSegment, format_wall_time};
use crate::tui::host::{Action, Ctx, MouseInput, MouseKind, Payload};
use crate::tui::keys::{Binding, SCREEN_BASE, bind};
use crate::tui::screens::modals::{DownloadModal, LabelModal, MarkdownModal};
use crate::tui::screens::view::{
    DIGEST_PANEL_BINDINGS, DigestPanel, TRANSCRIPT_PANEL_BINDINGS, TranscriptPanel, header,
    open_session_dir,
};
use crate::tui::widgets::split_bars;
use crate::tui::widgets::status_bar::StatusBar;
use crate::tui::widgets::text_area::TextArea;

pub const SESSION_BINDINGS: [Binding; 8] = [
    bind("tab", "app.focus_next", "Switch Panel").hidden(),
    SCREEN_BASE[1],
    SCREEN_BASE[2],
    SCREEN_BASE[3],
    bind("q", "quit_app", "Quit").priority(),
    bind("l", "rename_session", "Label").hidden(),
    bind("o", "open_session_dir", "Open").hidden(),
    bind("h", "show_help", "Help").priority(),
];

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

/// Which panel has the focus, in Textual's tab order.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Focus {
    Transcript,
    /// The speaker chips under the transcript, once a speaker is heard.
    Speakers,
    Digest,
    Context,
}

impl Focus {
    const ORDER: [Self; 4] = [
        Self::Transcript,
        Self::Speakers,
        Self::Digest,
        Self::Context,
    ];

    /// The next stop in the Tab order (`step` 1) or the one before (`step` -1).
    /// The speaker bar is a stop only while it shows chips.
    fn moved(self, step: isize, has_speakers: bool) -> Self {
        let count = Self::ORDER.len() as isize;
        let mut at = Self::ORDER.iter().position(|f| *f == self).unwrap_or(0) as isize;
        loop {
            at = (at + step).rem_euclid(count);
            let focus = Self::ORDER[at as usize];
            if focus != Self::Speakers || has_speakers {
                return focus;
            }
        }
    }
}

/// A screen's key hints for a session state.
pub type Hints = fn(&str) -> &'static str;

/// A session and the view that shows it, before a record or transcribe
/// screen takes them.
pub struct Session {
    pub runtime: SessionRuntime,
    pub view: SessionView,
}

impl Session {
    pub fn new(
        controller: SessionController,
        output_dir: PathBuf,
        label: &str,
        missing_digest: Vec<String>,
        missing_interactive: Vec<String>,
    ) -> Self {
        Self {
            runtime: SessionRuntime::new(
                controller,
                output_dir,
                label,
                missing_digest,
                missing_interactive,
            ),
            view: SessionView::default(),
        }
    }
}

/// What a closed modal gave back to the session.
pub enum Resumed {
    Nothing,
    SpeakerName(u32, String),
    Label(String),
}

pub struct SessionView {
    pub transcript: TranscriptPanel,
    pub digest: DigestPanel,
    pub context: TextArea,
    pub status: StatusBar,
    pub focus: Focus,
    /// The key hints for each session state.
    pub hints: Hints,
    /// The help modal's key table rows.
    pub help_rows: &'static [&'static str],
    awaiting_label: bool,
    /// The speaker whose name prompt is open.
    awaiting_speaker: Option<u32>,
    /// The chip clicked once: its lines are highlighted, a second click renames.
    selected_speaker: Option<u32>,
    /// What the view drew of the session so far.
    shown_lines: usize,
    shown_relabel: u64,
    shown_digest: u64,
    shown_state: &'static str,
}

impl Default for SessionView {
    fn default() -> Self {
        Self {
            transcript: TranscriptPanel::default(),
            digest: DigestPanel::default(),
            context: TextArea::default(),
            status: StatusBar::default(),
            focus: Focus::Transcript,
            hints: |_| "",
            help_rows: &[],
            awaiting_label: false,
            awaiting_speaker: None,
            selected_speaker: None,
            shown_lines: 0,
            shown_relabel: 0,
            shown_digest: 0,
            shown_state: "",
        }
    }
}

impl SessionView {
    fn header_text(&self, rt: &SessionRuntime) -> String {
        let metadata = &rt.controller.template().metadata;
        header_text(
            &metadata.name,
            &metadata.description,
            &metadata.locale,
            &rt.label,
        )
    }

    /// Python's `on_mount` for the shared parts.
    pub fn mount(&mut self, rt: &mut SessionRuntime) {
        self.status.quick_action_hints = rt
            .controller
            .template()
            .quick_actions
            .iter()
            .enumerate()
            .map(|(index, action)| format!("[{}] {}", index + 1, action.label))
            .collect::<Vec<_>>()
            .join("  ");
        self.status.buf_max = rt.controller.config().digest.min_lines;
        self.status.ai_notes = rt.missing_digest.is_empty();
        if !rt.missing_digest.is_empty() {
            let pulls: Vec<String> = rt
                .missing_digest
                .iter()
                .map(|model| format!("`ollama pull {model}`"))
                .collect();
            self.digest.update_digest(&format!(
                "**LLM model unavailable**\n\nDigests are disabled. To enable:\n\n{}\n\nThen restart.",
                pulls.join("\n\n")
            ));
        }
        self.sync(rt);
    }

    /// Bring the panels and the status bar up to the session's state.
    pub fn sync(&mut self, rt: &mut SessionRuntime) {
        let state = &mut rt.status;
        let bar = &mut self.status;
        bar.sync_from(state);
        if state.state != self.shown_state {
            self.shown_state = state.state;
            bar.keybinding_hints = (self.hints)(state.state).into();
        }
        if state.context_locked {
            self.context.read_only = true;
        }
        if state.digest_version != self.shown_digest {
            self.shown_digest = state.digest_version;
            self.digest.update_digest(&state.digest);
        }
        self.sync_lines(rt);
    }

    /// New lines go at the end; once speakers are known, every line is
    /// rewritten with its label.
    fn sync_lines(&mut self, rt: &SessionRuntime) {
        let segments = &rt.controller.all_segments;
        let relabeled = rt.status.relabel_version != self.shown_relabel;
        let grown = segments.len() > self.shown_lines;
        if relabeled || (grown && !rt.controller.speaker_turns.is_empty()) {
            self.render_all_lines(rt);
        } else if grown {
            for segment in &segments[self.shown_lines..] {
                self.append_line(segment, None);
            }
        }
        self.shown_relabel = rt.status.relabel_version;
        self.shown_lines = segments.len();
    }

    /// What the session asked for next, as a screen action.
    pub fn show(&mut self, step: Step) -> Action {
        match step {
            Step::None => Action::None,
            Step::ShowQuery {
                title,
                body,
                is_error,
            } => Action::Push(Box::new(MarkdownModal::query(&title, &body, is_error))),
            Step::ShowDownload { model } => Action::Push(Box::new(DownloadModal::new(&model))),
            Step::Close => Action::Pop(None),
        }
    }

    /// Draw the session as it is now. The view catches up with it first.
    pub fn render(&mut self, rt: &mut SessionRuntime, frame: &mut Frame, area: Rect, ctx: &Ctx) {
        self.sync(rt);
        let theme = &ctx.theme;
        let (top, body, bottom) = split_bars(area, 1, self.status.height());
        header(frame, top, &self.header_text(rt), theme);
        let left = body.width / 2;
        // The speaker bar is `height: auto`: one row once a speaker is heard.
        let chips = rt.speaker_chips();
        let chip_rows = wrap_chips(&chips, body.width / 2);
        let bar_rows = (chip_rows.len() as u16).min(body.height / 2);
        let transcript_area = Rect::new(body.x, body.y, left, body.height - bar_rows);
        self.transcript.render(
            frame,
            transcript_area,
            self.focus == Focus::Transcript,
            theme,
        );
        ctx.hit(transcript_area, "transcript", 0);
        if bar_rows > 0 {
            let bar = Rect::new(body.x, body.bottom() - bar_rows, left, bar_rows);
            let focused = self.focus == Focus::Speakers;
            render_speaker_bar(frame, bar, &chip_rows, self.selected_speaker, focused, ctx);
        }
        let column = Rect::new(body.x + left, body.y, body.width - left, body.height);
        // `height: 6`, and 40% of the column while it has the focus.
        let context_rows = match self.focus {
            Focus::Context => column.height * 40 / 100,
            _ => 6,
        }
        .min(column.height);
        let digest_area = Rect::new(
            column.x,
            column.y,
            column.width,
            column.height - context_rows,
        );
        self.digest
            .render(frame, digest_area, self.focus == Focus::Digest, theme);
        ctx.hit(digest_area, "digest", 0);
        let context_area = Rect::new(column.x, digest_area.bottom(), column.width, context_rows);
        self.context.render(
            frame,
            context_area,
            self.focus == Focus::Context,
            "Session Context",
            "primary",
            theme,
        );
        ctx.hit(context_area, "context", 0);
        self.status.render(frame, bottom, theme);
        crate::tui::widgets::hint_targets(frame, bottom, ctx);
    }

    /// A click focuses the panel under it; the wheel scrolls the transcript or
    /// the digest under the pointer.
    /// A click on a speaker chip opens its name prompt, as Python's `on_click`.
    pub fn mouse(&mut self, rt: &SessionRuntime, input: &MouseInput) -> Action {
        if let (Some(index), MouseKind::Click) = (input.on("speaker"), input.kind) {
            let speaker = index as u32;
            // A click puts the focus on the bar, so Enter renames the chip too.
            self.focus = Focus::Speakers;
            // The first click shows the speaker's lines; a second click on it renames.
            if self.selected_speaker != Some(speaker) {
                self.selected_speaker = Some(speaker);
                self.sync_highlight(rt);
                return Action::None;
            }
            return self.ask_speaker_name(rt, speaker);
        }
        self.scroll_or_focus(rt, input);
        Action::None
    }

    fn ask_speaker_name(&mut self, rt: &SessionRuntime, speaker: u32) -> Action {
        let current = rt
            .controller
            .speaker_names
            .get(&speaker)
            .cloned()
            .unwrap_or_default();
        self.awaiting_speaker = Some(speaker);
        Action::Push(Box::new(LabelModal::new(
            &current,
            &format!("Name for S{speaker}"),
            "Speaker name...",
        )))
    }

    fn move_focus(&mut self, rt: &SessionRuntime, step: isize) {
        let speakers = speaker_order(rt);
        self.focus = self.focus.moved(step, !speakers.is_empty());
        if self.focus == Focus::Speakers {
            select_on_entry(&speakers, &mut self.selected_speaker);
            self.sync_highlight(rt);
        }
    }

    fn scroll_or_focus(&mut self, rt: &SessionRuntime, input: &MouseInput) {
        let focus = if input.on("transcript").is_some() {
            Focus::Transcript
        } else if input.on("digest").is_some() {
            Focus::Digest
        } else if input.on("context").is_some() {
            Focus::Context
        } else {
            return;
        };
        let step = match input.kind {
            MouseKind::Click => {
                self.focus = focus;
                if focus == Focus::Transcript {
                    // A click in the transcript drops the highlight.
                    self.selected_speaker = None;
                    self.sync_highlight(rt);
                }
                return;
            }
            MouseKind::WheelUp => -1,
            MouseKind::WheelDown => 1,
        };
        match focus {
            Focus::Transcript => self.transcript.view.scroll_by(step),
            Focus::Digest => self.digest.view.scroll_by(step),
            Focus::Speakers | Focus::Context => {}
        }
    }

    /// The text the panels copy after the transcript: the context once it is locked.
    fn copy_suffix(&self) -> String {
        match self.context.read_only {
            true => self.context.text().trim().to_string(),
            false => String::new(),
        }
    }

    /// Keys the focused panel or text box takes first. None when it does not take the key.
    pub fn focused_key(
        &mut self,
        rt: &mut SessionRuntime,
        key: &KeyEvent,
        name: &str,
        ctx: &mut Ctx,
    ) -> Option<Action> {
        match self.focus {
            Focus::Context => {
                if !self.context.key(key, name) {
                    return None;
                }
                rt.set_user_context(self.context.text());
            }
            Focus::Transcript => {
                match crate::tui::keys::action_for(&TRANSCRIPT_PANEL_BINDINGS, name)? {
                    "copy_content" => self.transcript.copy(&self.copy_suffix(), ctx),
                    _ => {
                        self.transcript.view.key(name);
                    }
                }
            }
            Focus::Speakers => {
                let speakers = speaker_order(rt);
                match speaker_bar_key(name, &speakers, &mut self.selected_speaker)? {
                    BarKey::Rename(speaker) => return Some(self.ask_speaker_name(rt, speaker)),
                    BarKey::Selected => self.sync_highlight(rt),
                }
            }
            Focus::Digest => match crate::tui::keys::action_for(&DIGEST_PANEL_BINDINGS, name)? {
                "copy_content" => self.digest.copy(&self.copy_suffix(), ctx),
                _ => {
                    self.digest.view.key(name);
                }
            },
        }
        Some(Action::None)
    }

    /// A printable key the focused, editable context box takes before any binding.
    pub fn typed_into_context(
        &mut self,
        rt: &mut SessionRuntime,
        key: &KeyEvent,
        name: &str,
    ) -> bool {
        if self.focus != Focus::Context || !self.context.consumes(key) {
            return false;
        }
        self.context.key(key, name);
        rt.set_user_context(self.context.text());
        true
    }

    pub fn paste(&mut self, rt: &mut SessionRuntime, text: &str) {
        if self.focus == Focus::Context {
            self.context.paste(text);
            rt.set_user_context(self.context.text());
        }
    }

    /// The shared screen keys after the focused panel passed. None when not one of them.
    pub fn screen_key(
        &mut self,
        rt: &mut SessionRuntime,
        name: &str,
        ctx: &mut Ctx,
    ) -> Option<Action> {
        let action = crate::tui::keys::action_for(&SESSION_BINDINGS, name);
        match action {
            Some("app.focus_next") => self.move_focus(rt, 1),
            Some("app.focus_previous") => self.move_focus(rt, -1),
            Some("rename_session") => {
                self.awaiting_label = true;
                return Some(Action::Push(Box::new(LabelModal::session(&rt.label))));
            }
            Some("open_session_dir") if self.status.stopped => {
                open_session_dir(&rt.output_dir, ctx)
            }
            Some("open_session_dir") => {}
            Some("show_help") => return Some(self.show_help(rt)),
            _ => return self.quick_action_key(rt, name, ctx),
        }
        Some(Action::None)
    }

    fn quick_action_key(
        &mut self,
        rt: &mut SessionRuntime,
        name: &str,
        ctx: &mut Ctx,
    ) -> Option<Action> {
        let count = rt.quick_action_count();
        let index: usize = name.parse().ok()?;
        if index == 0 || index > count {
            return None;
        }
        // The key that started it: a press, or a click on its hint (the host logs those).
        tracing::info!(target: "ltn.app", "quick action {index} started by key {name:?}");
        rt.quick_action(name, ctx);
        Some(Action::None)
    }

    fn show_help(&self, rt: &SessionRuntime) -> Action {
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
        lines.extend(self.help_rows.iter().map(|row| row.to_string()));
        Action::Push(Box::new(MarkdownModal::help(&lines.join("\n"))))
    }

    /// One transcript line: `[hh:mm:ss]` dim, then the text.
    fn append_line(&mut self, segment: &TranscriptSegment, label: Option<&str>) {
        let timestamp = format_wall_time(segment.wall_start);
        let plain = match label {
            Some(label) => format!("{label}: {}", segment.text),
            None => segment.text.clone(),
        };
        self.transcript
            .all_text
            .push(format!("[{timestamp}] {plain}"));
        self.transcript.view.write(Line::from(vec![
            Span::styled(
                format!("[{timestamp}]"),
                Style::default().add_modifier(Modifier::DIM),
            ),
            Span::raw(format!(" {plain}")),
        ]));
    }

    /// Highlight the selected speaker's lines under the label they carry now.
    fn sync_highlight(&mut self, rt: &SessionRuntime) {
        self.transcript.highlight = self.selected_speaker.map(|speaker| {
            rt.controller
                .speaker_names
                .get(&speaker)
                .filter(|name| !name.is_empty())
                .cloned()
                .unwrap_or_else(|| format!("S{speaker}"))
        });
    }

    /// Rewrite every line with its speaker label (a diarization pass relabels old lines).
    fn render_all_lines(&mut self, rt: &SessionRuntime) {
        self.transcript.clear();
        self.sync_highlight(rt);
        for segment in &rt.controller.all_segments {
            let label = rt.controller.speaker_label(segment);
            self.append_line(segment, label.as_deref());
        }
    }

    /// The label modal or another screen this one pushed closed.
    pub fn resumed(&mut self, result: Option<Payload>) -> Resumed {
        let text = result.and_then(|payload| payload.downcast::<String>().ok());
        if let Some(speaker) = self.awaiting_speaker.take() {
            // Escape or an empty name keeps the old one.
            return match text {
                Some(name) => Resumed::SpeakerName(speaker, *name),
                None => Resumed::Nothing,
            };
        }
        if !std::mem::take(&mut self.awaiting_label) {
            return Resumed::Nothing;
        }
        match text {
            Some(label) => Resumed::Label(*label),
            None => Resumed::Nothing,
        }
    }
}

/// Each speaker heard so far: number, name, and seconds of speech.
fn speaker_order(rt: &SessionRuntime) -> Vec<u32> {
    rt.speaker_chips().iter().map(|chip| chip.0).collect()
}

/// One chip per speaker: `padding: 0 1` on the bar, and each chip `padding: 0 1`,
/// `margin-right: 1` on `$surface`. Names are the user's text, drawn as typed.
/// One chip: the speaker, its text (` name 4m12s `), and its width in cells.
pub type Chip = (u32, String, u16);

/// The chips in rows of `width` cells: a leading space, then each chip and a gap.
/// A chip wider than the row gets a row of its own.
pub fn wrap_chips(chips: &[(u32, String, f64)], width: u16) -> Vec<Vec<Chip>> {
    let mut rows: Vec<Vec<Chip>> = Vec::new();
    let mut used = 0u16;
    for (speaker, name, seconds) in chips {
        let shown = match name.is_empty() {
            true => format!("S{speaker}"),
            false => name.clone(),
        };
        let text = format!(" {shown} {} ", format_spoken_time(*seconds));
        let cells = unicode_width::UnicodeWidthStr::width(text.as_str()) as u16;
        let needed = cells + 1;
        if rows.is_empty() || used + needed > width {
            rows.push(Vec::new());
            used = 1;
        }
        used = used.saturating_add(needed);
        if let Some(row) = rows.last_mut() {
            row.push((*speaker, text, cells));
        }
    }
    rows
}

/// The chips under the transcript, one line per row; each chip is a click target.
/// `selected`: the chip clicked once, drawn in the primary color.
pub fn render_speaker_bar(
    frame: &mut Frame,
    area: Rect,
    rows: &[Vec<Chip>],
    selected: Option<u32>,
    focused: bool,
    ctx: &Ctx,
) {
    let theme = &ctx.theme;
    let chip = Style::default()
        .bg(theme.color("surface"))
        .fg(theme.on("foreground", "surface"));
    let chosen = Style::default()
        .bg(theme.color("primary"))
        .fg(theme.on("foreground", "primary"))
        .add_modifier(Modifier::BOLD);
    // The bar has no border; the underline shows that it has the focus.
    let chosen = match focused {
        true => chosen.add_modifier(Modifier::UNDERLINED),
        false => chosen,
    };
    let base = Style::default()
        .bg(theme.color("background"))
        .fg(theme.color("foreground"));
    for (offset, row) in rows.iter().take(area.height as usize).enumerate() {
        let y = area.y + offset as u16;
        let mut spans = vec![Span::raw(" ")];
        let mut x = area.x + 1;
        for (speaker, text, cells) in row {
            let visible = (*cells).min(area.right().saturating_sub(x));
            ctx.hit(Rect::new(x, y, visible, 1), "speaker", *speaker as usize);
            x = x.saturating_add(cells + 1);
            let style = match selected == Some(*speaker) {
                true => chosen,
                false => chip,
            };
            spans.push(Span::styled(text.clone(), style));
            spans.push(Span::raw(" "));
        }
        frame.render_widget(
            ratatui::widgets::Paragraph::new(Line::from(spans)).style(base),
            Rect::new(area.x, y, area.width, 1),
        );
    }
}

/// What a key on the focused speaker bar asks for.
pub enum BarKey {
    /// The selection moved or was dropped: redraw the highlight.
    Selected,
    /// Open the name prompt for this speaker.
    Rename(u32),
}

/// A key on the focused speaker bar, with `speakers` in chip order.
/// None when the bar does not take the key.
pub fn speaker_bar_key(name: &str, speakers: &[u32], selected: &mut Option<u32>) -> Option<BarKey> {
    // A selected speaker that a newer pass dropped counts as no selection.
    let at = selected.and_then(|speaker| speakers.iter().position(|s| *s == speaker));
    let last = speakers.len().saturating_sub(1);
    match (name, at) {
        ("enter", Some(at)) => return Some(BarKey::Rename(speakers[at])),
        ("enter", None) => {}
        ("left" | "right", None) => *selected = speakers.first().copied(),
        ("left", Some(at)) => *selected = Some(speakers[at.saturating_sub(1)]),
        ("right", Some(at)) => *selected = Some(speakers[(at + 1).min(last)]),
        ("escape", _) => *selected = None,
        _ => return None,
    }
    Some(BarKey::Selected)
}

/// The focus reached the bar: keep a selection it still shows, else select the first chip.
pub fn select_on_entry(speakers: &[u32], selected: &mut Option<u32>) {
    if !selected.is_some_and(|speaker| speakers.contains(&speaker)) {
        *selected = speakers.first().copied();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_header_drops_a_locale_that_reads_as_markup() {
        assert_eq!(
            header_text("Default", "General meeting digest (English)", "en", "x"),
            "  lazy-take-notes | Default - General meeting digest (English)  \u{2014} x"
        );
        assert_eq!(
            header_text("Mine", "", "EN", "sprint"),
            "  lazy-take-notes | Mine  \u{2014} sprint"
        );
        assert_eq!(
            header_text("Mine", "", "1x", ""),
            "  lazy-take-notes | Mine [1x]"
        );
    }
}
