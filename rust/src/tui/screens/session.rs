//! What the transcribe and record screens share (Python's `BaseApp`): the
//! panels, the digest and quick-action tasks, the label, help and the quit flow.
//!
//! The controller stays on the screen. A digest is three steps:
//! `begin_digest` here, the LLM call on a task, `finish_digest` when the reply
//! comes back as a message. A newer digest aborts the running one and abandons
//! its prompt at once, so the history never holds two prompts.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use crossterm::event::KeyEvent;
use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use tokio::task::AbortHandle;
use tracing::{debug, error};

use crate::adapters::os_error::OsError;
use crate::adapters::session_controller::SessionController;
use crate::app::container::safe_label;
use crate::app::logging::see_debug_log;
use crate::domain::session_files::SPEAKER_TRANSCRIPT;
use crate::domain::speaker_turn::SpeakerTurn;
use crate::domain::transcript::{TranscriptSegment, format_wall_time};
use crate::ports::llm::{ChatResponse, LlmError};
use crate::tui::host::{Action, Ctx, Payload};
use crate::tui::keys::{Binding, SCREEN_BASE, bind};
use crate::tui::screens::modals::{DownloadModal, LabelModal, MarkdownModal};
use crate::tui::screens::view::{
    DIGEST_PANEL_BINDINGS, DigestPanel, TRANSCRIPT_PANEL_BINDINGS, TranscriptPanel, header,
    open_session_dir,
};
use crate::tui::widgets::split_bars;
use crate::tui::widgets::status_bar::StatusBar;
use crate::tui::widgets::text_area::TextArea;
use crate::usecases::digest::DigestRequest;

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

/// Most file systems cap one name at 255 bytes; the timestamp prefix and the
/// separator use 18. CJK labels use 3 bytes per character.
const LABEL_MAX_BYTES: usize = 200;

/// Cut `label` to `LABEL_MAX_BYTES` without splitting a character.
pub fn fit_name(label: &str) -> String {
    let mut end = label.len().min(LABEL_MAX_BYTES);
    while !label.is_char_boundary(end) {
        end -= 1;
    }
    label[..end].to_string()
}

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

/// The quit and final-digest state. No screen code, so the rules have unit tests.
#[derive(Default, Debug)]
pub struct Flow {
    pub digest_running: bool,
    pub query_running: bool,
    pub label_running: bool,
    pub final_done: bool,
    pub pending_quit: bool,
    /// Never reset: a pending quit tries the final digest once, then closes even if it failed.
    pub final_attempted: bool,
    pub final_started: bool,
    pub session_closed: bool,
}

/// What a pending quit does next.
#[derive(Debug, PartialEq, Eq)]
pub enum QuitStep {
    Wait,
    RunFinal,
    Close,
}

impl Flow {
    pub fn quit_step(&self, needs_final: bool, worker_busy: bool) -> QuitStep {
        if !self.pending_quit || self.digest_running || self.label_running || worker_busy {
            return QuitStep::Wait;
        }
        if needs_final && !self.final_attempted {
            return QuitStep::RunFinal;
        }
        QuitStep::Close
    }
}

/// A digest reply, back from its task.
pub struct DigestReply {
    generation: u64,
    reply: Result<ChatResponse, LlmError>,
}

struct InFlight {
    generation: u64,
    request: DigestRequest,
    is_final: bool,
    task: AbortHandle,
}

/// A quick action's result, back from its task.
pub struct QueryReply {
    label: String,
    result: Option<Result<(String, String), String>>,
}

/// A digest whose prompt could not be built. It arrives as a message, as
/// Python's worker posts its failure.
pub struct DigestFailed {
    generation: u64,
    error: String,
    is_final: bool,
}

/// The AI label, back from its task.
pub struct LabelReply(Option<String>);

/// Which panel has the focus, in Textual's tab order.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Focus {
    Transcript,
    Digest,
    Context,
}

impl Focus {
    fn next(self) -> Self {
        match self {
            Self::Transcript => Self::Digest,
            Self::Digest => Self::Context,
            Self::Context => Self::Transcript,
        }
    }

    fn previous(self) -> Self {
        match self {
            Self::Transcript => Self::Context,
            Self::Digest => Self::Transcript,
            Self::Context => Self::Digest,
        }
    }
}

/// What differs between transcribe and record.
pub trait Mode {
    /// The key hints for a worker state.
    fn hints(&self, state: &str) -> &'static str;
    /// The help modal's key table rows.
    fn help_rows(&self) -> &'static [&'static str];
    /// True while the worker still delivers transcript lines.
    fn worker_busy(&self) -> bool;
    /// Move the session folder.
    fn move_folder(&mut self, from: &Path, to: &Path) -> std::io::Result<()>;
}

pub struct Session {
    pub controller: SessionController,
    pub output_dir: PathBuf,
    pub label: String,
    pub missing_digest: Vec<String>,
    pub missing_interactive: Vec<String>,
    pub transcript: TranscriptPanel,
    pub digest: DigestPanel,
    pub context: TextArea,
    pub status: StatusBar,
    pub focus: Focus,
    pub flow: Flow,
    /// Periodic digests start on their own while recording; a file waits for its end.
    pub auto_digest: bool,
    generation: u64,
    in_flight: Option<InFlight>,
    last_persist_error: Option<String>,
    awaiting_label: bool,
    download_open: bool,
}

/// Readable speech total: 42s, 4m58s, 1h12m (Python's `format_spoken_time`).
pub fn format_spoken_time(seconds: f64) -> String {
    let total = seconds.max(0.0) as u64;
    if total >= 3600 {
        return format!("{}h{}m", total / 3600, (total % 3600) / 60);
    }
    if total >= 60 {
        return format!("{}m{}s", total / 60, total % 60);
    }
    format!("{total}s")
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
            controller,
            output_dir,
            label: label.into(),
            missing_digest,
            missing_interactive,
            transcript: TranscriptPanel::default(),
            digest: DigestPanel::default(),
            context: TextArea::default(),
            status: StatusBar::default(),
            focus: Focus::Transcript,
            flow: Flow::default(),
            auto_digest: true,
            generation: 0,
            in_flight: None,
            last_persist_error: None,
            awaiting_label: false,
            download_open: false,
        }
    }

    fn header_text(&self) -> String {
        let metadata = &self.controller.template().metadata;
        header_text(
            &metadata.name,
            &metadata.description,
            &metadata.locale,
            &self.label,
        )
    }

    pub fn update_hints(&mut self, mode: &dyn Mode, state: &str) {
        self.status.keybinding_hints = mode.hints(state).into();
        self.status.quick_action_hints = self
            .controller
            .template()
            .quick_actions
            .iter()
            .enumerate()
            .map(|(index, action)| format!("[{}] {}", index + 1, action.label))
            .collect::<Vec<_>>()
            .join("  ");
    }

    /// Python's `on_mount` for the shared parts.
    pub fn mount(&mut self, mode: &dyn Mode, ctx: &mut Ctx) {
        self.update_hints(mode, "idle");
        self.status.buf_max = self.controller.config().digest.min_lines;
        self.status.ai_notes = self.missing_digest.is_empty();
        if !self.missing_digest.is_empty() {
            let pulls: Vec<String> = self
                .missing_digest
                .iter()
                .map(|model| format!("`ollama pull {model}`"))
                .collect();
            self.digest.update_digest(&format!(
                "**LLM model unavailable**\n\nDigests are disabled. To enable:\n\n{}\n\nThen restart.",
                pulls.join("\n\n")
            ));
        }
        if !self.missing_interactive.is_empty() {
            let models = self.missing_interactive.join(", ");
            ctx.warn(
                format!(
                    "Quick actions disabled: model {models} not found. Run: ollama pull {models}"
                ),
                Duration::from_secs(10),
            );
        }
    }

    pub fn render(&mut self, frame: &mut Frame, area: Rect, ctx: &Ctx) {
        let theme = &ctx.theme;
        let (top, body, bottom) = split_bars(area, 1, self.status.height());
        header(frame, top, &self.header_text(), theme);
        let left = body.width / 2;
        // The speaker bar is `height: auto`: one row once a speaker is heard.
        let chips = self.speaker_chips();
        let bar_rows = u16::from(!chips.is_empty()).min(body.height);
        self.transcript.render(
            frame,
            Rect::new(body.x, body.y, left, body.height - bar_rows),
            self.focus == Focus::Transcript,
            theme,
        );
        if bar_rows > 0 {
            let bar = Rect::new(body.x, body.bottom() - 1, left, 1);
            render_speaker_bar(frame, bar, &chips, theme);
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
        let context_area = Rect::new(column.x, digest_area.bottom(), column.width, context_rows);
        self.context.render(
            frame,
            context_area,
            self.focus == Focus::Context,
            "Session Context",
            "primary",
            theme,
        );
        self.status.render(frame, bottom, theme);
    }

    /// The text the panels copy after the transcript: the context once it is locked.
    fn copy_suffix(&self) -> String {
        match self.context.read_only {
            true => self.context.text().trim().to_string(),
            false => String::new(),
        }
    }

    /// Keys the focused panel or text box takes first. None when it does not take the key.
    pub fn focused_key(&mut self, key: &KeyEvent, name: &str, ctx: &mut Ctx) -> Option<Action> {
        match self.focus {
            Focus::Context => {
                if !self.context.key(key, name) {
                    return None;
                }
                self.controller.user_context = self.context.text();
            }
            Focus::Transcript => {
                match crate::tui::keys::action_for(&TRANSCRIPT_PANEL_BINDINGS, name)? {
                    "copy_content" => self.transcript.copy(&self.copy_suffix(), ctx),
                    _ => {
                        self.transcript.view.key(name);
                    }
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
    pub fn typed_into_context(&mut self, key: &KeyEvent, name: &str) -> bool {
        if self.focus != Focus::Context || !self.context.consumes(key) {
            return false;
        }
        self.context.key(key, name);
        self.controller.user_context = self.context.text();
        true
    }

    pub fn paste(&mut self, text: &str) {
        if self.focus == Focus::Context {
            self.context.paste(text);
            self.controller.user_context = self.context.text();
        }
    }

    /// The shared screen keys after the focused panel passed. None when not one of them.
    pub fn screen_key(&mut self, mode: &mut dyn Mode, name: &str, ctx: &mut Ctx) -> Option<Action> {
        let action = crate::tui::keys::action_for(&SESSION_BINDINGS, name);
        match action {
            Some("app.focus_next") => self.focus = self.focus.next(),
            Some("app.focus_previous") => self.focus = self.focus.previous(),
            Some("rename_session") => {
                self.awaiting_label = true;
                return Some(Action::Push(Box::new(LabelModal::session(&self.label))));
            }
            Some("open_session_dir") if self.status.stopped => {
                open_session_dir(&self.output_dir, ctx)
            }
            Some("open_session_dir") => {}
            Some("show_help") => return Some(self.show_help(mode)),
            _ => return self.quick_action_key(name, ctx),
        }
        Some(Action::None)
    }

    fn quick_action_key(&mut self, name: &str, ctx: &mut Ctx) -> Option<Action> {
        let count = self.controller.template().quick_actions.len();
        let index: usize = name.parse().ok()?;
        if index == 0 || index > count {
            return None;
        }
        self.quick_action(name, ctx);
        Some(Action::None)
    }

    fn show_help(&self, mode: &dyn Mode) -> Action {
        let metadata = &self.controller.template().metadata;
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
        let min_lines = self.controller.config().digest.min_lines;
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
        lines.extend(mode.help_rows().iter().map(|row| row.to_string()));
        Action::Push(Box::new(MarkdownModal::help(&lines.join("\n"))))
    }

    // --- Worker events ---

    /// Lines from the worker: keep, save, show, and maybe start a digest.
    pub fn on_transcript_chunk(&mut self, segments: &[TranscriptSegment], ctx: &mut Ctx) {
        debug!(
            target: "ltn.app",
            "TranscriptChunk received: {} segments, total_all={}",
            segments.len(),
            self.controller.all_segments.len() + segments.len()
        );
        let should_digest = self.controller.on_transcript_segments(segments);
        self.report_persist_error(ctx);
        if self.controller.speaker_turns.is_empty() {
            for segment in segments {
                self.append_line(segment, None);
            }
        } else {
            self.render_all_lines();
            // The screen labels each new line now; the saved file must not lag until the next pass.
            self.save_speaker_transcript(ctx);
        }
        self.status.buf_count = self.controller.digest_state.buffer.len();
        if self.auto_digest
            && should_digest
            && !self.flow.digest_running
            && self.missing_digest.is_empty()
        {
            self.start_digest(false, ctx);
        }
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

    /// Rewrite every line with its speaker label (a diarization pass relabels old lines).
    fn render_all_lines(&mut self) {
        self.transcript.view.clear();
        self.transcript.all_text.clear();
        let lines: Vec<(TranscriptSegment, Option<String>)> = self
            .controller
            .all_segments
            .iter()
            .map(|segment| (segment.clone(), self.controller.speaker_label(segment)))
            .collect();
        for (segment, label) in lines {
            self.append_line(&segment, label.as_deref());
        }
    }

    /// Each speaker heard so far: number, name, and seconds of speech.
    fn speaker_chips(&self) -> Vec<(u32, String, f64)> {
        let mut speech: std::collections::BTreeMap<u32, f64> = std::collections::BTreeMap::new();
        for turn in &self.controller.speaker_turns {
            *speech.entry(turn.speaker).or_default() += turn.end - turn.start;
        }
        speech
            .into_iter()
            .map(|(speaker, seconds)| {
                let name = self
                    .controller
                    .speaker_names
                    .get(&speaker)
                    .cloned()
                    .unwrap_or_default();
                (speaker, name, seconds)
            })
            .collect()
    }

    /// A diarization pass: keep the turns, save them, relabel the lines.
    pub fn on_speaker_turns(&mut self, turns: Vec<SpeakerTurn>, ctx: &mut Ctx) {
        self.controller.set_speaker_turns(turns);
        self.persist_speakers(ctx);
        self.render_all_lines();
    }

    fn persist_speakers(&mut self, ctx: &mut Ctx) {
        let names: std::collections::BTreeMap<String, &String> = self
            .controller
            .speaker_names
            .iter()
            .map(|(speaker, name)| (speaker.to_string(), name))
            .collect();
        let payload = serde_json::json!({
            "names": names,
            "turns": self.controller.speaker_turns,
        });
        let text = serde_json::to_string_pretty(&payload).unwrap_or_default();
        if let Err(e) = std::fs::write(self.output_dir.join("speakers.json"), text) {
            self.speakers_not_saved("speakers.json", &e, ctx);
            return;
        }
        self.save_speaker_transcript(ctx);
    }

    /// The transcript with the labels shown on screen, in the format `diarize` writes.
    fn save_speaker_transcript(&mut self, ctx: &mut Ctx) {
        let text: String = self
            .controller
            .all_segments
            .iter()
            .map(|segment| {
                let line = match self.controller.speaker_label(segment) {
                    Some(label) => format!("{label}: {}", segment.text),
                    None => segment.text.clone(),
                };
                format!("[{}] {line}\n", format_wall_time(segment.wall_start))
            })
            .collect();
        let path = self.output_dir.join(SPEAKER_TRANSCRIPT);
        if let Err(e) = std::fs::write(path, text) {
            self.speakers_not_saved(SPEAKER_TRANSCRIPT, &e, ctx);
        }
    }

    fn speakers_not_saved(&mut self, name: &str, e: &std::io::Error, ctx: &mut Ctx) {
        let shown = OsError::from_io(e, Some(&self.output_dir.join(name)), None);
        error!(target: "ltn.app", "Could not save {name}: {e}");
        if self.last_persist_error.as_deref() != Some("speakers") {
            self.last_persist_error = Some("speakers".into());
            ctx.warn(
                format!("Could not save speakers: {shown}"),
                Duration::from_secs(10),
            );
        }
    }

    /// Warn once per distinct disk failure; the session keeps going.
    fn report_persist_error(&mut self, ctx: &mut Ctx) {
        let Some(error) = self.controller.take_persist_error() else {
            return;
        };
        if self.last_persist_error.as_deref() == Some(&error) {
            return;
        }
        self.last_persist_error = Some(error.clone());
        ctx.warn(
            format!("{error}{}", see_debug_log()),
            Duration::from_secs(10),
        );
    }

    /// Download progress: show it, and open the download modal once.
    pub fn on_download_progress(&mut self, percent: u32, model_name: &str) -> Action {
        self.status.download_percent = percent as i32;
        self.status.download_model = model_name.into();
        if self.download_open {
            return Action::None;
        }
        self.download_open = true;
        Action::Push(Box::new(DownloadModal::new(model_name)))
    }

    /// The download modal closed itself on this state.
    pub fn download_closed(&mut self) {
        self.download_open = false;
    }

    // --- Digest ---

    /// Start a digest. A running one is aborted and its prompt removed first.
    pub fn start_digest(&mut self, is_final: bool, ctx: &mut Ctx) {
        self.status.activity = match is_final {
            true => "Final digest...".into(),
            false => "Digesting...".into(),
        };
        if let Some(old) = self.in_flight.take() {
            old.task.abort();
            self.controller.abandon_digest(&old.request);
        }
        self.generation += 1;
        self.flow.digest_running = true;
        let generation = self.generation;
        let request = match self.controller.begin_digest(is_final) {
            Ok(request) => request,
            Err(error) => {
                ctx.spawn(async move {
                    DigestFailed {
                        generation,
                        error,
                        is_final,
                    }
                });
                return;
            }
        };
        let llm = self.controller.llm();
        let model = self.controller.digest_model().to_string();
        let messages = request.messages.clone();
        let task = ctx.spawn_or(
            async move {
                DigestReply {
                    generation,
                    reply: llm.chat(&model, &messages).await,
                }
            },
            move |message| DigestReply {
                generation,
                reply: Err(LlmError::new("panic", message)),
            },
        );
        self.in_flight = Some(InFlight {
            generation,
            request,
            is_final,
            task,
        });
    }

    /// The final digest, once until it fails.
    pub fn run_final_digest(&mut self, ctx: &mut Ctx) {
        if self.flow.final_started {
            return;
        }
        self.flow.final_started = true;
        self.flow.final_attempted = true;
        self.start_digest(true, ctx);
    }

    /// A reply came back. A reply for a replaced digest is dropped.
    fn on_digest_reply(&mut self, reply: DigestReply, mode: &dyn Mode, ctx: &mut Ctx) -> Action {
        let current = self
            .in_flight
            .as_ref()
            .is_some_and(|flight| flight.generation == reply.generation);
        if !current {
            return Action::None;
        }
        let Some(flight) = self.in_flight.take() else {
            return Action::None;
        };
        let result = self
            .controller
            .finish_digest(&flight.request, flight.is_final, reply.reply);
        if flight.generation == self.generation {
            self.flow.digest_running = false;
        }
        match result {
            Ok(markdown) => {
                if flight.is_final && !self.flow.pending_quit {
                    ctx.notify_with(
                        "Final digest ready. Press q to quit.",
                        crate::tui::host::Severity::Information,
                        Duration::from_secs(10),
                        "",
                    );
                }
                self.digest_ready(&markdown, flight.is_final, mode, ctx)
            }
            Err(error) => self.digest_failed(&error, flight.is_final, mode, ctx),
        }
    }

    fn on_digest_failed(&mut self, failed: DigestFailed, mode: &dyn Mode, ctx: &mut Ctx) -> Action {
        if failed.generation != self.generation || self.in_flight.is_some() {
            return Action::None;
        }
        self.flow.digest_running = false;
        self.digest_failed(&failed.error, failed.is_final, mode, ctx)
    }

    fn digest_ready(
        &mut self,
        markdown: &str,
        is_final: bool,
        mode: &dyn Mode,
        ctx: &mut Ctx,
    ) -> Action {
        self.digest.update_digest(markdown);
        self.status.buf_count = self.controller.digest_state.buffer.len();
        self.status.last_digest_time = Some(Instant::now());
        self.status.activity.clear();
        self.report_persist_error(ctx);
        if is_final {
            self.flow.final_done = true;
            self.run_label_task(ctx);
        }
        self.continue_pending_quit(mode, ctx)
    }

    fn digest_failed(
        &mut self,
        error: &str,
        is_final: bool,
        mode: &dyn Mode,
        ctx: &mut Ctx,
    ) -> Action {
        if is_final && !self.flow.pending_quit {
            ctx.error(
                "Final digest failed. Press q to quit.",
                Duration::from_secs(10),
            );
        }
        // Allow a retry after a failed final.
        self.flow.final_started = false;
        self.status.activity.clear();
        ctx.error(
            format!("Digest failed: {error}{}", see_debug_log()),
            Duration::from_secs(8),
        );
        self.continue_pending_quit(mode, ctx)
    }

    /// True when the session has notes to finish and AI notes are on.
    pub fn needs_final_digest(&self) -> bool {
        if !self.missing_digest.is_empty() || self.flow.final_done {
            return false;
        }
        let state = &self.controller.digest_state;
        !state.buffer.is_empty() || state.digest_count > 0
    }

    fn ai_off(&self, models: &[String], feature: &str, ctx: &mut Ctx) -> bool {
        if models.is_empty() {
            return false;
        }
        ctx.warn(
            format!(
                "{feature} are off for this session: {} failed the start check. The transcript is still recorded.",
                models.join(", ")
            ),
            Duration::from_secs(6),
        );
        true
    }

    pub fn force_digest(&mut self, ctx: &mut Ctx) {
        if self.ai_off(&self.missing_digest.clone(), "AI notes", ctx) {
            return;
        }
        if self.flow.digest_running {
            ctx.warn("Digest already in progress", Duration::from_secs(3));
            return;
        }
        if self.controller.digest_state.buffer.is_empty() {
            ctx.notify_with(
                "Nothing in buffer to digest yet",
                crate::tui::host::Severity::Information,
                Duration::from_secs(3),
                "",
            );
            return;
        }
        self.start_digest(false, ctx);
    }

    // --- Quick actions and the label ---

    fn quick_action(&mut self, key: &str, ctx: &mut Ctx) {
        if self.ai_off(&self.missing_interactive.clone(), "Quick actions", ctx) {
            return;
        }
        if self.flow.digest_running {
            ctx.warn("Digest in progress, please wait", Duration::from_secs(3));
            return;
        }
        if self.flow.query_running {
            ctx.warn(
                "Quick action already running, please wait",
                Duration::from_secs(3),
            );
            return;
        }
        let actions = &self.controller.template().quick_actions;
        let label = key
            .parse::<usize>()
            .ok()
            .and_then(|index| actions.get(index.wrapping_sub(1)))
            .map_or(key.to_string(), |action| action.label.clone());
        self.status.activity = format!("{label}...");
        self.flow.query_running = true;
        let job = self.controller.quick_action_job(key);
        let fallback = label.clone();
        ctx.spawn_or(
            async move {
                QueryReply {
                    label,
                    result: job.await,
                }
            },
            move |message| QueryReply {
                label: fallback,
                result: Some(Err(message)),
            },
        );
    }

    fn on_query_reply(&mut self, reply: QueryReply) -> Action {
        self.flow.query_running = false;
        let (body, title, is_error) = match reply.result {
            None => return Action::None,
            Some(Ok((text, label))) => (text, label, false),
            Some(Err(error)) => {
                error!(target: "ltn.app", "Quick action failed: {error}");
                (format!("Error: {error}"), reply.label, true)
            }
        };
        self.status.activity.clear();
        Action::Push(Box::new(MarkdownModal::query(&title, &body, is_error)))
    }

    fn run_label_task(&mut self, ctx: &mut Ctx) {
        if self.flow.label_running
            || !self.controller.config().output.auto_label
            || !self.label.is_empty()
        {
            return;
        }
        let Some(job) = self.controller.label_job() else {
            return;
        };
        self.flow.label_running = true;
        ctx.spawn_or(async move { LabelReply(job.await) }, |_| LabelReply(None));
    }

    fn on_label_reply(&mut self, reply: LabelReply, mode: &mut dyn Mode, ctx: &mut Ctx) -> Action {
        if let Some(label) = reply.0
            && self.label.is_empty()
        {
            self.apply_label(&label, mode, ctx);
        }
        self.flow.label_running = false;
        self.continue_pending_quit(mode, ctx)
    }

    /// Rename the folder to `<timestamp>_<label>` and show the label.
    pub fn apply_label(&mut self, label: &str, mode: &mut dyn Mode, ctx: &mut Ctx) {
        let safe = fit_name(&safe_label(label.trim()));
        if safe.is_empty() {
            return;
        }
        let current = self
            .output_dir
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default();
        // The timestamp is always the first 17 characters: "2026-02-21_143052".
        let prefix: String = current.chars().take(17).collect();
        let new_name = format!("{prefix}_{safe}");
        let new_path = self
            .output_dir
            .parent()
            .map_or_else(|| PathBuf::from(&new_name), |parent| parent.join(&new_name));
        if new_path != self.output_dir {
            if new_path.exists() {
                ctx.error(
                    format!("Session folder {new_name} already exists"),
                    Duration::from_secs(8),
                );
                return;
            }
            if let Err(e) = mode.move_folder(&self.output_dir, &new_path) {
                let text = OsError::from_io(&e, Some(&self.output_dir), Some(&new_path));
                error!(target: "ltn.app", "Could not rename {}: {text}", self.output_dir.display());
                ctx.error(
                    format!("Could not rename the session folder: {text}"),
                    Duration::from_secs(8),
                );
                return;
            }
            self.output_dir = new_path.clone();
            self.controller.relocate(new_path);
        }
        self.label = safe.clone();
        ctx.notify_with(
            format!("Session: {safe}"),
            crate::tui::host::Severity::Information,
            Duration::from_secs(3),
            "",
        );
    }

    /// The label modal or another screen this one pushed closed.
    pub fn resumed(&mut self, result: Option<Payload>, mode: &mut dyn Mode, ctx: &mut Ctx) {
        if !std::mem::take(&mut self.awaiting_label) {
            return;
        }
        if let Some(label) = result.and_then(|payload| payload.downcast::<String>().ok()) {
            self.apply_label(&label, mode, ctx);
        }
    }

    // --- Quit ---

    /// After q: wait for the worker, finish the notes once, let the label land, then close.
    pub fn continue_pending_quit(&mut self, mode: &dyn Mode, ctx: &mut Ctx) -> Action {
        match self
            .flow
            .quit_step(self.needs_final_digest(), mode.worker_busy())
        {
            QuitStep::Wait => Action::None,
            QuitStep::RunFinal => {
                self.run_final_digest(ctx);
                Action::None
            }
            QuitStep::Close => self.close_session(),
        }
    }

    /// Leave the session screen once, closing any modal above it first.
    pub fn close_session(&mut self) -> Action {
        if self.flow.session_closed {
            return Action::None;
        }
        self.flow.session_closed = true;
        self.flow.pending_quit = false;
        Action::Pop(None)
    }

    /// A task result for the session. Gives the message back when it is not one.
    pub fn task_message(
        &mut self,
        payload: Payload,
        mode: &mut dyn Mode,
        ctx: &mut Ctx,
    ) -> Result<Action, Payload> {
        let payload = match payload.downcast::<DigestReply>() {
            Ok(reply) => return Ok(self.on_digest_reply(*reply, mode, ctx)),
            Err(payload) => payload,
        };
        let payload = match payload.downcast::<DigestFailed>() {
            Ok(failed) => return Ok(self.on_digest_failed(*failed, mode, ctx)),
            Err(payload) => payload,
        };
        let payload = match payload.downcast::<QueryReply>() {
            Ok(reply) => return Ok(self.on_query_reply(*reply)),
            Err(payload) => payload,
        };
        match payload.downcast::<LabelReply>() {
            Ok(reply) => Ok(self.on_label_reply(*reply, mode, ctx)),
            Err(payload) => Err(payload),
        }
    }

    /// Abort the running tasks when the screen goes away.
    pub fn abort_tasks(&mut self) {
        if let Some(flight) = self.in_flight.take() {
            flight.task.abort();
        }
    }
}

/// One chip per speaker: `padding: 0 1` on the bar, and each chip `padding: 0 1`,
/// `margin-right: 1` on `$surface`. Names are the user's text, drawn as typed.
fn render_speaker_bar(
    frame: &mut Frame,
    area: Rect,
    chips: &[(u32, String, f64)],
    theme: &crate::tui::theme::Theme,
) {
    let chip = Style::default()
        .bg(theme.color("surface"))
        .fg(theme.on("foreground", "surface"));
    let mut spans = vec![Span::raw(" ")];
    for (speaker, name, seconds) in chips {
        let shown = match name.is_empty() {
            true => format!("S{speaker}"),
            false => name.clone(),
        };
        spans.push(Span::styled(
            format!(" {shown} {} ", format_spoken_time(*seconds)),
            chip,
        ));
        spans.push(Span::raw(" "));
    }
    let base = Style::default()
        .bg(theme.color("background"))
        .fg(theme.color("foreground"));
    frame.render_widget(
        ratatui::widgets::Paragraph::new(Line::from(spans)).style(base),
        area,
    );
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

    #[test]
    fn a_long_label_is_cut_on_a_character_boundary() {
        let label = "會".repeat(100);
        let fitted = fit_name(&label);
        assert_eq!(fitted.len(), 198);
        assert!(fitted.chars().all(|c| c == '會'));
    }

    #[test]
    fn a_pending_quit_waits_then_runs_the_final_digest_once_then_closes() {
        let mut flow = Flow::default();
        assert_eq!(flow.quit_step(true, false), QuitStep::Wait, "no quit asked");
        flow.pending_quit = true;
        assert_eq!(
            flow.quit_step(true, true),
            QuitStep::Wait,
            "the worker still sends lines"
        );
        flow.digest_running = true;
        assert_eq!(flow.quit_step(true, false), QuitStep::Wait);
        flow.digest_running = false;
        assert_eq!(flow.quit_step(true, false), QuitStep::RunFinal);
        flow.final_attempted = true;
        assert_eq!(
            flow.quit_step(true, false),
            QuitStep::Close,
            "a failed final is not retried"
        );
        flow.label_running = true;
        assert_eq!(
            flow.quit_step(false, false),
            QuitStep::Wait,
            "the AI label may still land"
        );
    }
}
