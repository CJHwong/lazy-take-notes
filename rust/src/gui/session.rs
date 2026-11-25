//! A recording or a file transcription, as the TUI's record and transcribe
//! screens: the transcript and the speaker chips on the left, the digest and
//! the session context on the right, the status line and the keys below. The
//! session itself is `app::session`; this view forwards input and draws it.

use std::rc::Rc;
use std::time::{Duration, Instant};

use gpui_kit::component::input::{InputEvent, Textarea, TextareaState};
use gpui_kit::component::text::{TextView, TextViewState};
use gpui_kit::prelude::FluentBuilder as _;
use gpui_kit::*;
use tracing::{error, info};

use crate::app::events::Event;
use crate::app::notices::CONSENT_NOTICE_BODY;
use crate::app::session::file::FileRuntime;
use crate::app::session::record::RecordRuntime;
use crate::app::session::status_line::StatusLine;
use crate::app::session::texts;
use crate::app::session::{SHUTDOWN_GRACE, SessionIo, SessionRuntime, Severity, Step};
use crate::app::session_start::{FileStart, RecordStart};
use crate::app::tasks::Payload;
use crate::domain::transcript::format_wall_time;
use crate::gui::bridge::{Bridge, GuiIo, Incoming, notify};
use crate::gui::dialogs::{self, Extra};
use crate::gui::theme::{color, on, tinted};
use crate::gui::widgets::{Hint, footer, header, hint, panel};

/// The keys a quick action takes, by position.
const QUICK_KEYS: [&str; 9] = ["1", "2", "3", "4", "5", "6", "7", "8", "9"];
const COPY_TIMEOUT: Duration = Duration::from_secs(2);
const ERROR_TIMEOUT: Duration = Duration::from_secs(8);
/// How often the status line redraws its timer and level meter.
const TICK: Duration = Duration::from_millis(250);

pub enum SessionEvent {
    /// Back to the menu.
    Closed,
    /// The window closes: the app ends.
    Exit,
}

enum Mode {
    Record(Box<RecordRuntime>),
    File(Box<FileRuntime>),
}

impl Mode {
    fn session(&self) -> &SessionRuntime {
        match self {
            Self::Record(record) => &record.session,
            Self::File(file) => &file.session,
        }
    }

    fn session_mut(&mut self) -> &mut SessionRuntime {
        match self {
            Self::Record(record) => &mut record.session,
            Self::File(file) => &mut file.session,
        }
    }
}

/// Which panel the keys go to, in the TUI's Tab order.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Focus {
    Transcript,
    Speakers,
    Digest,
    Context,
}

/// One drawn transcript line.
struct Line {
    time: SharedString,
    text: SharedString,
    /// The text without its time, for the copy and the speaker highlight.
    plain: String,
}

pub struct SessionView {
    mode: Mode,
    bridge: Bridge,
    status: StatusLine,
    focus: Focus,
    focus_handle: FocusHandle,
    lines: Vec<Line>,
    transcript: ListState,
    digest: Entity<TextViewState>,
    context: Entity<TextareaState>,
    /// The speaker chip clicked once: a second click renames it.
    selected_speaker: Option<u32>,
    /// The window close waits for the audio flush.
    closing: bool,
    shown_lines: usize,
    shown_relabel: u64,
    shown_digest: u64,
    _pump: Task<()>,
    _ticker: Task<()>,
    _subscriptions: Vec<Subscription>,
}

impl EventEmitter<SessionEvent> for SessionView {}

impl SessionView {
    pub fn record(start: RecordStart, window: &mut Window, cx: &mut Context<Self>) -> Self {
        let RecordStart {
            session,
            workers,
            silence_threshold,
        } = start;
        let mode = Mode::Record(Box::new(RecordRuntime::new(session, workers)));
        let mut view = Self::new(mode, "Record", "", silence_threshold, window, cx);
        view.mount(window, cx);
        let noticed = match &view.mode {
            Mode::Record(record) => record.consent_noticed.clone(),
            Mode::File(_) => return view,
        };
        if !noticed.exists() {
            cx.defer_in(window, move |_, window, cx| {
                consent_notice(noticed, window, cx);
            });
        }
        view
    }

    pub fn file(start: FileStart, window: &mut Window, cx: &mut Context<Self>) -> Self {
        let FileStart {
            session,
            launcher,
            silence_threshold,
        } = start;
        let mode = Mode::File(Box::new(FileRuntime::new(session, launcher)));
        let mut view = Self::new(
            mode,
            "Transcribe",
            "● Transcribing",
            silence_threshold,
            window,
            cx,
        );
        view.mount(window, cx);
        view
    }

    fn new(
        mode: Mode,
        mode_label: &str,
        active_label: &str,
        silence_threshold: f64,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Self {
        let mut bridge = Bridge::new();
        let pump = bridge.pump(window, cx, Self::incoming);
        let ticker = cx.spawn(async move |this, cx| {
            loop {
                cx.background_executor().timer(TICK).await;
                if this.update(cx, |_, cx| cx.notify()).is_err() {
                    break;
                }
            }
        });
        let digest = cx.new(|cx| TextViewState::markdown("", cx));
        let context = cx.new(|cx| {
            TextareaState::new(window, cx).placeholder("Notes for the AI: names, topic, goals...")
        });
        let subscriptions = vec![cx.subscribe_in(&context, window, Self::on_context)];
        let transcript = ListState::new(0, ListAlignment::Top, px(400.));
        transcript.set_follow_mode(FollowMode::Tail);
        let mut status = StatusLine::default();
        status.mode_label = mode_label.into();
        status.silence_threshold = silence_threshold;
        if !active_label.is_empty() {
            status.active_label = active_label.into();
        }
        let focus_handle = cx.focus_handle();
        // The context box may take the focus as it mounts; the keys start on the view.
        cx.defer_in(window, |view, window, cx| {
            view.focus_handle.focus(window, cx)
        });
        Self {
            mode,
            bridge,
            status,
            focus: Focus::Transcript,
            focus_handle,
            lines: Vec::new(),
            transcript,
            digest,
            context,
            selected_speaker: None,
            closing: false,
            shown_lines: 0,
            shown_relabel: 0,
            shown_digest: 0,
            _pump: pump,
            _ticker: ticker,
            _subscriptions: subscriptions,
        }
    }

    /// Run `call` with the session's notices and tasks, then act on what it asked for.
    fn with_io<R>(
        &mut self,
        window: &mut Window,
        cx: &mut Context<Self>,
        call: impl FnOnce(&mut Mode, &mut dyn SessionIo) -> R,
    ) -> R {
        let tasks = self.bridge.tasks.clone();
        let mut io = GuiIo::new(window, cx, &tasks);
        let result = call(&mut self.mode, &mut io);
        if io.exited {
            cx.emit(SessionEvent::Exit);
        }
        result
    }

    fn mount(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.with_io(window, cx, |mode, io| match mode {
            Mode::Record(record) => record.mount(io),
            Mode::File(file) => file.mount(io),
        });
        let session = self.mode.session();
        self.status.quick_action_hints = session
            .controller
            .template()
            .quick_actions
            .iter()
            .enumerate()
            .map(|(index, action)| format!("[{}] {}", index + 1, action.label))
            .collect::<Vec<_>>()
            .join("  ");
        self.status.buf_max = session.controller.config().digest.min_lines;
        self.status.ai_notes = session.missing_digest.is_empty();
        if !session.missing_digest.is_empty() {
            let notice = texts::missing_models_notice(&session.missing_digest);
            self.digest
                .update(cx, |digest, cx| digest.set_text(&notice, cx));
        }
        self.sync(window, cx);
    }

    /// A worker event or a task result from the bridge.
    fn incoming(&mut self, incoming: Incoming, window: &mut Window, cx: &mut Context<Self>) {
        let step = match incoming {
            Incoming::Worker(event) => self.on_event(event, window, cx),
            Incoming::Task(payload) => self.on_task(payload, window, cx),
        };
        self.show(step, window, cx);
    }

    fn on_event(&mut self, event: Event, window: &mut Window, cx: &mut Context<Self>) -> Step {
        self.with_io(window, cx, |mode, io| match mode {
            Mode::Record(record) => record.on_event(event, io),
            Mode::File(file) => file.on_event(event, io).unwrap_or(Step::None),
        })
    }

    fn on_task(&mut self, payload: Payload, window: &mut Window, cx: &mut Context<Self>) -> Step {
        let result = self.with_io(window, cx, |mode, io| match mode {
            Mode::Record(record) => record.task_result(payload, io),
            Mode::File(file) => file.task_result(payload, io),
        });
        result.unwrap_or_else(|_| {
            error!(target: "ltn.app", "a task result nobody takes");
            Step::None
        })
    }

    /// Bring the panels and the status line up to the session's state.
    fn sync(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let session = self.mode.session_mut();
        self.status.sync_from(&mut session.status);
        if session.status.context_locked {
            self.context
                .update(cx, |context, cx| context.set_readonly(true, cx));
        }
        if session.status.digest_version != self.shown_digest {
            self.shown_digest = session.status.digest_version;
            let digest = session.status.digest.clone();
            self.digest
                .update(cx, |view, cx| view.set_text(&digest, cx));
        }
        self.sync_lines();
        let _ = window;
    }

    /// New lines go at the end; a speaker pass rewrites every line with its label.
    fn sync_lines(&mut self) {
        let session = self.mode.session();
        let controller = &session.controller;
        let segments = &controller.all_segments;
        let relabeled = session.status.relabel_version != self.shown_relabel;
        let grown = segments.len() > self.shown_lines;
        let labeled = !controller.speaker_turns.is_empty();
        let from = match (relabeled || (grown && labeled), grown) {
            (true, _) => 0,
            (false, true) => self.shown_lines,
            (false, false) => return,
        };
        self.lines.truncate(from);
        for segment in &segments[from..] {
            let label = labeled.then(|| controller.speaker_label(segment)).flatten();
            let plain = texts::spoken_text(segment, label.as_deref());
            self.lines.push(Line {
                time: format!("[{}]", format_wall_time(segment.wall_start)).into(),
                text: plain.clone().into(),
                plain,
            });
        }
        let old = self.transcript.item_count();
        if from == 0 {
            self.transcript.reset(self.lines.len());
        } else {
            self.transcript.splice(old..old, self.lines.len() - old);
        }
        self.shown_relabel = session.status.relabel_version;
        self.shown_lines = segments.len();
    }

    /// Show what the session asked for, after the view caught up with it.
    fn show(&mut self, step: Step, window: &mut Window, cx: &mut Context<Self>) {
        self.sync(window, cx);
        match step {
            Step::None => {}
            Step::ShowQuery {
                title,
                body,
                is_error,
            } => {
                let accent = if is_error { "error" } else { "primary" };
                dialogs::markdown(window, cx, title, &body, accent, None, None);
            }
            Step::ShowDownload { model } => self.show_download(model, window, cx),
            Step::Close => cx.emit(SessionEvent::Closed),
        }
        cx.notify();
    }

    fn show_download(&mut self, model: String, window: &mut Window, cx: &mut Context<Self>) {
        let body = format!(
            "Downloading **{model}** for the first time. It is cached for later sessions.\n\nThe progress shows in the status line."
        );
        let view = cx.entity().downgrade();
        let on_close: dialogs::Callback = Rc::new(move |_, cx| {
            let _ = view.update(cx, |view, _| view.mode.session_mut().download_closed());
        });
        dialogs::markdown(
            window,
            cx,
            "Downloading the speech model",
            &body,
            "primary",
            None,
            Some(on_close),
        );
    }

    fn on_context(
        &mut self,
        context: &Entity<TextareaState>,
        event: &InputEvent,
        _: &mut Window,
        cx: &mut Context<Self>,
    ) {
        match event {
            InputEvent::Change => {
                let text = context.read(cx).value().to_string();
                self.mode.session_mut().set_user_context(text);
            }
            InputEvent::Focus => self.focus = Focus::Context,
            InputEvent::Blur if self.focus == Focus::Context => self.focus = Focus::Transcript,
            InputEvent::Blur | InputEvent::PressEnter { .. } => {}
        }
        cx.notify();
    }

    fn state(&self) -> &'static str {
        self.mode.session().status.state
    }

    fn hints(&self) -> &'static str {
        match &self.mode {
            Mode::Record(_) => texts::record_hints(self.state()),
            Mode::File(_) => texts::file_hints(self.state()),
        }
    }

    fn speakers(&self) -> Vec<(u32, String, f64)> {
        self.mode.session().speaker_chips()
    }

    fn set_focus(&mut self, focus: Focus, window: &mut Window, cx: &mut Context<Self>) {
        self.focus = focus;
        if focus == Focus::Context {
            let handle = self.context.read(cx).focus_handle(cx);
            handle.focus(window, cx);
        } else {
            self.focus_handle.focus(window, cx);
        }
        if focus == Focus::Speakers && self.selected_speaker.is_none() {
            self.selected_speaker = self.speakers().first().map(|chip| chip.0);
        }
        cx.notify();
    }

    fn move_focus(&mut self, step: isize, window: &mut Window, cx: &mut Context<Self>) {
        const ORDER: [Focus; 4] = [
            Focus::Transcript,
            Focus::Speakers,
            Focus::Digest,
            Focus::Context,
        ];
        let has_speakers = !self.speakers().is_empty();
        let mut at = ORDER.iter().position(|f| *f == self.focus).unwrap_or(0) as isize;
        loop {
            at = (at + step).rem_euclid(ORDER.len() as isize);
            if ORDER[at as usize] != Focus::Speakers || has_speakers {
                break;
            }
        }
        self.set_focus(ORDER[at as usize], window, cx);
    }

    /// A key, or a click on its hint. `name` is the TUI's key name.
    fn press(&mut self, name: &str, window: &mut Window, cx: &mut Context<Self>) {
        if self.focus == Focus::Speakers && self.speaker_key(name, window, cx) {
            return;
        }
        let step = match name {
            "q" => self.with_io(window, cx, |mode, io| match mode {
                Mode::Record(record) => record.quit(io),
                Mode::File(file) => file.quit(io),
            }),
            "space" => self.with_io(window, cx, |mode, io| {
                if let Mode::Record(record) = mode {
                    record.toggle_pause(io);
                }
                Step::None
            }),
            "s" => self.with_io(window, cx, |mode, io| {
                match mode {
                    Mode::Record(record) => record.stop_recording(io),
                    Mode::File(file) => file.stop_transcription(io),
                }
                Step::None
            }),
            "d" => self.with_io(window, cx, |mode, io| {
                match mode {
                    Mode::Record(record) => record.force_digest(io),
                    Mode::File(file) => file.force_digest(io),
                }
                Step::None
            }),
            "m" => self.with_io(window, cx, |mode, io| {
                if let Mode::Record(record) = mode {
                    record.toggle_mic(io);
                }
                Step::None
            }),
            "c" => {
                self.copy(window, cx);
                Step::None
            }
            "l" => {
                self.ask_label(window, cx);
                Step::None
            }
            "o" if self.status.stopped => {
                cx.open_with_system(&self.mode.session().output_dir);
                Step::None
            }
            "h" => {
                self.show_help(window, cx);
                Step::None
            }
            "tab" => {
                self.move_focus(1, window, cx);
                Step::None
            }
            "shift-tab" => {
                self.move_focus(-1, window, cx);
                Step::None
            }
            "ctrl-q" => {
                if !self.should_close(window, cx) {
                    return;
                }
                cx.emit(SessionEvent::Exit);
                Step::None
            }
            _ => self.quick_action(name, window, cx),
        };
        self.show(step, window, cx);
    }

    fn quick_action(&mut self, name: &str, window: &mut Window, cx: &mut Context<Self>) -> Step {
        let count = self.mode.session().quick_action_count();
        match name.parse::<usize>() {
            Ok(index) if (1..=count).contains(&index) => {
                info!(target: "ltn.app", "quick action {index} started by key {name:?}");
                self.with_io(window, cx, |mode, io| {
                    mode.session_mut().quick_action(name, io)
                });
            }
            _ => {}
        }
        Step::None
    }

    /// Left and right choose a speaker; Enter names it. True when the key was one of them.
    fn speaker_key(&mut self, name: &str, window: &mut Window, cx: &mut Context<Self>) -> bool {
        let speakers: Vec<u32> = self.speakers().iter().map(|chip| chip.0).collect();
        let at = self
            .selected_speaker
            .and_then(|speaker| speakers.iter().position(|s| *s == speaker));
        let moved = match (name, at) {
            ("left", Some(at)) => at.checked_sub(1),
            ("right", Some(at)) => (at + 1 < speakers.len()).then_some(at + 1),
            ("enter", Some(at)) => {
                self.ask_speaker_name(speakers[at], window, cx);
                return true;
            }
            _ => return false,
        };
        if let Some(at) = moved {
            self.selected_speaker = Some(speakers[at]);
            cx.notify();
        }
        true
    }

    fn click_speaker(&mut self, speaker: u32, window: &mut Window, cx: &mut Context<Self>) {
        self.set_focus(Focus::Speakers, window, cx);
        if self.selected_speaker != Some(speaker) {
            self.selected_speaker = Some(speaker);
            return;
        }
        self.ask_speaker_name(speaker, window, cx);
    }

    fn ask_speaker_name(&mut self, speaker: u32, window: &mut Window, cx: &mut Context<Self>) {
        let current = self
            .mode
            .session()
            .controller
            .speaker_names
            .get(&speaker)
            .cloned()
            .unwrap_or_default();
        let view = cx.entity().downgrade();
        dialogs::prompt(
            window,
            cx,
            format!("Name for S{speaker}"),
            &current,
            "Speaker name...",
            move |name, window, cx| {
                let _ = view.update(cx, |view, cx| {
                    view.with_io(window, cx, |mode, io| {
                        mode.session_mut().set_speaker_name(speaker, name, io)
                    });
                    view.show(Step::None, window, cx);
                });
            },
        );
    }

    fn ask_label(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let current = self.mode.session().label.clone();
        let view = cx.entity().downgrade();
        dialogs::prompt(
            window,
            cx,
            "Session label",
            &current,
            "Session label...",
            move |label, window, cx| {
                let _ = view.update(cx, |view, cx| {
                    view.with_io(window, cx, |mode, io| match mode {
                        Mode::Record(record) => record.apply_label(&label, io),
                        Mode::File(file) => file.apply_label(&label, io),
                    });
                    view.show(Step::None, window, cx);
                });
            },
        );
    }

    fn show_help(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let rows: &[&str] = match &self.mode {
            Mode::Record(_) => &texts::RECORD_HELP_ROWS,
            Mode::File(_) => &texts::FILE_HELP_ROWS,
        };
        let body = texts::help_markdown(self.mode.session(), rows);
        dialogs::markdown(window, cx, "Help", &body, "primary", None, None);
    }

    /// The session context, once it is locked: the copies end with it.
    fn copy_suffix(&self, cx: &App) -> String {
        match self.mode.session().status.context_locked {
            true => self.context.read(cx).value().trim().to_string(),
            false => String::new(),
        }
    }

    /// `c`: the digest when it has the focus, otherwise the transcript.
    fn copy(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let context = self.copy_suffix(cx);
        let (mut text, what, empty) = if self.focus == Focus::Digest {
            let digest = self.mode.session().status.digest.clone();
            (digest, "Digest copied", "No digest to copy")
        } else {
            let lines: Vec<String> = self
                .lines
                .iter()
                .map(|line| format!("{} {}", line.time, line.plain))
                .collect();
            (
                lines.join("\n"),
                "Transcript copied",
                "No transcript to copy",
            )
        };
        if text.is_empty() {
            notify(window, cx, empty.into(), Severity::Warning, COPY_TIMEOUT);
            return;
        }
        if !context.is_empty() {
            text.push_str(&match self.focus {
                Focus::Digest => format!("\n\n---\n\n**Session Context**\n\n{context}"),
                _ => format!("\n\n---\nSession Context:\n{context}"),
            });
        }
        cx.write_to_clipboard(ClipboardItem::new_string(text));
        notify(window, cx, what.into(), Severity::Information, COPY_TIMEOUT);
    }

    /// The window may close now. Otherwise the recording stops, and the app
    /// ends once the audio is flushed, or after the grace time.
    pub fn should_close(&mut self, _: &mut Window, cx: &mut Context<Self>) -> bool {
        if self.closing {
            return false;
        }
        let flushing = match &mut self.mode {
            Mode::Record(record) => record.stop_for_shutdown(),
            Mode::File(file) => file.stop_for_shutdown(),
        };
        if !flushing {
            return true;
        }
        self.closing = true;
        cx.spawn(async move |this, cx| {
            cx.background_executor().timer(SHUTDOWN_GRACE).await;
            let _ = this.update(cx, |_, cx| cx.emit(SessionEvent::Exit));
        })
        .detach();
        false
    }

    fn on_key(&mut self, event: &KeyDownEvent, window: &mut Window, cx: &mut Context<Self>) {
        let keystroke = &event.keystroke;
        let name = match (
            keystroke.modifiers.control,
            keystroke.modifiers.shift,
            keystroke.key.as_str(),
        ) {
            (true, _, "q") => "ctrl-q".to_string(),
            (_, true, "tab") => "shift-tab".to_string(),
            (false, _, key) if !keystroke.modifiers.platform && !keystroke.modifiers.alt => {
                key.to_string()
            }
            _ => return,
        };
        // A locked context box takes no text, so the keys act as anywhere else.
        let locked = self.mode.session().status.context_locked;
        let context_focused = self.context.read(cx).focus_handle(cx).is_focused(window);
        if context_focused && !locked {
            // The context box takes every key but Escape and Tab.
            match name.as_str() {
                "escape" => self.set_focus(Focus::Transcript, window, cx),
                "ctrl-q" => self.press(&name, window, cx),
                _ => return,
            }
            cx.stop_propagation();
            return;
        }
        cx.stop_propagation();
        self.press(&name, window, cx);
    }

    fn render_transcript(&self, cx: &mut Context<Self>) -> Div {
        let highlight = self.selected_speaker.map(|speaker| {
            let session = self.mode.session();
            let name = session
                .controller
                .speaker_names
                .get(&speaker)
                .filter(|name| !name.is_empty())
                .cloned()
                .unwrap_or_else(|| format!("S{speaker}"));
            format!("{name}: ")
        });
        let marked = tinted(cx, "background", "primary", 0.35);
        let muted = color(cx, "text-muted");
        let list = list(
            self.transcript.clone(),
            cx.processor(move |view: &mut Self, index: usize, _, _| {
                let Some(line) = view.lines.get(index) else {
                    return div().into_any_element();
                };
                let mine = highlight
                    .as_deref()
                    .is_some_and(|label| line.plain.starts_with(label));
                div()
                    .w_full()
                    .flex()
                    .gap_2()
                    .px_1()
                    .py(px(1.))
                    .when(mine, |row| row.bg(marked))
                    .child(div().flex_none().text_color(muted).child(line.time.clone()))
                    .child(div().flex_1().min_w_0().child(line.text.clone()))
                    .into_any_element()
            }),
        )
        .size_full();
        panel("Transcript", self.focus == Focus::Transcript, cx)
            .flex_1()
            .child(
                div()
                    .id("transcript")
                    .flex_1()
                    .min_h_0()
                    .on_click(cx.listener(|view, _, window, cx| {
                        view.selected_speaker = None;
                        view.set_focus(Focus::Transcript, window, cx);
                    }))
                    .child(list),
            )
    }

    fn render_speakers(&self, cx: &mut Context<Self>) -> Option<Div> {
        let chips = self.speakers();
        if chips.is_empty() {
            return None;
        }
        let focused = self.focus == Focus::Speakers;
        let items = chips.into_iter().map(|(speaker, name, seconds)| {
            let selected = self.selected_speaker == Some(speaker);
            let (background, text) = if selected && focused {
                (
                    color(cx, "block-cursor-background"),
                    on(cx, "block-cursor-foreground", "block-cursor-background"),
                )
            } else if selected {
                (
                    color(cx, "block-cursor-blurred-background"),
                    color(cx, "foreground"),
                )
            } else {
                (color(cx, "surface"), on(cx, "foreground", "surface"))
            };
            let shown = if name.is_empty() {
                format!("S{speaker}")
            } else {
                name
            };
            div()
                .id(("speaker", speaker as usize))
                .px_2()
                .py(px(2.))
                .rounded(px(10.))
                .bg(background)
                .text_color(text)
                .text_sm()
                .cursor_pointer()
                .child(format!(
                    "{shown}  {}",
                    crate::app::session::format_spoken_time(seconds)
                ))
                .on_click(
                    cx.listener(move |view, _, window, cx| view.click_speaker(speaker, window, cx)),
                )
        });
        Some(div().flex().flex_wrap().gap_2().pt_2().children(items))
    }

    fn render_status(&self, cx: &App) -> Div {
        let separator = || div().text_color(color(cx, "text-muted")).child("│");
        let mut parts = Vec::new();
        for (index, (text, alert)) in self.status.parts(Instant::now()).into_iter().enumerate() {
            if index > 0 {
                parts.push(separator());
            }
            let part = div().child(text);
            parts.push(if alert {
                part.px_1()
                    .bg(color(cx, "error"))
                    .text_color(on(cx, "text", "error"))
                    .font_weight(FontWeight::BOLD)
            } else {
                part
            });
        }
        div()
            .flex()
            .flex_none()
            .items_center()
            .gap_2()
            .h(px(26.))
            .px_3()
            .bg(color(cx, "surface"))
            .text_color(on(cx, "text", "surface"))
            .text_sm()
            .font_family("Menlo")
            .children(parts)
    }

    /// The key hints for this state, then the quick actions.
    fn footer_hints(&self) -> Vec<Hint> {
        let mut hints = parse_hints(self.hints());
        let actions = &self.mode.session().controller.template().quick_actions;
        for (index, action) in actions.iter().enumerate().take(QUICK_KEYS.len()) {
            let key = QUICK_KEYS[index];
            hints.push(hint(key, action.label.clone(), key));
        }
        hints
    }
}

/// `[Space] pause  [s] stop` as footer hints. The key name is the bracket
/// text in lower case, as the TUI names keys.
fn parse_hints(text: &'static str) -> Vec<Hint> {
    text.split("  ")
        .filter_map(|item| {
            let rest = item.trim().strip_prefix('[')?;
            let (key, label) = rest.split_once("] ")?;
            let name = match key {
                "Space" => "space",
                "Tab" => "tab",
                key => key,
            };
            Some(hint(key, label.to_string(), name))
        })
        .collect()
}

/// The recording notice, until the user stops it for good.
fn consent_notice(noticed: std::path::PathBuf, window: &mut Window, cx: &mut App) {
    let extra = Extra {
        label: "Don't show again",
        run: Rc::new(move |window, cx| {
            let written = noticed
                .parent()
                .map_or(Ok(()), std::fs::create_dir_all)
                .and_then(|()| std::fs::write(&noticed, ""));
            if let Err(e) = written {
                notify(
                    window,
                    cx,
                    format!("Cannot save the setting: {e}"),
                    Severity::Error,
                    ERROR_TIMEOUT,
                );
            }
        }),
    };
    dialogs::markdown(
        window,
        cx,
        "Recording Notice",
        CONSENT_NOTICE_BODY,
        "warning",
        Some(extra),
        None,
    );
}

impl Focusable for SessionView {
    fn focus_handle(&self, _: &App) -> FocusHandle {
        self.focus_handle.clone()
    }
}

impl Render for SessionView {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let session = self.mode.session();
        let metadata = &session.controller.template().metadata;
        let title = texts::header_text(
            &metadata.name,
            &metadata.description,
            &metadata.locale,
            &session.label,
        );
        let context_focused = self.focus == Focus::Context;
        let transcript = self.render_transcript(cx);
        let speakers = self.render_speakers(cx);
        let digest =
            panel("Digest", self.focus == Focus::Digest, cx)
                .flex_1()
                .child(
                    div()
                        .id("digest")
                        .flex_1()
                        .min_h_0()
                        .overflow_y_scroll()
                        .on_click(cx.listener(|view, _, window, cx| {
                            view.set_focus(Focus::Digest, window, cx)
                        }))
                        .child(TextView::new(&self.digest).selectable(true)),
                );
        let context = panel("Session Context", context_focused, cx)
            .when(context_focused, |panel| panel.h(relative(0.4)))
            .when(!context_focused, |panel| panel.h(px(130.)))
            .flex_none()
            .child(
                Textarea::new(&self.context)
                    .bordered(false)
                    .h_full()
                    .bg(color(cx, "background")),
            );
        let entity = cx.entity().downgrade();
        let on_press = Rc::new(
            move |name: &'static str, window: &mut Window, cx: &mut App| {
                let _ = entity.update(cx, |view, cx| view.press(name, window, cx));
            },
        );
        div()
            .key_context("Session")
            .track_focus(&self.focus_handle)
            .on_key_down(cx.listener(Self::on_key))
            .size_full()
            .flex()
            .flex_col()
            .child(header(title, cx))
            .child(
                div()
                    .flex_1()
                    .min_h_0()
                    .flex()
                    .gap_3()
                    .p_3()
                    .pt_4()
                    .child(
                        div()
                            .flex_1()
                            .min_w_0()
                            .flex()
                            .flex_col()
                            .child(transcript)
                            .children(speakers),
                    )
                    .child(
                        div()
                            .flex_1()
                            .min_w_0()
                            .flex()
                            .flex_col()
                            .gap_4()
                            .child(digest)
                            .child(context),
                    ),
            )
            .child(self.render_status(cx))
            .child(footer(self.footer_hints(), on_press, cx))
    }
}
