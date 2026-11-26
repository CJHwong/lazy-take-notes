//! A recording or a file transcription in the detail pane: the title and the
//! toolbar in the title bar row, the transcript and the speakers on the left,
//! the digest and the session context on the right, and a status strip below.
//! The commands are toolbar buttons and the Session menu. The session itself
//! is `app::session`; this view forwards input and draws it.

use std::rc::Rc;
use std::time::{Duration, Instant};

use gpui_kit::component::button::{Button, ButtonVariants as _};
use gpui_kit::component::input::{InputEvent, InputState, Textarea, TextareaState};
use gpui_kit::component::menu::DropdownMenu as _;
use gpui_kit::component::text::TextViewState;
use gpui_kit::component::{ActiveTheme, Disableable as _, Icon, IconName, Sizable as _};
use gpui_kit::prelude::FluentBuilder as _;
use gpui_kit::*;
use tracing::{error, info};

use crate::app::events::Event;
use crate::app::notices::CONSENT_NOTICE_BODY;
use crate::app::session::file::FileRuntime;
use crate::app::session::record::RecordRuntime;
use crate::app::session::status_line::{LEVEL_TOP, StatusLine};
use crate::app::session::texts;
use crate::app::session::{SHUTDOWN_GRACE, SessionIo, SessionRuntime, Severity, Step};
use crate::app::session_start::{FileStart, RecordStart};
use crate::app::tasks::Payload;
use crate::domain::transcript::format_wall_time;
use crate::gui::bridge::{Bridge, GuiIo, Incoming, notify};
use crate::gui::design::{control, radius, size, space, text, tint};
use crate::gui::dialogs::{self, Extra};
use crate::gui::icons;
use crate::gui::widgets::{
    DragArm, column_header, empty_state, inline_tool, markdown, pane_header, speaker_chip,
    status_strip, tag, title_field, tool, transcript_line,
};
use crate::gui::{
    CopyDigest, CopyTranscript, DigestNow, FinishSession, QuickAction1, QuickAction2, QuickAction3,
    QuickAction4, QuickAction5, RenameSession, SESSION_CONTEXT, ShowHelp, ShowInFinder,
    StopSession, ToggleMute, TogglePause, quick_action,
};

const COPY_TIMEOUT: Duration = Duration::from_secs(2);
const ERROR_TIMEOUT: Duration = Duration::from_secs(8);
/// How often the status strip redraws its timer and level meter.
const TICK: Duration = Duration::from_millis(250);

/// The window's keys, for the Help menu.
pub const SHORTCUTS: &str = "\
| Key | Action |
|-----|--------|
| `Space` | Pause or resume the recording |
| `⌘.` | Stop |
| `⌘↩` | Finish: write the final digest and save |
| `⌘D` | Digest now |
| `⇧⌘M` | Mute or unmute the microphone |
| `⌘1` to `⌘5` | The template's quick actions |
| `⇧⌘C` | Copy the transcript |
| `⌥⌘C` | Copy the digest |
| `⌘L` | Rename the session |
| `⇧⌘R` | Show the session folder in Finder |
| `Tab` | Move between the transcript, the speakers and the session context |
| `←` `→` `Return` | On the speakers: choose one, name it |
| `⌘N` | New session: choose a template, then record or transcribe |
| `⌘,` | Settings |
| `⌘W` `⌘Q` | Close the window; a recording stops and saves first |";

pub enum SessionEvent {
    /// The session ended: show what it saved.
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

/// Where Tab is: the keys of the speakers work only on them.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Focus {
    Transcript,
    Speakers,
    Context,
}

/// What a copy takes.
#[derive(Clone, Copy)]
enum Copied {
    Transcript,
    Digest,
}

/// One drawn transcript line.
struct Line {
    time: SharedString,
    /// The speaker's label, once the speakers are known.
    speaker: Option<SharedString>,
    /// The words alone, without the speaker.
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
    /// The session label, edited in place in the title bar row.
    title: Entity<InputState>,
    shown_label: String,
    drag: DragArm,
    /// The speaker chosen with the keys; its lines are marked.
    selected_speaker: Option<u32>,
    /// The speaker chip under the pointer; its lines are marked.
    hovered_speaker: Option<u32>,
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
        let mut view = Self::new(mode, silence_threshold, window, cx);
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
        let mut view = Self::new(mode, silence_threshold, window, cx);
        view.mount(window, cx);
        view
    }

    fn new(
        mode: Mode,
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
        let label = mode.session().label.clone();
        let title = cx.new(|cx| {
            InputState::new(window, cx)
                .placeholder("Untitled")
                .default_value(shown_title(&label))
        });
        let subscriptions = vec![
            cx.subscribe_in(&context, window, Self::on_context),
            cx.subscribe_in(&title, window, Self::on_title),
        ];
        let transcript = ListState::new(0, ListAlignment::Top, size::LIST_OVERDRAW);
        transcript.set_follow_mode(FollowMode::Tail);
        let mut status = StatusLine::default();
        status.silence_threshold = silence_threshold;
        let focus_handle = cx.focus_handle();
        // An input may take the focus as it mounts; the keys start on the view.
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
            title,
            shown_label: label,
            drag: DragArm::default(),
            selected_speaker: None,
            hovered_speaker: None,
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

    /// A command: run it on the session, then show what it asked for.
    fn run(
        &mut self,
        window: &mut Window,
        cx: &mut Context<Self>,
        call: impl FnOnce(&mut Mode, &mut dyn SessionIo) -> Step,
    ) {
        let step = self.with_io(window, cx, call);
        self.show(step, window, cx);
    }

    fn mount(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.with_io(window, cx, |mode, io| match mode {
            Mode::Record(record) => record.mount(io),
            Mode::File(file) => file.mount(io),
        });
        let session = self.mode.session();
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
        result.unwrap_or_else(|payload| {
            if !is_thread_end(&payload) {
                error!(target: "ltn.app", "a task result nobody takes");
            }
            Step::None
        })
    }

    /// Bring the panes, the title and the status strip up to the session's state.
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
        self.sync_title(window, cx);
        self.sync_lines();
    }

    /// The AI label, or a rename that changed the text: the title shows it,
    /// unless the user is typing there.
    fn sync_title(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let label = self.mode.session().label.clone();
        if label == self.shown_label || self.title_focused(window, cx) {
            return;
        }
        self.title.update(cx, |title, cx| {
            title.set_value(shown_title(&label), window, cx)
        });
        self.shown_label = label;
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
                time: format_wall_time(segment.wall_start).into(),
                speaker: label.map(Into::into),
                text: segment.text.clone().into(),
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
            "Downloading **{model}** for the first time. It is cached for later sessions.\n\nThe progress shows in the status strip."
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

    /// Return keeps the new title; leaving the field keeps it too.
    fn on_title(
        &mut self,
        _: &Entity<InputState>,
        event: &InputEvent,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        match event {
            InputEvent::PressEnter { .. } => self.focus_handle.focus(window, cx),
            InputEvent::Blur => self.commit_title(window, cx),
            InputEvent::Change | InputEvent::Focus => {}
        }
    }

    fn commit_title(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let typed = self.title.read(cx).value().trim().to_string();
        if typed == shown_title(&self.mode.session().label) {
            return;
        }
        self.run(window, cx, |mode, io| {
            match mode {
                Mode::Record(record) => record.apply_label(&typed, io),
                Mode::File(file) => file.apply_label(&typed, io),
            }
            Step::None
        });
        // An empty or refused name leaves the label as it was: show that again.
        let label = self.mode.session().label.clone();
        self.title.update(cx, |title, cx| {
            title.set_value(shown_title(&label), window, cx)
        });
        self.shown_label = label;
    }

    fn title_focused(&self, window: &Window, cx: &App) -> bool {
        self.title.read(cx).focus_handle(cx).is_focused(window)
    }

    /// The session state: `recording` while audio flows, `paused`, or another.
    fn state(&self) -> &'static str {
        self.mode.session().status.state
    }

    /// Audio flows or waits on a pause: Stop applies, Finish does not yet.
    fn active(&self) -> bool {
        matches!(self.state(), "recording" | "paused")
    }

    fn is_record(&self) -> bool {
        matches!(self.mode, Mode::Record(_))
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
        const ORDER: [Focus; 3] = [Focus::Transcript, Focus::Speakers, Focus::Context];
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

    /// `from` names the control, so the log tells a click from a key.
    fn toggle_pause(&mut self, from: &str, window: &mut Window, cx: &mut Context<Self>) {
        info!(target: "ltn.app", "command: pause or resume ({from})");
        self.run(window, cx, |mode, io| {
            if let Mode::Record(record) = mode {
                record.toggle_pause(io);
            }
            Step::None
        });
    }

    fn stop(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        info!(target: "ltn.app", "command: stop");
        self.run(window, cx, |mode, io| {
            match mode {
                Mode::Record(record) => record.stop_recording(io),
                Mode::File(file) => file.stop_transcription(io),
            }
            Step::None
        });
    }

    /// The final digest, then the saved session. A second press does not
    /// wait for the AI label.
    fn finish(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        info!(target: "ltn.app", "command: finish");
        self.run(window, cx, |mode, io| match mode {
            Mode::Record(record) => record.quit(io),
            Mode::File(file) => file.quit(io),
        });
    }

    fn digest_now(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        info!(target: "ltn.app", "command: digest now");
        self.run(window, cx, |mode, io| {
            match mode {
                Mode::Record(record) => record.force_digest(io),
                Mode::File(file) => file.force_digest(io),
            }
            Step::None
        });
    }

    fn toggle_mute(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        info!(target: "ltn.app", "command: mute or unmute");
        self.run(window, cx, |mode, io| {
            if let Mode::Record(record) = mode {
                record.toggle_mic(io);
            }
            Step::None
        });
    }

    /// The quick action at `index`, from 0.
    fn quick_action(&mut self, index: usize, window: &mut Window, cx: &mut Context<Self>) {
        if index >= self.mode.session().quick_action_count() {
            return;
        }
        let key = (index + 1).to_string();
        info!(target: "ltn.app", "quick action {key} started");
        self.run(window, cx, |mode, io| {
            mode.session_mut().quick_action(&key, io);
            Step::None
        });
    }

    fn rename(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.title.update(cx, |title, cx| title.focus(window, cx));
    }

    fn show_in_finder(&mut self, cx: &mut Context<Self>) {
        cx.reveal_path(self.output_dir());
    }

    /// Left and right choose a speaker; Return names it. True when the key was one of them.
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

    /// A click on a speaker names it.
    fn click_speaker(&mut self, speaker: u32, window: &mut Window, cx: &mut Context<Self>) {
        self.selected_speaker = Some(speaker);
        self.set_focus(Focus::Speakers, window, cx);
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
                    view.run(window, cx, |mode, io| {
                        mode.session_mut().set_speaker_name(speaker, name, io);
                        Step::None
                    });
                });
            },
        );
    }

    /// The template, its quick actions, then the window's keys.
    fn show_help(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let template = self.mode.session().controller.template();
        let metadata = &template.metadata;
        let mut lines = Vec::new();
        for (name, value) in [
            ("Template", &metadata.name),
            ("Description", &metadata.description),
            ("Locale", &metadata.locale),
        ] {
            if !value.is_empty() {
                lines.push(format!("**{name}:** {value}\n"));
            }
        }
        if !template.quick_actions.is_empty() {
            lines.push("### Quick Actions".into());
            for (index, action) in template.quick_actions.iter().enumerate() {
                lines.push(format!("{}. {}", index + 1, action.label));
            }
            lines.push(String::new());
        }
        lines.push("### Keyboard Shortcuts".into());
        lines.push(SHORTCUTS.into());
        dialogs::markdown(window, cx, "Help", &lines.join("\n"), "primary", None, None);
    }

    /// The session context, once it is locked: the copies end with it.
    fn copy_suffix(&self, cx: &App) -> String {
        match self.mode.session().status.context_locked {
            true => self.context.read(cx).value().trim().to_string(),
            false => String::new(),
        }
    }

    fn copy(&mut self, copied: Copied, window: &mut Window, cx: &mut Context<Self>) {
        let context = self.copy_suffix(cx);
        let (mut text, what, empty) = match copied {
            Copied::Digest => {
                let digest = self.mode.session().status.digest.clone();
                (digest, "Digest copied", "No digest to copy")
            }
            Copied::Transcript => {
                let lines: Vec<String> = self
                    .lines
                    .iter()
                    .map(|line| format!("[{}] {}", line.time, line.plain))
                    .collect();
                (
                    lines.join("\n"),
                    "Transcript copied",
                    "No transcript to copy",
                )
            }
        };
        if text.is_empty() {
            notify(window, cx, empty.into(), Severity::Warning, COPY_TIMEOUT);
            return;
        }
        if !context.is_empty() {
            text.push_str(&match copied {
                Copied::Digest => format!("\n\n---\n\n**Session Context**\n\n{context}"),
                Copied::Transcript => format!("\n\n---\nSession Context:\n{context}"),
            });
        }
        cx.write_to_clipboard(ClipboardItem::new_string(text));
        notify(window, cx, what.into(), Severity::Information, COPY_TIMEOUT);
    }

    /// The session folder now. A rename moves it.
    pub fn output_dir(&self) -> &std::path::Path {
        &self.mode.session().output_dir
    }

    /// A recording runs and has not stopped.
    pub fn is_recording(&self) -> bool {
        self.is_record() && !self.status.stopped
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

    /// The keys with no Cmd: Space, Tab, and the speaker keys. A text field
    /// takes them, except Escape, which leaves the field.
    fn on_key(&mut self, event: &KeyDownEvent, window: &mut Window, cx: &mut Context<Self>) {
        let keystroke = &event.keystroke;
        let modifiers = &keystroke.modifiers;
        if modifiers.platform || modifiers.control || modifiers.alt {
            return;
        }
        let key = match (modifiers.shift, keystroke.key.as_str()) {
            (true, "tab") => "shift-tab",
            (_, key) => key,
        };
        let locked = self.mode.session().status.context_locked;
        let in_context = self.context.read(cx).focus_handle(cx).is_focused(window);
        let in_title = self.title_focused(window, cx);
        if (in_context && !locked) || in_title {
            if key == "escape" {
                self.leave_field(in_title, window, cx);
                cx.stop_propagation();
            }
            return;
        }
        let handled = match key {
            "space" if self.is_record() => {
                self.toggle_pause("space key", window, cx);
                true
            }
            "tab" => {
                self.move_focus(1, window, cx);
                true
            }
            "shift-tab" => {
                self.move_focus(-1, window, cx);
                true
            }
            _ if self.focus == Focus::Speakers => self.speaker_key(key, window, cx),
            _ => false,
        };
        if handled {
            cx.stop_propagation();
        }
    }

    /// Escape: the title drops what was typed; the context keeps it.
    fn leave_field(&mut self, in_title: bool, window: &mut Window, cx: &mut Context<Self>) {
        if in_title {
            let label = shown_title(&self.mode.session().label);
            self.title
                .update(cx, |title, cx| title.set_value(label, window, cx));
        }
        self.set_focus(Focus::Transcript, window, cx);
    }
}

/// The label as the title shows it: the folder name's underscores are spaces.
fn shown_title(label: &str) -> String {
    label.replace('_', " ")
}

/// A toolbar icon outside the kit's default set; `icons.rs` embeds it.
/// A worker thread (audio, transcription, speakers) posts its empty return
/// value when it ends. The terminal app drops it too: it carries nothing.
fn is_thread_end(payload: &Payload) -> bool {
    payload.is::<()>()
}

/// An alert in the status strip.
fn badge(text: impl Into<SharedString>, background: Hsla, foreground: Hsla) -> Div {
    div()
        .flex()
        .items_center()
        .gap(space::XS)
        .px(space::XS)
        .rounded(radius::SM)
        .bg(background)
        .text_color(foreground)
        .font_weight(text::STRONG)
        .child(text.into())
}

impl SessionView {
    /// The title field, the template's name, then the commands.
    fn render_title_row(&self, cx: &mut Context<Self>) -> Stateful<Div> {
        pane_header("session-title", &self.drag, cx)
            .child(title_field("title-field", &self.title, cx))
            .child(div().flex_1())
            .child(self.render_live_state(cx))
            .child(self.render_controls(cx))
    }

    /// The state, the clock, the level and any audio alert: the one place to
    /// glance at during a call.
    fn render_live_state(&self, cx: &App) -> Div {
        let theme = cx.theme();
        let status = &self.status;
        let (state, dot) = self.state_label(cx);
        let mut live = div()
            .flex()
            .flex_none()
            .items_center()
            .gap(space::SM)
            .mr(space::SM)
            .child(div().size(size::DOT).rounded_full().bg(dot))
            .child(div().text_color(theme.foreground).child(state))
            .child(
                div()
                    .font_family(text::MONO)
                    .text_color(theme.muted_foreground)
                    .child(status.format_elapsed(Instant::now())),
            );
        if status.recording && self.is_record() {
            live = live.child(self.render_meter(cx));
        }
        if !status.audio_alert.is_empty() {
            live = live.child(badge(
                status.audio_alert.clone(),
                theme.danger,
                theme.danger_foreground,
            ));
        }
        live
    }

    /// Show in Finder, then the controls of the recording itself. Stop is
    /// red, as a record control; Finish is the accent once audio stops.
    fn render_controls(&self, cx: &mut Context<Self>) -> Div {
        let active = self.active();
        let paused = self.state() == "paused";
        let muted = self.status.mic_muted;
        let finishing = self.mode.session().flow.pending_quit;
        let mut tools = div()
            .flex()
            .flex_none()
            .items_center()
            .gap(space::XS)
            .child(
                tool("finder", IconName::FolderOpen, "Show in Finder")
                    .on_click(cx.listener(|view, _, _, cx| view.show_in_finder(cx))),
            );
        if self.is_record() && active {
            let (icon, label) = match paused {
                true => (IconName::Play, "Resume"),
                false => (IconName::Pause, "Pause"),
            };
            tools = tools.child(tool("pause", icon, label).on_click(
                cx.listener(|view, _, window, cx| view.toggle_pause("button", window, cx)),
            ));
            let (icon, label) = match muted {
                true => (icons::extra("mic-off"), "Unmute"),
                false => (Icon::new(IconName::Mic), "Mute"),
            };
            tools = tools.child(
                tool("mute", icon, label)
                    .toggled(muted)
                    .disabled(paused)
                    .on_click(cx.listener(|view, _, window, cx| view.toggle_mute(window, cx))),
            );
        }
        match active {
            true => tools.child(
                Button::new("stop")
                    .danger()
                    .with_size(control::CONTROL)
                    .icon(icons::extra("square-stop"))
                    .label("Stop")
                    .on_click(cx.listener(|view, _, window, cx| view.stop(window, cx))),
            ),
            false => tools.child(
                Button::new("finish")
                    .primary()
                    .with_size(control::CONTROL)
                    .label(if finishing { "Finishing…" } else { "Finish" })
                    .loading(finishing)
                    .tooltip("Write the final digest and save")
                    .on_click(cx.listener(|view, _, window, cx| view.finish(window, cx))),
            ),
        }
    }

    /// The template's quick actions, in one menu.
    fn render_quick_actions(&self) -> Option<impl IntoElement> {
        let labels: Vec<String> = self
            .mode
            .session()
            .controller
            .template()
            .quick_actions
            .iter()
            .map(|action| action.label.clone())
            .collect();
        if labels.is_empty() {
            return None;
        }
        let handle = self.focus_handle.clone();
        let button = inline_tool("quick-actions", icons::extra("zap"), "Quick Actions");
        Some(button.dropdown_menu(move |menu, _, _| {
            let mut menu = menu.action_context(handle.clone());
            for (index, label) in labels.iter().enumerate() {
                if let Some(action) = quick_action(index) {
                    menu = menu.menu(label.clone(), action);
                }
            }
            menu
        }))
    }

    /// The Copy button at the end of a column header.
    fn copy_button(&self, id: &'static str, copied: Copied, cx: &mut Context<Self>) -> Button {
        inline_tool(id, IconName::Copy, "Copy")
            .on_click(cx.listener(move |view, _, window, cx| view.copy(copied, window, cx)))
    }

    fn render_transcript(&self, cx: &mut Context<Self>) -> Div {
        let marked_speaker = self.hovered_speaker.or(match self.focus {
            Focus::Speakers => self.selected_speaker,
            _ => None,
        });
        let highlight = marked_speaker.map(|speaker| {
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
        let marked = cx.theme().primary.opacity(tint::STRONG);
        let muted = cx.theme().muted_foreground;
        let list = list(
            self.transcript.clone(),
            cx.processor(move |view: &mut Self, index: usize, _, _| {
                let Some(line) = view.lines.get(index) else {
                    return div().into_any_element();
                };
                let mine = highlight
                    .as_deref()
                    .is_some_and(|label| line.plain.starts_with(label));
                let turn = index
                    .checked_sub(1)
                    .and_then(|before| view.lines.get(before))
                    .is_none_or(|before| before.speaker != line.speaker);
                let speaker = line.speaker.clone().filter(|_| turn);
                transcript_line(
                    line.time.clone(),
                    speaker,
                    line.text.clone(),
                    mine.then_some(marked),
                    muted,
                )
                .into_any_element()
            }),
        )
        .size_full();
        div()
            .flex()
            .flex_col()
            .flex_none()
            .w(relative(0.4))
            .min_w_0()
            .child(column_header("Transcript", cx).child(self.copy_button(
                "copy-transcript",
                Copied::Transcript,
                cx,
            )))
            .child(
                div()
                    .id("transcript")
                    .flex_1()
                    .min_h_0()
                    .px(space::SM)
                    .text_size(text::READING)
                    .on_click(cx.listener(|view, _, window, cx| {
                        view.selected_speaker = None;
                        view.set_focus(Focus::Transcript, window, cx);
                    }))
                    .map(|column| match self.lines.is_empty() {
                        true => column.child(self.transcript_empty(cx)),
                        false => column.child(list),
                    }),
            )
    }

    /// The speakers, as chips at the strip's start.
    fn render_speakers(&self, cx: &mut Context<Self>) -> Div {
        let chips = self.speakers();
        let focused = self.focus == Focus::Speakers;
        let items = chips.into_iter().map(|(speaker, name, seconds)| {
            // The keys choose a speaker only while the chips have the focus.
            let selected = focused && self.selected_speaker == Some(speaker);
            let shown = if name.is_empty() {
                format!("S{speaker}")
            } else {
                name
            };
            let label = format!(
                "{shown}  {}",
                crate::app::session::format_spoken_time(seconds)
            );
            speaker_chip(("speaker", speaker as usize), label, selected, cx)
                .on_hover(cx.listener(move |view, hovered: &bool, _, cx| {
                    let now = hovered.then_some(speaker);
                    if *hovered || view.hovered_speaker == Some(speaker) {
                        view.hovered_speaker = now;
                        cx.notify();
                    }
                }))
                .on_click(
                    cx.listener(move |view, _, window, cx| view.click_speaker(speaker, window, cx)),
                )
        });
        div()
            .flex()
            .flex_1()
            .min_w_0()
            .items_center()
            .gap(space::XS)
            .overflow_hidden()
            .children(items)
    }

    fn render_notes(&self, cx: &mut Context<Self>) -> Div {
        let context_focused = self.focus == Focus::Context;
        // An empty context folds to its header; a click opens it.
        let context_open = context_focused || !self.context.read(cx).value().is_empty();
        let border = cx.theme().border;
        let background = cx.theme().background;
        div()
            .flex()
            .flex_col()
            .flex_1()
            .min_w_0()
            .child(
                column_header("Notes", cx).child(
                    div()
                        .flex()
                        .items_center()
                        .gap(space::XS)
                        .child(
                            inline_tool("digest", icons::extra("sparkles"), "Digest Now").on_click(
                                cx.listener(|view, _, window, cx| view.digest_now(window, cx)),
                            ),
                        )
                        .children(self.render_quick_actions())
                        .child(self.copy_button("copy-digest", Copied::Digest, cx)),
                ),
            )
            .child(
                div()
                    .id("digest")
                    .flex_1()
                    .min_h_0()
                    .px(space::LG)
                    .pb(space::SM)
                    .text_size(text::READING)
                    .overflow_y_scroll()
                    .map(|column| match self.has_digest() {
                        true => column.child(markdown(&self.digest)),
                        false => column.child(self.digest_empty(cx)),
                    }),
            )
            .child(
                div()
                    .flex()
                    .flex_col()
                    .flex_none()
                    .border_t(size::HAIRLINE)
                    .border_color(border)
                    .when(context_focused, |column| column.h(relative(0.45)))
                    .when(!context_focused && context_open, |column| {
                        column.h(size::CONTEXT_PANEL)
                    })
                    .child(
                        div()
                            .id("context-header")
                            .cursor_pointer()
                            .on_click(cx.listener(|view, _, window, cx| {
                                view.set_focus(Focus::Context, window, cx)
                            }))
                            .child(
                                column_header("Session Context", cx).child(
                                    Icon::new(match context_open {
                                        true => IconName::ChevronDown,
                                        false => IconName::ChevronRight,
                                    })
                                    .size(size::ICON_SM),
                                ),
                            ),
                    )
                    .when(context_open, |column| {
                        column.child(
                            div().flex_1().min_h_0().px(space::SM).pb(space::SM).child(
                                Textarea::new(&self.context)
                                    .bordered(false)
                                    .h_full()
                                    .bg(background),
                            ),
                        )
                    }),
            )
    }

    /// The state with its color, as the strip's first item.
    fn state_label(&self, cx: &App) -> (String, Hsla) {
        let status = &self.status;
        let theme = cx.theme();
        if status.stopped {
            ("Stopped".into(), theme.muted_foreground)
        } else if status.paused {
            ("Paused".into(), theme.warning)
        } else if status.recording && self.is_record() {
            ("Recording".into(), theme.danger)
        } else if status.recording {
            ("Transcribing".into(), theme.primary)
        } else if status.download_percent >= 0 {
            let text = format!(
                "Downloading {} ({}%)",
                status.download_model, status.download_percent
            );
            (text, theme.primary)
        } else if status.audio_status == "loading_model" {
            ("Loading the speech model".into(), theme.primary)
        } else if status.audio_status == "error" {
            ("Error".into(), theme.danger)
        } else {
            ("Idle".into(), theme.muted_foreground)
        }
    }

    /// The input level, one bar per recent reading. Flat means silence.
    fn render_meter(&self, cx: &App) -> Div {
        let theme = cx.theme();
        let (quiet, loud) = (theme.muted_foreground.opacity(tint::FAINT), theme.success);
        let rise = size::METER_HEIGHT - size::METER_BAR;
        let bars = self.status.level_steps().into_iter().map(move |step| {
            let height = size::METER_BAR + rise * (step as f32 / LEVEL_TOP as f32);
            div()
                .w(size::METER_BAR)
                .h(height)
                .rounded_full()
                .bg(if step == 0 { quiet } else { loud })
        });
        div()
            .flex()
            .items_end()
            .gap(space::XXS)
            .h(size::METER_HEIGHT)
            .children(bars)
    }

    /// The transcript before its first line.
    fn transcript_empty(&self, cx: &App) -> Div {
        let line = match self.is_record() {
            true => "Listening. Lines appear here as people speak.",
            false => "Transcribing. Lines appear here as the file plays.",
        };
        empty_state(IconName::Mic, line, cx)
    }

    fn has_digest(&self) -> bool {
        self.mode.session().controller.latest_digest.is_some()
    }

    /// The notes before the first digest.
    fn digest_empty(&self, cx: &App) -> Div {
        let line = match self.status.ai_notes {
            true => format!(
                "Notes appear after {} lines, or when you press Digest Now.",
                self.status.buf_max
            ),
            false => "AI notes are off for this session.".into(),
        };
        empty_state(icons::extra("sparkles"), line, cx)
    }

    /// Speakers at the start; the work in progress and the digest count at
    /// the end.
    fn render_status(&self, cx: &mut Context<Self>) -> Div {
        let status = &self.status;
        let mut right = div().flex().flex_none().items_center().gap(space::MD);
        if status.mic_muted {
            let theme = cx.theme();
            right = right.child(
                badge("Mic muted", theme.secondary, theme.secondary_foreground)
                    .child(icons::extra("mic-off").size(size::ICON_SM)),
            );
        }
        for (busy, text) in [
            (status.transcribing, "Transcribing…".to_string()),
            (status.labeling_speakers, "Labeling speakers…".into()),
            (!status.activity.is_empty(), status.activity.clone()),
        ] {
            if busy {
                right = right.child(text);
            }
        }
        right = right.child(match status.ai_notes {
            true => format!(
                "Next digest: {} of {} lines",
                status.buf_count, status.buf_max
            ),
            false => "AI notes off".into(),
        });
        if let Some(last) = status.last_digest_time {
            let since = (Instant::now() - last).as_secs();
            right = right.child(match since {
                0..60 => format!("Last digest {since}s ago"),
                _ => format!("Last digest {}m ago", since / 60),
            });
        }
        let template = &self.mode.session().controller.template().metadata.name;
        status_strip(cx)
            .justify_between()
            .gap(space::MD)
            .child(
                div()
                    .flex()
                    .flex_1()
                    .min_w_0()
                    .items_center()
                    .gap(space::SM)
                    .child(tag(template.clone(), cx))
                    .child(self.render_speakers(cx)),
            )
            .child(right)
    }
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
        let record = self.is_record();
        let active = self.active();
        let recording = self.state() == "recording";
        // A quick action the template lacks stays off in the menu.
        let quick = self.mode.session().quick_action_count();
        let border = cx.theme().border;
        div()
            .key_context(SESSION_CONTEXT)
            .track_focus(&self.focus_handle)
            .on_key_down(cx.listener(Self::on_key))
            // A command shows in the menu bar only while it applies.
            .when(record && active, |root| {
                root.on_action(cx.listener(|view, _: &TogglePause, window, cx| {
                    view.toggle_pause("menu", window, cx)
                }))
            })
            .when(record && recording, |root| {
                root.on_action(
                    cx.listener(|view, _: &ToggleMute, window, cx| view.toggle_mute(window, cx)),
                )
            })
            .when(active, |root| {
                root.on_action(
                    cx.listener(|view, _: &StopSession, window, cx| view.stop(window, cx)),
                )
            })
            .when(!active, |root| {
                root.on_action(
                    cx.listener(|view, _: &FinishSession, window, cx| view.finish(window, cx)),
                )
            })
            .on_action(cx.listener(|view, _: &DigestNow, window, cx| view.digest_now(window, cx)))
            .when(quick >= 1, |root| {
                root.on_action(cx.listener(|view, _: &QuickAction1, window, cx| {
                    view.quick_action(0, window, cx)
                }))
            })
            .when(quick >= 2, |root| {
                root.on_action(cx.listener(|view, _: &QuickAction2, window, cx| {
                    view.quick_action(1, window, cx)
                }))
            })
            .when(quick >= 3, |root| {
                root.on_action(cx.listener(|view, _: &QuickAction3, window, cx| {
                    view.quick_action(2, window, cx)
                }))
            })
            .when(quick >= 4, |root| {
                root.on_action(cx.listener(|view, _: &QuickAction4, window, cx| {
                    view.quick_action(3, window, cx)
                }))
            })
            .when(quick >= 5, |root| {
                root.on_action(cx.listener(|view, _: &QuickAction5, window, cx| {
                    view.quick_action(4, window, cx)
                }))
            })
            .on_action(cx.listener(|view, _: &CopyTranscript, window, cx| {
                view.copy(Copied::Transcript, window, cx)
            }))
            .on_action(
                cx.listener(|view, _: &CopyDigest, window, cx| {
                    view.copy(Copied::Digest, window, cx)
                }),
            )
            .on_action(cx.listener(|view, _: &RenameSession, window, cx| view.rename(window, cx)))
            .on_action(cx.listener(|view, _: &ShowInFinder, _, cx| view.show_in_finder(cx)))
            .on_action(cx.listener(|view, _: &ShowHelp, window, cx| view.show_help(window, cx)))
            .size_full()
            .flex()
            .flex_col()
            .child(self.render_title_row(cx))
            .child(
                div()
                    .flex()
                    .flex_1()
                    .min_h_0()
                    .child(self.render_transcript(cx))
                    .child(div().w(size::HAIRLINE).h_full().bg(border))
                    .child(self.render_notes(cx)),
            )
            .child(self.render_status(cx))
    }
}

#[cfg(test)]
mod tests {
    use super::{Payload, is_thread_end};

    #[test]
    fn only_an_empty_result_is_a_thread_end() {
        assert!(is_thread_end(&(Box::new(()) as Payload)));
        assert!(!is_thread_end(&(Box::new(String::new()) as Payload)));
    }
}
