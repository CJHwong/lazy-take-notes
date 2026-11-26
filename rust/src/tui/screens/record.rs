//! Live recording: the screen over `app::session::record` (Python's `RecordApp`).

use crossterm::event::KeyEvent;
use ratatui::Frame;
use ratatui::layout::Rect;

use crate::app::session::record::RecordRuntime;
pub use crate::app::session::record::{DiarizationLauncher, RecordWorkers};
use crate::app::session::{Step, texts};
use crate::tui::host::{Action, Ctx, MouseInput, Msg, Payload, Screen};
use crate::tui::keys::{Binding, action_for, bind};
use crate::tui::screens::modals::MarkdownModal;
use crate::tui::screens::session::{Resumed, SESSION_BINDINGS, Session, SessionView};

pub const RECORD_BINDINGS: [Binding; 12] = [
    SESSION_BINDINGS[0],
    SESSION_BINDINGS[1],
    SESSION_BINDINGS[2],
    SESSION_BINDINGS[3],
    SESSION_BINDINGS[4],
    SESSION_BINDINGS[5],
    SESSION_BINDINGS[6],
    SESSION_BINDINGS[7],
    bind("space", "toggle_pause", "Pause/Resume").priority(),
    bind("s", "stop_recording", "Stop").priority(),
    bind("d", "force_digest", "Digest now").hidden(),
    bind("m", "toggle_mic", "Mute mic").hidden(),
];

pub struct RecordScreen {
    runtime: RecordRuntime,
    view: SessionView,
}

impl RecordScreen {
    pub fn new(session: Session, silence_threshold: f64, workers: RecordWorkers) -> Self {
        let mut view = session.view;
        view.status.mode_label = "Record".into();
        view.status.silence_threshold = silence_threshold;
        view.hints = texts::record_hints;
        view.help_rows = &texts::RECORD_HELP_ROWS;
        Self {
            runtime: RecordRuntime::new(session.runtime, workers),
            view,
        }
    }

    /// Show what the session asked for, after the view caught up with it.
    fn show(&mut self, step: Step) -> Action {
        self.view.sync(&mut self.runtime.session);
        self.view.show(step)
    }

    fn priority_key(&mut self, action: Option<&str>, name: &str, ctx: &mut Ctx) -> Option<Action> {
        let step = match action {
            Some("quit_app") => self.runtime.quit(ctx),
            Some("toggle_pause") => {
                self.runtime.toggle_pause(ctx);
                Step::None
            }
            Some("stop_recording") => {
                self.runtime.stop_recording(ctx);
                Step::None
            }
            Some("show_help") => {
                return self
                    .view
                    .screen_key(&mut self.runtime.session, name, ctx)
                    .or(Some(Action::None));
            }
            _ => return None,
        };
        Some(self.show(step))
    }
}

impl Screen for RecordScreen {
    fn mounted(&mut self, ctx: &mut Ctx) -> Action {
        self.runtime.mount(ctx);
        self.view.mount(&mut self.runtime.session);
        if self.runtime.consent_noticed.exists() {
            return Action::None;
        }
        Action::Push(Box::new(MarkdownModal::consent_notice(
            self.runtime.consent_noticed.clone(),
        )))
    }

    fn render(&mut self, frame: &mut Frame, area: Rect, ctx: &Ctx) {
        self.view
            .render(&mut self.runtime.session, frame, area, ctx);
    }

    fn mouse(&mut self, input: &MouseInput, _ctx: &mut Ctx) -> Action {
        self.view.mouse(&self.runtime.session, input)
    }

    fn key(&mut self, key: &KeyEvent, name: &str, ctx: &mut Ctx) -> Action {
        // An editable text box takes printable keys before any binding.
        if self
            .view
            .typed_into_context(&mut self.runtime.session, key, name)
        {
            return Action::None;
        }
        let action = action_for(&RECORD_BINDINGS, name);
        let priority = RECORD_BINDINGS
            .iter()
            .any(|binding| binding.key == name && binding.priority);
        if priority && let Some(done) = self.priority_key(action, name, ctx) {
            return done;
        }
        if let Some(done) = self
            .view
            .focused_key(&mut self.runtime.session, key, name, ctx)
        {
            return done;
        }
        match action {
            Some("force_digest") => self.runtime.force_digest(ctx),
            Some("toggle_mic") => self.runtime.toggle_mic(ctx),
            _ => {
                let done = self
                    .view
                    .screen_key(&mut self.runtime.session, name, ctx)
                    .unwrap_or(Action::None);
                self.view.sync(&mut self.runtime.session);
                return done;
            }
        }
        self.show(Step::None)
    }

    fn paste(&mut self, text: &str, _ctx: &mut Ctx) -> Action {
        self.view.paste(&mut self.runtime.session, text);
        Action::None
    }

    fn message(&mut self, msg: Msg, ctx: &mut Ctx) -> (Option<Msg>, Action) {
        let step = match msg {
            Msg::Worker(event) => self.runtime.on_event(event, ctx),
            Msg::Task(payload) => match self.runtime.task_result(payload, ctx) {
                Ok(step) => step,
                Err(payload) => return (Some(Msg::Task(payload)), Action::None),
            },
            other => return (Some(other), Action::None),
        };
        (None, self.show(step))
    }

    fn resumed(&mut self, result: Option<Payload>, ctx: &mut Ctx) -> Action {
        match self.view.resumed(result) {
            Resumed::Nothing => {}
            Resumed::SpeakerName(speaker, name) => {
                self.runtime.session.set_speaker_name(speaker, name, ctx)
            }
            Resumed::Label(label) => self.runtime.apply_label(&label, ctx),
        }
        self.show(Step::None)
    }

    fn hosts_session(&self) -> bool {
        true
    }

    /// End the session for a shutdown signal. True while the audio still flushes.
    fn stop_for_shutdown(&mut self, _ctx: &mut Ctx) -> bool {
        self.runtime.stop_for_shutdown()
    }

    fn quit_hint(&self) -> Option<String> {
        self.runtime
            .recording_or_paused()
            .then(|| texts::QUIT_HINT.to_string())
    }
}
