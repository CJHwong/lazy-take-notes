//! Transcribe an audio file with streaming output, then write the final notes:
//! the screen over `app::session::file` (Python's `TranscribeApp`).

use crossterm::event::KeyEvent;
use ratatui::Frame;
use ratatui::layout::Rect;

use crate::app::headless::Launcher;
use crate::app::session::Step;
use crate::app::session::file::FileRuntime;
use crate::tui::host::{Action, Ctx, MouseInput, Msg, Payload, Screen};
use crate::tui::keys::{Binding, action_for, bind};
use crate::tui::screens::session::{Resumed, SESSION_BINDINGS, Session, SessionView};

pub const TRANSCRIBE_BINDINGS: [Binding; 10] = [
    SESSION_BINDINGS[0],
    SESSION_BINDINGS[1],
    SESSION_BINDINGS[2],
    SESSION_BINDINGS[3],
    SESSION_BINDINGS[4],
    SESSION_BINDINGS[5],
    SESSION_BINDINGS[6],
    SESSION_BINDINGS[7],
    bind("s", "stop_transcription", "Stop").priority(),
    bind("d", "force_digest", "Digest now").hidden(),
];

const HELP_ROWS: [&str; 11] = [
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

fn hints(state: &str) -> &'static str {
    match state {
        "recording" => "[s] stop  [d] digest  [c] copy  [l] label  [h] help",
        _ => "[c] copy  [l] label  [o] open  [h] help  [q] quit",
    }
}

pub struct TranscribeScreen {
    runtime: FileRuntime,
    view: SessionView,
}

impl TranscribeScreen {
    pub fn new(session: Session, silence_threshold: f64, launcher: Launcher) -> Self {
        let mut view = session.view;
        view.status.mode_label = "Transcribe".into();
        view.status.active_label = "● Transcribing".into();
        view.status.silence_threshold = silence_threshold;
        view.hints = hints;
        view.help_rows = &HELP_ROWS;
        Self {
            runtime: FileRuntime::new(session.runtime, launcher),
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
            Some("stop_transcription") => {
                self.runtime.stop_transcription(ctx);
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

impl Screen for TranscribeScreen {
    fn mounted(&mut self, ctx: &mut Ctx) -> Action {
        self.runtime.mount(ctx);
        self.view.mount(&mut self.runtime.session);
        Action::None
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
        let action = action_for(&TRANSCRIBE_BINDINGS, name);
        let priority = TRANSCRIBE_BINDINGS
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
        if action == Some("force_digest") {
            self.runtime.force_digest(ctx);
            return self.show(Step::None);
        }
        let done = self
            .view
            .screen_key(&mut self.runtime.session, name, ctx)
            .unwrap_or(Action::None);
        self.view.sync(&mut self.runtime.session);
        done
    }

    fn paste(&mut self, text: &str, _ctx: &mut Ctx) -> Action {
        self.view.paste(&mut self.runtime.session, text);
        Action::None
    }

    fn message(&mut self, msg: Msg, ctx: &mut Ctx) -> (Option<Msg>, Action) {
        let step = match msg {
            Msg::Worker(event) => match self.runtime.on_event(event, ctx) {
                Ok(step) => step,
                Err(event) => return (Some(Msg::Worker(event)), Action::None),
            },
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

    fn stop_for_shutdown(&mut self, _ctx: &mut Ctx) -> bool {
        self.runtime.stop_for_shutdown()
    }
}
