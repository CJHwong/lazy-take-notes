//! Transcribe an audio file with streaming output, then write the final notes
//! (Python's `TranscribeApp`).

use std::path::Path;
use std::time::Duration;

use crossterm::event::KeyEvent;
use ratatui::Frame;
use ratatui::layout::Rect;

use crate::app::events::{Cancel, Event, WorkerState};
use crate::app::headless::Launcher;
use crate::app::logging::see_debug_log;
use crate::tui::host::{Action, Ctx, MouseInput, Msg, Payload, Screen};
use crate::tui::keys::{Binding, action_for, bind};
use crate::tui::screens::session::{Mode, SESSION_BINDINGS, Session};

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

const HELP_ROWS: [&str; 10] = [
    "| Key | Action |",
    "|-----|--------|",
    "| `s` | Stop transcription |",
    "| `d` | Force digest now |",
    "| `c` | Copy focused panel |",
    "| `Tab` | Switch panel focus |",
    "| `l` | Rename session |",
    "| `o` | Open session directory |",
    "| `h` | Toggle this help |",
    "| `q` | Quit |",
];

struct FileMode {
    worker_done: bool,
}

impl Mode for FileMode {
    fn hints(&self, state: &str) -> &'static str {
        match state {
            "recording" => "[s] stop  [d] digest  [c] copy  [l] label  [h] help",
            _ => "[c] copy  [l] label  [o] open  [h] help  [q] quit",
        }
    }

    fn help_rows(&self) -> &'static [&'static str] {
        &HELP_ROWS
    }

    fn worker_busy(&self) -> bool {
        !self.worker_done
    }

    fn move_folder(&mut self, from: &Path, to: &Path) -> std::io::Result<()> {
        std::fs::rename(from, to)
    }
}

pub struct TranscribeScreen {
    session: Session,
    mode: FileMode,
    launcher: Option<Launcher>,
    cancel: Cancel,
}

impl TranscribeScreen {
    pub fn new(mut session: Session, silence_threshold: f64, launcher: Launcher) -> Self {
        session.auto_digest = false;
        session.status.mode_label = "Transcribe".into();
        session.status.active_label = "● Transcribing".into();
        session.status.silence_threshold = silence_threshold;
        Self {
            session,
            mode: FileMode { worker_done: false },
            launcher: Some(launcher),
            cancel: Cancel::default(),
        }
    }

    fn on_worker_status(&mut self, state: WorkerState, error: &str, ctx: &mut Ctx) -> Action {
        let session = &mut self.session;
        session.status.audio_status = state.as_str().into();
        session.status.download_percent = -1;
        if matches!(
            state,
            WorkerState::ModelReady | WorkerState::Recording | WorkerState::Error
        ) {
            session.download_closed();
        }
        match state {
            WorkerState::Recording => {
                session.status.set_recording(true);
                session.status.set_stopped(false);
                session.update_hints(&self.mode, "recording");
                ctx.notify("Press d to digest manually");
            }
            WorkerState::Stopped => {
                session.status.set_recording(false);
                session.status.set_stopped(true);
                self.mode.worker_done = true;
                session.update_hints(&self.mode, "stopped");
                if session.flow.pending_quit {
                    return session.continue_pending_quit(&self.mode, ctx);
                }
                if session.needs_final_digest() {
                    // The file is done: write the final notes.
                    session.run_final_digest(ctx);
                }
            }
            WorkerState::Error => {
                self.mode.worker_done = true;
                session.update_hints(&self.mode, "error");
                if !error.is_empty() {
                    ctx.error(
                        format!("Transcription error: {error}{}", see_debug_log()),
                        Duration::from_secs(12),
                    );
                }
                return session.continue_pending_quit(&self.mode, ctx);
            }
            WorkerState::LoadingModel | WorkerState::ModelReady => {
                session.update_hints(&self.mode, "idle");
            }
            WorkerState::Warning => {}
        }
        Action::None
    }

    fn stop_transcription(&mut self, ctx: &mut Ctx) {
        if self.mode.worker_done {
            return;
        }
        self.cancel.cancel();
        ctx.notify_with(
            "Stopping transcription...",
            crate::tui::host::Severity::Information,
            Duration::from_secs(3),
            "",
        );
    }

    fn quit(&mut self, ctx: &mut Ctx) -> Action {
        let session = &mut self.session;
        if session.flow.pending_quit && !session.flow.digest_running && !self.mode.worker_busy() {
            // A second q does not wait for the AI label.
            return session.close_session();
        }
        self.cancel.cancel();
        if session.flow.digest_running {
            ctx.notify_with(
                "Quitting when the digest finishes.",
                crate::tui::host::Severity::Information,
                Duration::from_secs(4),
                "",
            );
        }
        session.flow.pending_quit = true;
        session.continue_pending_quit(&self.mode, ctx)
    }
}

impl Screen for TranscribeScreen {
    fn mounted(&mut self, ctx: &mut Ctx) -> Action {
        self.session.mount(&self.mode, ctx);
        if let Some(launcher) = self.launcher.take() {
            let (poster, cancel) = (ctx.poster(), self.cancel.clone());
            ctx.spawn_thread("file-transcription", move || launcher(poster, cancel));
        }
        Action::None
    }

    fn render(&mut self, frame: &mut Frame, area: Rect, ctx: &Ctx) {
        self.session.render(frame, area, ctx);
    }

    fn mouse(&mut self, input: &MouseInput, _ctx: &mut Ctx) -> Action {
        self.session.mouse(input)
    }

    fn key(&mut self, key: &KeyEvent, name: &str, ctx: &mut Ctx) -> Action {
        // An editable text box takes printable keys before any binding.
        if self.session.typed_into_context(key, name) {
            return Action::None;
        }
        let action = action_for(&TRANSCRIBE_BINDINGS, name);
        let priority = TRANSCRIBE_BINDINGS
            .iter()
            .any(|binding| binding.key == name && binding.priority);
        if priority {
            match action {
                Some("quit_app") => return self.quit(ctx),
                Some("stop_transcription") => {
                    self.stop_transcription(ctx);
                    return Action::None;
                }
                Some("show_help") => {
                    return self
                        .session
                        .screen_key(&mut self.mode, name, ctx)
                        .unwrap_or(Action::None);
                }
                _ => {}
            }
        }
        if let Some(action) = self.session.focused_key(key, name, ctx) {
            return action;
        }
        if action == Some("force_digest") {
            if !self.session.flow.pending_quit {
                self.session.force_digest(ctx);
            }
            return Action::None;
        }
        self.session
            .screen_key(&mut self.mode, name, ctx)
            .unwrap_or(Action::None)
    }

    fn paste(&mut self, text: &str, _ctx: &mut Ctx) -> Action {
        self.session.paste(text);
        Action::None
    }

    fn message(&mut self, msg: Msg, ctx: &mut Ctx) -> (Option<Msg>, Action) {
        let action = match msg {
            Msg::Worker(Event::TranscriptChunk(segments)) => {
                self.session.on_transcript_chunk(&segments, ctx);
                Action::None
            }
            Msg::Worker(Event::WorkerStatus { state, error }) => {
                self.on_worker_status(state, &error, ctx)
            }
            Msg::Worker(Event::ModelDownloadProgress {
                percent,
                model_name,
            }) => self.session.on_download_progress(percent, &model_name),
            Msg::Worker(Event::TranscriptionStatus { active }) => {
                self.session.status.transcribing = active;
                Action::None
            }
            Msg::Task(payload) => match self.session.task_message(payload, &mut self.mode, ctx) {
                Ok(action) => action,
                Err(payload) => return (Some(Msg::Task(payload)), Action::None),
            },
            other => return (Some(other), Action::None),
        };
        (None, action)
    }

    fn resumed(&mut self, result: Option<Payload>, ctx: &mut Ctx) -> Action {
        self.session.resumed(result, &mut self.mode, ctx);
        Action::None
    }

    fn hosts_session(&self) -> bool {
        true
    }

    fn stop_for_shutdown(&mut self, _ctx: &mut Ctx) -> bool {
        // Python's TranscribeApp has no shutdown hook: the app exits at once.
        self.cancel.cancel();
        self.session.abort_tasks();
        false
    }
}
