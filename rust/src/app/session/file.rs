//! A file transcription without any screen: the worker, stop, the final notes
//! and the quit sequence (the logic of Python's `TranscribeApp`).

use std::path::Path;
use std::time::Duration;

use crate::app::events::{Cancel, Event, WorkerState};
use crate::app::headless::Launcher;
use crate::app::logging::see_debug_log;
use crate::app::session::{SessionIo, SessionRuntime, Step, Text, Worker};
use crate::app::tasks::Payload;

pub struct FileMode {
    worker_done: bool,
}

impl Worker for FileMode {
    fn worker_busy(&self) -> bool {
        !self.worker_done
    }

    fn move_folder(&mut self, from: &Path, to: &Path) -> std::io::Result<()> {
        std::fs::rename(from, to)
    }
}

pub struct FileRuntime {
    pub session: SessionRuntime,
    pub mode: FileMode,
    launcher: Option<Launcher>,
    cancel: Cancel,
}

impl FileRuntime {
    pub fn new(mut session: SessionRuntime, launcher: Launcher) -> Self {
        // A file waits for its end before the notes.
        session.auto_digest = false;
        Self {
            session,
            mode: FileMode { worker_done: false },
            launcher: Some(launcher),
            cancel: Cancel::default(),
        }
    }

    /// The session opened: start the worker.
    pub fn mount(&mut self, io: &mut dyn SessionIo) {
        self.session.mount(io);
        if let Some(launcher) = self.launcher.take() {
            let tasks = io.tasks();
            let (poster, cancel) = (tasks.poster(), self.cancel.clone());
            tasks.spawn_thread("file-transcription", move || launcher(poster, cancel));
        }
    }

    fn on_worker_status(
        &mut self,
        state: WorkerState,
        error: &str,
        io: &mut dyn SessionIo,
    ) -> Step {
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
                session.status.recording = true;
                session.status.stopped = false;
                session.status.state = "recording";
                let text = io.text(Text::ManualHint);
                io.info(text, Duration::from_secs(5));
            }
            WorkerState::Stopped => {
                session.status.recording = false;
                session.status.stopped = true;
                self.mode.worker_done = true;
                session.status.state = "stopped";
                if session.flow.pending_quit {
                    return session.continue_pending_quit(&self.mode, io);
                }
                if session.needs_final_digest() {
                    // The file is done: write the final notes.
                    session.run_final_digest(io);
                }
            }
            WorkerState::Error => {
                self.mode.worker_done = true;
                session.status.state = "error";
                if !error.is_empty() {
                    io.error(
                        format!("Transcription error: {error}{}", see_debug_log()),
                        Duration::from_secs(12),
                    );
                }
                return session.continue_pending_quit(&self.mode, io);
            }
            WorkerState::LoadingModel | WorkerState::ModelReady => {
                session.status.state = "idle";
            }
            WorkerState::Warning => {}
        }
        Step::None
    }

    pub fn stop_transcription(&mut self, io: &mut dyn SessionIo) {
        if self.mode.worker_done {
            return;
        }
        self.cancel.cancel();
        io.info("Stopping transcription...", Duration::from_secs(3));
    }

    pub fn force_digest(&mut self, io: &mut dyn SessionIo) {
        if !self.session.flow.pending_quit {
            self.session.force_digest(io);
        }
    }

    pub fn quit(&mut self, io: &mut dyn SessionIo) -> Step {
        let session = &mut self.session;
        if session.flow.pending_quit && !session.flow.digest_running && !self.mode.worker_busy() {
            // A second q does not wait for the AI label.
            return session.close_session();
        }
        self.cancel.cancel();
        if session.flow.digest_running {
            let text = io.text(Text::QuitAfterDigest);
            io.info(text, Duration::from_secs(4));
        }
        session.flow.pending_quit = true;
        session.continue_pending_quit(&self.mode, io)
    }

    /// A worker event. Gives it back when the file session does not use it.
    pub fn on_event(&mut self, event: Event, io: &mut dyn SessionIo) -> Result<Step, Event> {
        let step = match event {
            Event::TranscriptChunk(segments) => {
                self.session.on_transcript_chunk(&segments, io);
                Step::None
            }
            Event::WorkerStatus { state, error } => self.on_worker_status(state, &error, io),
            Event::ModelDownloadProgress {
                percent,
                model_name,
            } => self.session.on_download_progress(percent, &model_name),
            Event::TranscriptionStatus { active } => {
                self.session.status.transcribing = active;
                Step::None
            }
            other => return Err(other),
        };
        Ok(step)
    }

    /// A task result. Gives the payload back when it is not the session's.
    pub fn task_result(
        &mut self,
        payload: Payload,
        io: &mut dyn SessionIo,
    ) -> Result<Step, Payload> {
        self.session.task_result(payload, &mut self.mode, io)
    }

    pub fn apply_label(&mut self, label: &str, io: &mut dyn SessionIo) {
        self.session.apply_label(label, &mut self.mode, io);
    }

    pub fn stop_for_shutdown(&mut self) -> bool {
        // Python's TranscribeApp has no shutdown hook: the app exits at once.
        self.cancel.cancel();
        self.session.abort_tasks();
        false
    }
}
