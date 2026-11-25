//! A live recording without any screen: the audio and diarization workers,
//! pause, stop, mute, capture health and the quit sequence (the logic of
//! Python's `RecordApp`).

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use tracing::warn;

use crate::adapters::audio::session_folder::SessionFolder;
use crate::app::events::{AudioHealthReport, Cancel, Event, Nudge, Poster, WorkerState};
use crate::app::headless::Launcher;
use crate::app::logging::see_debug_log;
use crate::app::session::{SessionIo, SessionRuntime, Step, Worker};
use crate::app::tasks::Payload;
use crate::ports::audio_source::AudioSource;

/// The diarization worker: (poster, cancel: stop now, finish: the recording is
/// complete, run the last pass).
pub type DiarizationLauncher = Box<dyn FnOnce(Poster, Cancel, Cancel, Nudge) + Send>;

/// What a recording starts. The launchers run on their own threads.
pub struct RecordWorkers {
    pub audio: Launcher,
    /// The diarization worker, when `diarization.enabled` and `save_audio` allow it.
    pub diarization: Option<DiarizationLauncher>,
    /// Shared with the audio worker: set while paused.
    pub paused: Arc<AtomicBool>,
    pub source: Arc<dyn AudioSource>,
    pub folder: Arc<SessionFolder>,
    pub mic_muted: bool,
    /// Python's `.consent_noticed` flag file.
    pub consent_noticed: PathBuf,
    /// A note for the user before the session starts (diarization needs `save_audio`).
    pub startup_warning: Option<String>,
}

pub struct RecordMode {
    audio_stopped: bool,
    worker_done: bool,
    /// The diarization worker runs; after the stop, until its last pass is done.
    diarizing: bool,
    folder: Arc<SessionFolder>,
}

impl Worker for RecordMode {
    fn worker_busy(&self) -> bool {
        self.audio_stopped && (!self.worker_done || self.diarizing)
    }

    /// The WAV writer holds the folder's lock, so it follows the new path.
    fn move_folder(&mut self, _from: &Path, to: &Path) -> std::io::Result<()> {
        self.folder
            .rename(to)
            .map_err(|e| std::io::Error::other(e.to_string()))
    }
}

/// The last health the session reported: mic, system audio, processing.
#[derive(Clone, PartialEq)]
struct Health {
    mic: String,
    system: String,
    processing: String,
}

impl Default for Health {
    fn default() -> Self {
        Self {
            mic: "ok".into(),
            system: "ok".into(),
            processing: "ok".into(),
        }
    }
}

pub struct RecordRuntime {
    pub session: SessionRuntime,
    pub mode: RecordMode,
    audio: Option<Launcher>,
    diarization: Option<DiarizationLauncher>,
    cancel: Cancel,
    /// Stops the diarization worker at once (a shutdown, or the session closing).
    diarization_cancel: Cancel,
    /// Asks the diarization worker for its last pass (the recording is complete).
    diarization_finish: Cancel,
    /// Asks the diarization worker for a pass now (a pause).
    diarization_now: Nudge,
    paused: Arc<AtomicBool>,
    source: Arc<dyn AudioSource>,
    mic_muted: bool,
    shutdown_requested: bool,
    health: Health,
    audio_notice: String,
    pub consent_noticed: PathBuf,
    startup_warning: Option<String>,
}

impl RecordRuntime {
    pub fn new(mut session: SessionRuntime, workers: RecordWorkers) -> Self {
        session.status.mic_muted = workers.mic_muted;
        Self {
            session,
            mode: RecordMode {
                audio_stopped: false,
                worker_done: false,
                diarizing: false,
                folder: workers.folder,
            },
            audio: Some(workers.audio),
            diarization: workers.diarization,
            cancel: Cancel::default(),
            diarization_cancel: Cancel::default(),
            diarization_finish: Cancel::default(),
            diarization_now: Nudge::default(),
            paused: workers.paused,
            source: workers.source,
            mic_muted: workers.mic_muted,
            shutdown_requested: false,
            health: Health::default(),
            audio_notice: String::new(),
            consent_noticed: workers.consent_noticed,
            startup_warning: workers.startup_warning,
        }
    }

    /// The session opened: warn, then start the workers.
    pub fn mount(&mut self, io: &mut dyn SessionIo) {
        self.session.mount(io);
        if let Some(warning) = self.startup_warning.take() {
            io.warn(warning, Duration::from_secs(8));
        }
        self.start_workers(io);
    }

    fn on_health(&mut self, report: &AudioHealthReport, io: &mut dyn SessionIo) {
        let previous = std::mem::replace(
            &mut self.health,
            Health {
                mic: report.mic.clone(),
                system: report.system.clone(),
                processing: report.processing.clone(),
            },
        );
        let alerts = [
            match report.mic.as_str() {
                "lost" => "MIC LOST",
                "silent" => "MIC SILENT",
                _ => "",
            },
            if report.system == "lost" {
                "SYS LOST"
            } else {
                ""
            },
            if report.processing == "error" {
                "AUDIO ERROR"
            } else {
                ""
            },
        ];
        self.session.status.audio_alert = alerts
            .iter()
            .filter(|alert| !alert.is_empty())
            .copied()
            .collect::<Vec<_>>()
            .join("  ");
        announce(io, "Microphone", &previous.mic, &report.mic, &report.detail);
        announce(
            io,
            "System audio",
            &previous.system,
            &report.system,
            &report.detail,
        );
        if report.processing == "error" && previous.processing != "error" {
            io.error(
                format!(
                    "Audio processing keeps failing ({}). Nothing is recorded until it recovers.",
                    report.detail
                ),
                Duration::from_secs(20),
            );
        } else if report.processing == "ok" && previous.processing == "error" {
            io.info("Audio processing recovered.", Duration::from_secs(8));
        }
        if !report.notice.is_empty() && report.notice != self.audio_notice {
            io.warn(report.notice.clone(), Duration::from_secs(15));
        }
        self.audio_notice = report.notice.clone();
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
        if matches!(state, WorkerState::ModelReady | WorkerState::Recording) {
            session.download_closed();
        }
        match state {
            WorkerState::Recording => {
                session.status.recording = true;
                session.status.paused = false;
                session.status.stopped = false;
                session.status.state = "recording";
            }
            WorkerState::Stopped => {
                self.mode.worker_done = true;
                if self.shutdown_requested {
                    io.exit();
                    return Step::None;
                }
                session.status.recording = false;
                session.status.stopped = true;
                session.status.mic_muted = false;
                session.status.state = "stopped";
                if !self.mode.audio_stopped {
                    // The worker ended without [s]: mark the session stopped.
                    warn!(
                        target: "ltn.app",
                        "Audio worker stopped without explicit user stop — marking as stopped"
                    );
                    self.mode.audio_stopped = true;
                    session.status.context_locked = true;
                }
                if self.mode.diarizing {
                    // The WAV is complete: label the speakers in all of it before the
                    // final notes, which carry the labels.
                    self.diarization_finish.cancel();
                    io.info(
                        "Labeling the speakers in the whole recording...",
                        Duration::from_secs(5),
                    );
                    return Step::None;
                }
                return self.after_last_audio(io);
            }
            WorkerState::Warning => {
                if !error.is_empty() {
                    io.warn(error.to_string(), Duration::from_secs(10));
                }
            }
            WorkerState::Error => {
                session.download_closed();
                session.status.state = "error";
                if !error.is_empty() {
                    io.error(
                        format!("Audio error: {error}{}", see_debug_log()),
                        Duration::from_secs(12),
                    );
                }
            }
            WorkerState::LoadingModel | WorkerState::ModelReady => {
                session.status.state = "idle";
            }
        }
        Step::None
    }

    pub fn toggle_pause(&mut self, io: &mut dyn SessionIo) {
        if self.mode.audio_stopped {
            io.warn("Recording already stopped", Duration::from_secs(3));
            return;
        }
        let status = &mut self.session.status;
        if self.paused.load(Ordering::SeqCst) {
            self.paused.store(false, Ordering::SeqCst);
            status.paused = false;
            status.recording = true;
            status.state = "recording";
            io.info("Recording resumed", Duration::from_secs(2));
        } else {
            self.paused.store(true, Ordering::SeqCst);
            // The labels catch up with the audio so far while the session waits.
            self.diarization_now.nudge();
            status.paused = true;
            status.recording = false;
            status.state = "paused";
            io.info("Recording paused", Duration::from_secs(2));
        }
    }

    /// Mark the session stopped: the state and the hints.
    fn show_stopped(&mut self) {
        let status = &mut self.session.status;
        status.recording = false;
        status.paused = false;
        status.stopped = true;
        status.state = "stopped";
    }

    pub fn stop_recording(&mut self, io: &mut dyn SessionIo) {
        if self.mode.audio_stopped {
            return;
        }
        self.mode.audio_stopped = true;
        self.cancel.cancel();
        self.show_stopped();
        self.session.status.mic_muted = false;
        self.session.status.context_locked = true;
        io.info(
            "Recording stopped. You can still browse and run quick actions.",
            Duration::from_secs(5),
        );
    }

    pub fn toggle_mic(&mut self, io: &mut dyn SessionIo) {
        if self.mode.audio_stopped {
            return;
        }
        if self.paused.load(Ordering::SeqCst) {
            io.warn("Muting unavailable while paused", Duration::from_secs(2));
            return;
        }
        self.mic_muted = !self.mic_muted;
        self.source.set_mic_muted(self.mic_muted);
        self.session.status.mic_muted = self.mic_muted;
        let text = if self.mic_muted {
            "Mic muted"
        } else {
            "Mic unmuted"
        };
        io.info(text, Duration::from_secs(2));
    }

    pub fn force_digest(&mut self, io: &mut dyn SessionIo) {
        if !self.session.flow.pending_quit {
            self.session.force_digest(io);
        }
    }

    pub fn recording_or_paused(&self) -> bool {
        self.session.status.recording || self.session.status.paused
    }

    pub fn quit(&mut self, io: &mut dyn SessionIo) -> Step {
        // q does nothing while recording or paused: press s first.
        if self.recording_or_paused() {
            return Step::None;
        }
        let session = &mut self.session;
        if session.flow.pending_quit && !session.flow.digest_running && !self.mode.worker_busy() {
            // A second q does not wait for the AI label.
            return session.close_session();
        }
        self.cancel.cancel();
        if !self.mode.audio_stopped {
            self.mode.audio_stopped = true;
            self.show_stopped();
        }
        let session = &mut self.session;
        if session.flow.digest_running {
            io.info("Quitting when the digest finishes.", Duration::from_secs(4));
        }
        session.flow.pending_quit = true;
        // Waits for the worker's Stopped while the flush still runs.
        session.continue_pending_quit(&self.mode, io)
    }

    /// The audio and the speaker labels are final: the notes can be finished.
    fn after_last_audio(&mut self, io: &mut dyn SessionIo) -> Step {
        let session = &mut self.session;
        if session.flow.pending_quit {
            return session.continue_pending_quit(&self.mode, io);
        }
        if self.mode.audio_stopped && session.needs_final_digest() {
            session.run_final_digest(io);
        }
        Step::None
    }

    fn start_workers(&mut self, io: &mut dyn SessionIo) {
        let tasks = io.tasks();
        if let Some(audio) = self.audio.take() {
            let (poster, cancel) = (tasks.poster(), self.cancel.clone());
            tasks.spawn_thread("audio", move || audio(poster, cancel));
        }
        if let Some(diarization) = self.diarization.take() {
            let poster = tasks.poster();
            let fallback = poster.clone();
            let cancel = self.diarization_cancel.clone();
            let finish = self.diarization_finish.clone();
            let now = self.diarization_now.clone();
            self.mode.diarizing = true;
            tasks.spawn_thread("diarization", move || {
                let run = std::panic::AssertUnwindSafe(|| diarization(poster, cancel, finish, now));
                if std::panic::catch_unwind(run).is_err() {
                    // The notes wait for this: a crashed worker must not hold them.
                    fallback.post(Event::DiarizationFinished);
                }
            });
        }
    }

    /// A worker event. Every worker event belongs to the recording.
    pub fn on_event(&mut self, event: Event, io: &mut dyn SessionIo) -> Step {
        match event {
            Event::TranscriptChunk(segments) => {
                self.session.on_transcript_chunk(&segments, io);
                Step::None
            }
            Event::WorkerStatus { state, error } => self.on_worker_status(state, &error, io),
            Event::AudioHealth(report) => {
                self.on_health(&report, io);
                Step::None
            }
            Event::AudioLevel { rms } => {
                self.session.status.levels.push(f64::from(rms));
                Step::None
            }
            Event::DiarizationFinished => {
                self.mode.diarizing = false;
                self.session.status.labeling_speakers = false;
                match self.mode.worker_done && !self.shutdown_requested {
                    true => self.after_last_audio(io),
                    false => Step::None,
                }
            }
            Event::SpeakerTurns(turns) => {
                self.session.on_speaker_turns(turns, io);
                Step::None
            }
            Event::ModelDownloadProgress {
                percent,
                model_name,
            } => self.session.on_download_progress(percent, &model_name),
            Event::TranscriptionStatus { active } => {
                self.session.status.transcribing = active;
                Step::None
            }
            Event::DiarizationStatus { active } => {
                self.session.status.labeling_speakers = active;
                Step::None
            }
        }
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

    /// End the session for a shutdown signal. True while the audio still flushes.
    pub fn stop_for_shutdown(&mut self) -> bool {
        self.shutdown_requested = true;
        self.mode.audio_stopped = true;
        self.cancel.cancel();
        // A shutdown does not wait for the last speaker pass.
        self.diarization_cancel.cancel();
        self.session.controller.save_context_now();
        self.session.abort_tasks();
        !self.mode.worker_done
    }
}

/// A notice when one source's health changes.
fn announce(io: &mut dyn SessionIo, name: &str, before: &str, after: &str, detail: &str) {
    if after == before {
        return;
    }
    match after {
        "lost" => {
            let reason = match detail.is_empty() {
                true => String::new(),
                false => format!(" ({detail})"),
            };
            io.error(
                format!("{name} lost{reason}. Reconnecting; recording continues."),
                Duration::from_secs(20),
            );
        }
        "silent" => io.warn(
            format!("{name} sends no signal at all. Check its input volume or mute switch."),
            Duration::from_secs(20),
        ),
        _ if matches!(before, "lost" | "silent") => {
            io.info(format!("{name} is back."), Duration::from_secs(8))
        }
        _ => {}
    }
}

impl Drop for RecordRuntime {
    /// Leaving the session (back to the menu) ends the diarization worker.
    fn drop(&mut self) {
        self.diarization_cancel.cancel();
    }
}
