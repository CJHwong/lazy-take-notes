//! Live recording: microphone and system audio, transcription and digests
//! (Python's `RecordApp`).

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use crossterm::event::KeyEvent;
use ratatui::Frame;
use ratatui::layout::Rect;
use tracing::warn;

use crate::adapters::audio::session_folder::SessionFolder;
use crate::app::events::{AudioHealthReport, Cancel, Event, WorkerState};
use crate::app::headless::Launcher;
use crate::app::logging::see_debug_log;
use crate::ports::audio_source::AudioSource;
use crate::tui::host::{Action, Ctx, Msg, Payload, Screen, Severity};
use crate::tui::keys::{Binding, action_for, bind};
use crate::tui::screens::modals::MarkdownModal;
use crate::tui::screens::session::{Mode, SESSION_BINDINGS, Session};

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

const HELP_ROWS: [&str; 12] = [
    "| Key | Action |",
    "|-----|--------|",
    "| `Space` | Pause / Resume |",
    "| `s` | Stop recording |",
    "| `d` | Force digest now |",
    "| `m` | Mute / Unmute mic |",
    "| `c` | Copy focused panel |",
    "| `Tab` | Switch panel focus |",
    "| `l` | Rename session |",
    "| `o` | Open session directory |",
    "| `h` | Toggle this help |",
    "| `q` | Quit |",
];

const QUIT_HINT: &str =
    "Press s to stop and save, then q for the final notes. ctrl+q saves and quits without them.";

/// What the record screen starts. The launchers run on their own threads.
pub struct RecordWorkers {
    pub audio: Launcher,
    /// The diarization worker, when `diarization.enabled` and `save_audio` allow it.
    pub diarization: Option<Launcher>,
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

struct RecordMode {
    audio_stopped: bool,
    worker_done: bool,
    folder: Arc<SessionFolder>,
}

impl Mode for RecordMode {
    fn hints(&self, state: &str) -> &'static str {
        match state {
            "recording" => {
                "[Space] pause  [s] stop  [d] digest  [m] mute mic  [c] copy  [l] label  [h] help"
            }
            "paused" => "[Space] resume  [s] stop  [c] copy  [l] label  [h] help",
            _ => "[c] copy  [l] label  [o] open  [h] help  [q] quit",
        }
    }

    fn help_rows(&self) -> &'static [&'static str] {
        &HELP_ROWS
    }

    fn worker_busy(&self) -> bool {
        self.audio_stopped && !self.worker_done
    }

    /// The WAV writer holds the folder's lock, so it follows the new path.
    fn move_folder(&mut self, _from: &Path, to: &Path) -> std::io::Result<()> {
        self.folder
            .rename(to)
            .map_err(|e| std::io::Error::other(e.to_string()))
    }
}

/// The last health the screen showed: mic, system audio, processing.
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

pub struct RecordScreen {
    session: Session,
    mode: RecordMode,
    audio: Option<Launcher>,
    diarization: Option<Launcher>,
    cancel: Cancel,
    paused: Arc<AtomicBool>,
    source: Arc<dyn AudioSource>,
    mic_muted: bool,
    shutdown_requested: bool,
    health: Health,
    audio_notice: String,
    consent_noticed: PathBuf,
    startup_warning: Option<String>,
}

impl RecordScreen {
    pub fn new(mut session: Session, silence_threshold: f64, workers: RecordWorkers) -> Self {
        session.status.mode_label = "Record".into();
        session.status.silence_threshold = silence_threshold;
        session.status.mic_muted = workers.mic_muted;
        Self {
            session,
            mode: RecordMode {
                audio_stopped: false,
                worker_done: false,
                folder: workers.folder,
            },
            audio: Some(workers.audio),
            diarization: workers.diarization,
            cancel: Cancel::default(),
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

    fn on_health(&mut self, report: &AudioHealthReport, ctx: &mut Ctx) {
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
        announce(
            ctx,
            "Microphone",
            &previous.mic,
            &report.mic,
            &report.detail,
        );
        announce(
            ctx,
            "System audio",
            &previous.system,
            &report.system,
            &report.detail,
        );
        if report.processing == "error" && previous.processing != "error" {
            ctx.error(
                format!(
                    "Audio processing keeps failing ({}). Nothing is recorded until it recovers.",
                    report.detail
                ),
                Duration::from_secs(20),
            );
        } else if report.processing == "ok" && previous.processing == "error" {
            ctx.notify_with(
                "Audio processing recovered.",
                Severity::Information,
                Duration::from_secs(8),
                "",
            );
        }
        if !report.notice.is_empty() && report.notice != self.audio_notice {
            ctx.warn(report.notice.clone(), Duration::from_secs(15));
        }
        self.audio_notice = report.notice.clone();
    }

    fn on_worker_status(&mut self, state: WorkerState, error: &str, ctx: &mut Ctx) -> Action {
        let session = &mut self.session;
        session.status.audio_status = state.as_str().into();
        session.status.download_percent = -1;
        if matches!(state, WorkerState::ModelReady | WorkerState::Recording) {
            session.download_closed();
        }
        match state {
            WorkerState::Recording => {
                session.status.set_recording(true);
                session.status.set_paused(false);
                session.status.set_stopped(false);
                session.update_hints(&self.mode, "recording");
            }
            WorkerState::Stopped => {
                self.mode.worker_done = true;
                if self.shutdown_requested {
                    ctx.exit();
                    return Action::None;
                }
                session.status.set_recording(false);
                session.status.set_stopped(true);
                session.status.mic_muted = false;
                session.update_hints(&self.mode, "stopped");
                if !self.mode.audio_stopped {
                    // The worker ended without [s]: mark the session stopped.
                    warn!(
                        target: "ltn.app",
                        "Audio worker stopped without explicit user stop — marking as stopped"
                    );
                    self.mode.audio_stopped = true;
                    session.context.read_only = true;
                }
                if session.flow.pending_quit {
                    // The flush is complete.
                    return session.continue_pending_quit(&self.mode, ctx);
                }
                if self.mode.audio_stopped && session.needs_final_digest() {
                    session.run_final_digest(ctx);
                }
            }
            WorkerState::Warning => {
                if !error.is_empty() {
                    ctx.warn(error.to_string(), Duration::from_secs(10));
                }
            }
            WorkerState::Error => {
                session.download_closed();
                session.update_hints(&self.mode, "error");
                if !error.is_empty() {
                    ctx.error(
                        format!("Audio error: {error}{}", see_debug_log()),
                        Duration::from_secs(12),
                    );
                }
            }
            WorkerState::LoadingModel | WorkerState::ModelReady => {
                session.update_hints(&self.mode, "idle");
            }
        }
        Action::None
    }

    fn toggle_pause(&mut self, ctx: &mut Ctx) {
        if self.mode.audio_stopped {
            ctx.warn("Recording already stopped", Duration::from_secs(3));
            return;
        }
        let status = &mut self.session.status;
        let state = if self.paused.load(Ordering::SeqCst) {
            self.paused.store(false, Ordering::SeqCst);
            status.set_paused(false);
            status.set_recording(true);
            ctx.notify_with(
                "Recording resumed",
                Severity::Information,
                Duration::from_secs(2),
                "",
            );
            "recording"
        } else {
            self.paused.store(true, Ordering::SeqCst);
            status.set_paused(true);
            status.set_recording(false);
            ctx.notify_with(
                "Recording paused",
                Severity::Information,
                Duration::from_secs(2),
                "",
            );
            "paused"
        };
        self.session.update_hints(&self.mode, state);
    }

    /// Mark the session stopped on screen: the bar, the hints and the context box.
    fn show_stopped(&mut self) {
        let status = &mut self.session.status;
        status.set_recording(false);
        status.set_paused(false);
        status.set_stopped(true);
        self.session.update_hints(&self.mode, "stopped");
    }

    fn stop_recording(&mut self, ctx: &mut Ctx) {
        if self.mode.audio_stopped {
            return;
        }
        self.mode.audio_stopped = true;
        self.cancel.cancel();
        self.show_stopped();
        self.session.status.mic_muted = false;
        self.session.context.read_only = true;
        ctx.notify_with(
            "Recording stopped. You can still browse and run quick actions.",
            Severity::Information,
            Duration::from_secs(5),
            "",
        );
    }

    fn toggle_mic(&mut self, ctx: &mut Ctx) {
        if self.mode.audio_stopped {
            return;
        }
        if self.paused.load(Ordering::SeqCst) {
            ctx.warn("Muting unavailable while paused", Duration::from_secs(2));
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
        ctx.notify_with(text, Severity::Information, Duration::from_secs(2), "");
    }

    fn recording_or_paused(&self) -> bool {
        self.session.status.recording || self.session.status.paused
    }

    fn quit(&mut self, ctx: &mut Ctx) -> Action {
        // q does nothing while recording or paused: press s first.
        if self.recording_or_paused() {
            return Action::None;
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
            ctx.notify_with(
                "Quitting when the digest finishes.",
                Severity::Information,
                Duration::from_secs(4),
                "",
            );
        }
        session.flow.pending_quit = true;
        // Waits for the worker's Stopped while the flush still runs.
        session.continue_pending_quit(&self.mode, ctx)
    }

    fn start_workers(&mut self, ctx: &mut Ctx) {
        if let Some(audio) = self.audio.take() {
            let (poster, cancel) = (ctx.poster(), self.cancel.clone());
            ctx.spawn_thread("audio", move || audio(poster, cancel));
        }
        if let Some(diarization) = self.diarization.take() {
            let (poster, cancel) = (ctx.poster(), self.cancel.clone());
            ctx.spawn_thread("diarization", move || diarization(poster, cancel));
        }
    }
}

/// A toast when one source's health changes.
fn announce(ctx: &mut Ctx, name: &str, before: &str, after: &str, detail: &str) {
    if after == before {
        return;
    }
    match after {
        "lost" => {
            let reason = match detail.is_empty() {
                true => String::new(),
                false => format!(" ({detail})"),
            };
            ctx.error(
                format!("{name} lost{reason}. Reconnecting; recording continues."),
                Duration::from_secs(20),
            );
        }
        "silent" => ctx.warn(
            format!("{name} sends no signal at all. Check its input volume or mute switch."),
            Duration::from_secs(20),
        ),
        _ if matches!(before, "lost" | "silent") => ctx.notify_with(
            format!("{name} is back."),
            Severity::Information,
            Duration::from_secs(8),
            "",
        ),
        _ => {}
    }
}

impl Screen for RecordScreen {
    fn mounted(&mut self, ctx: &mut Ctx) -> Action {
        self.session.mount(&self.mode, ctx);
        if let Some(warning) = self.startup_warning.take() {
            ctx.warn(warning, Duration::from_secs(8));
        }
        self.start_workers(ctx);
        if self.consent_noticed.exists() {
            return Action::None;
        }
        Action::Push(Box::new(MarkdownModal::consent_notice(
            self.consent_noticed.clone(),
        )))
    }

    fn render(&mut self, frame: &mut Frame, area: Rect, ctx: &Ctx) {
        self.session.render(frame, area, ctx);
    }

    fn key(&mut self, key: &KeyEvent, name: &str, ctx: &mut Ctx) -> Action {
        // An editable text box takes printable keys before any binding.
        if self.session.typed_into_context(key, name) {
            return Action::None;
        }
        let action = action_for(&RECORD_BINDINGS, name);
        let priority = RECORD_BINDINGS
            .iter()
            .any(|binding| binding.key == name && binding.priority);
        if priority {
            match action {
                Some("quit_app") => return self.quit(ctx),
                Some("toggle_pause") => {
                    self.toggle_pause(ctx);
                    return Action::None;
                }
                Some("stop_recording") => {
                    self.stop_recording(ctx);
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
        match action {
            Some("force_digest") => {
                if !self.session.flow.pending_quit {
                    self.session.force_digest(ctx);
                }
                Action::None
            }
            Some("toggle_mic") => {
                self.toggle_mic(ctx);
                Action::None
            }
            _ => self
                .session
                .screen_key(&mut self.mode, name, ctx)
                .unwrap_or(Action::None),
        }
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
            Msg::Worker(Event::AudioHealth(report)) => {
                self.on_health(&report, ctx);
                Action::None
            }
            Msg::Worker(Event::AudioLevel { rms }) => {
                self.session.status.push_level(f64::from(rms));
                Action::None
            }
            Msg::Worker(Event::SpeakerTurns(turns)) => {
                self.session.on_speaker_turns(turns, ctx);
                Action::None
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

    /// End the session for a shutdown signal. True while the audio still flushes.
    fn stop_for_shutdown(&mut self, _ctx: &mut Ctx) -> bool {
        self.shutdown_requested = true;
        self.mode.audio_stopped = true;
        self.cancel.cancel();
        self.session.controller.save_context_now();
        self.session.abort_tasks();
        !self.mode.worker_done
    }

    fn quit_hint(&self) -> Option<String> {
        self.recording_or_paused().then(|| QUIT_HINT.to_string())
    }
}
