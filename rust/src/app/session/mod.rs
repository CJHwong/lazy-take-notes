//! A live session without any screen: the controller, the digest and
//! quick-action tasks, the label, the speakers and the quit flow (the logic of
//! Python's `BaseApp`). A front end owns one, forwards keys and worker events
//! to it, and draws what it holds.
//!
//! A digest is three steps: `begin_digest` here, the LLM call on a task,
//! `finish_digest` when the reply comes back. A newer digest aborts the running
//! one and abandons its prompt at once, so the history never holds two prompts.

pub mod file;
pub mod record;
pub mod status_line;
pub mod texts;

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use tokio::task::AbortHandle;
use tracing::{debug, error};

use crate::adapters::os_error::OsError;
use crate::adapters::session_controller::SessionController;
use crate::app::container::safe_label;
use crate::app::logging::see_debug_log;
use crate::app::tasks::{Payload, Tasks};
use crate::domain::session_files::SPEAKER_TRANSCRIPT;
use crate::domain::speaker_turn::SpeakerTurn;
use crate::domain::transcript::TranscriptSegment;
use crate::ports::llm::{ChatResponse, LlmError};
use crate::usecases::digest::DigestRequest;

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    Information,
    Warning,
    Error,
}

/// What a session needs from its front end.
pub trait SessionIo {
    /// Show `text` to the user for `timeout`.
    fn notify(&mut self, text: String, severity: Severity, timeout: Duration);
    /// Background work whose results come back to this session.
    fn tasks(&self) -> Tasks;
    /// End the app now.
    fn exit(&mut self);
    /// The notice when the final digest ends. The terminal names its quit
    /// key; a front end with no such key says less.
    fn final_digest_notice(&self, ok: bool) -> &'static str {
        match ok {
            true => "Final digest ready. Press q to quit.",
            false => "Final digest failed. Press q to quit.",
        }
    }
}

/// Shorthands for `notify`.
impl dyn SessionIo + '_ {
    pub fn info(&mut self, text: impl Into<String>, timeout: Duration) {
        self.notify(text.into(), Severity::Information, timeout);
    }

    pub fn warn(&mut self, text: impl Into<String>, timeout: Duration) {
        self.notify(text.into(), Severity::Warning, timeout);
    }

    pub fn error(&mut self, text: impl Into<String>, timeout: Duration) {
        self.notify(text.into(), Severity::Error, timeout);
    }
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

/// How long a shutdown waits for the audio worker to flush its last chunk.
pub const SHUTDOWN_GRACE: Duration = Duration::from_secs(3);

/// What the front end shows next, besides notices.
#[derive(Debug, PartialEq, Eq)]
pub enum Step {
    None,
    /// A quick action's answer, or its error.
    ShowQuery {
        title: String,
        body: String,
        is_error: bool,
    },
    /// A speech model download started.
    ShowDownload {
        model: String,
    },
    /// The session is over: leave it.
    Close,
}

/// What a mode's worker tells the shared flow.
pub trait Worker {
    /// True while the worker still delivers transcript lines.
    fn worker_busy(&self) -> bool;
    /// Move the session folder.
    fn move_folder(&mut self, from: &Path, to: &Path) -> std::io::Result<()>;
}

/// The quit and final-digest state.
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

/// The session state a front end shows: the status line and the panels.
#[derive(Debug)]
pub struct Status {
    /// The worker state the key hints follow: idle, recording, paused, stopped or error.
    pub state: &'static str,
    /// The worker's last reported state, as Python names it.
    pub audio_status: String,
    pub recording: bool,
    pub paused: bool,
    pub stopped: bool,
    pub mic_muted: bool,
    /// A capture problem that lasts until fixed, such as `MIC LOST`.
    pub audio_alert: String,
    pub transcribing: bool,
    /// A diarization pass runs.
    pub labeling_speakers: bool,
    /// The running digest or quick action, or empty.
    pub activity: String,
    pub download_percent: i32,
    pub download_model: String,
    /// Lines waiting for the next digest.
    pub buf_count: usize,
    pub last_digest_at: Option<Instant>,
    /// Input levels since the front end last took them.
    pub levels: Vec<f64>,
    /// The session context can no longer change (the recording stopped).
    pub context_locked: bool,
    /// The latest digest, and a count that changes each time it does.
    pub digest: String,
    pub digest_version: u64,
    /// Changes each time old transcript lines get new speaker labels.
    pub relabel_version: u64,
}

impl Default for Status {
    fn default() -> Self {
        Self {
            state: "idle",
            audio_status: String::new(),
            recording: false,
            paused: false,
            stopped: false,
            mic_muted: false,
            audio_alert: String::new(),
            transcribing: false,
            labeling_speakers: false,
            activity: String::new(),
            download_percent: -1,
            download_model: String::new(),
            buf_count: 0,
            last_digest_at: None,
            levels: Vec::new(),
            context_locked: false,
            digest: String::new(),
            digest_version: 0,
            relabel_version: 0,
        }
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

/// A digest whose prompt could not be built. It arrives as a task result, as
/// Python's worker posts its failure.
pub struct DigestFailed {
    generation: u64,
    error: String,
    is_final: bool,
}

/// The AI label, back from its task.
pub struct LabelReply(Option<String>);

pub struct SessionRuntime {
    pub controller: SessionController,
    pub output_dir: PathBuf,
    pub label: String,
    pub missing_digest: Vec<String>,
    pub missing_interactive: Vec<String>,
    pub flow: Flow,
    pub status: Status,
    /// Periodic digests start on their own while recording; a file waits for its end.
    pub auto_digest: bool,
    generation: u64,
    in_flight: Option<InFlight>,
    last_persist_error: Option<String>,
    /// The download notice is open.
    download_shown: bool,
}

impl SessionRuntime {
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
            flow: Flow::default(),
            status: Status::default(),
            auto_digest: true,
            generation: 0,
            in_flight: None,
            last_persist_error: None,
            download_shown: false,
        }
    }

    /// Python's `on_mount` for the shared parts.
    pub fn mount(&mut self, io: &mut dyn SessionIo) {
        if !self.missing_interactive.is_empty() {
            let models = self.missing_interactive.join(", ");
            io.warn(
                format!(
                    "Quick actions disabled: model {models} not found. Run: ollama pull {models}"
                ),
                Duration::from_secs(10),
            );
        }
    }

    /// The user edited the session context.
    /// One chip per speaker, in speaker order: the number, the name the user
    /// gave (or empty), and the seconds of speech.
    pub fn speaker_chips(&self) -> Vec<(u32, String, f64)> {
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

    pub fn set_user_context(&mut self, text: String) {
        self.controller.user_context = text;
    }

    // --- Worker events ---

    /// Lines from the worker: keep, save, and maybe start a digest.
    pub fn on_transcript_chunk(&mut self, segments: &[TranscriptSegment], io: &mut dyn SessionIo) {
        debug!(
            target: "ltn.app",
            "TranscriptChunk received: {} segments, total_all={}",
            segments.len(),
            self.controller.all_segments.len() + segments.len()
        );
        let should_digest = self.controller.on_transcript_segments(segments);
        self.report_persist_error(io);
        if !self.controller.speaker_turns.is_empty() {
            // The screen labels each new line now; the saved file must not lag until the next pass.
            self.save_speaker_transcript(io);
        }
        self.status.buf_count = self.controller.digest_state.buffer.len();
        if self.auto_digest
            && should_digest
            && !self.flow.digest_running
            && self.missing_digest.is_empty()
        {
            self.start_digest(false, io);
        }
    }

    /// A diarization pass: keep the turns, save them, relabel the lines.
    pub fn on_speaker_turns(&mut self, turns: Vec<SpeakerTurn>, io: &mut dyn SessionIo) {
        self.controller.set_speaker_turns(turns);
        self.persist_speakers(io);
        self.status.relabel_version += 1;
    }

    /// The user named a speaker.
    pub fn set_speaker_name(&mut self, speaker: u32, name: String, io: &mut dyn SessionIo) {
        self.controller.set_speaker_name(speaker, name);
        self.persist_speakers(io);
        self.status.relabel_version += 1;
    }

    fn persist_speakers(&mut self, io: &mut dyn SessionIo) {
        let text = self.controller.speakers_json();
        if let Err(e) = std::fs::write(self.output_dir.join("speakers.json"), text) {
            self.speakers_not_saved("speakers.json", &e, io);
            return;
        }
        self.save_speaker_transcript(io);
    }

    /// The transcript with the labels shown on screen, in the format `diarize` writes.
    fn save_speaker_transcript(&mut self, io: &mut dyn SessionIo) {
        let text = self.controller.speaker_transcript();
        let path = self.output_dir.join(SPEAKER_TRANSCRIPT);
        if let Err(e) = std::fs::write(path, text) {
            self.speakers_not_saved(SPEAKER_TRANSCRIPT, &e, io);
        }
    }

    fn speakers_not_saved(&mut self, name: &str, e: &std::io::Error, io: &mut dyn SessionIo) {
        let shown = OsError::from_io(e, Some(&self.output_dir.join(name)), None);
        error!(target: "ltn.app", "Could not save {name}: {e}");
        if self.last_persist_error.as_deref() != Some("speakers") {
            self.last_persist_error = Some("speakers".into());
            io.warn(
                format!("Could not save speakers: {shown}"),
                Duration::from_secs(10),
            );
        }
    }

    /// Warn once per distinct disk failure; the session keeps going.
    fn report_persist_error(&mut self, io: &mut dyn SessionIo) {
        let Some(error) = self.controller.take_persist_error() else {
            return;
        };
        if self.last_persist_error.as_deref() == Some(&error) {
            return;
        }
        self.last_persist_error = Some(error.clone());
        io.warn(
            format!("{error}{}", see_debug_log()),
            Duration::from_secs(10),
        );
    }

    /// Download progress: keep it, and show the download notice once.
    pub fn on_download_progress(&mut self, percent: u32, model_name: &str) -> Step {
        self.status.download_percent = percent as i32;
        self.status.download_model = model_name.into();
        if self.download_shown {
            return Step::None;
        }
        self.download_shown = true;
        Step::ShowDownload {
            model: model_name.into(),
        }
    }

    /// The download notice closed itself on this worker state.
    pub fn download_closed(&mut self) {
        self.download_shown = false;
    }

    // --- Digest ---

    /// Start a digest. A running one is aborted and its prompt removed first.
    pub fn start_digest(&mut self, is_final: bool, io: &mut dyn SessionIo) {
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
        let tasks = io.tasks();
        let request = match self.controller.begin_digest(is_final) {
            Ok(request) => request,
            Err(error) => {
                tasks.spawn(async move {
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
        let task = tasks.spawn_or(
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
    pub fn run_final_digest(&mut self, io: &mut dyn SessionIo) {
        if self.flow.final_started {
            return;
        }
        self.flow.final_started = true;
        self.flow.final_attempted = true;
        self.start_digest(true, io);
    }

    /// A reply came back. A reply for a replaced digest is dropped.
    fn on_digest_reply(
        &mut self,
        reply: DigestReply,
        worker: &dyn Worker,
        io: &mut dyn SessionIo,
    ) -> Step {
        let current = self
            .in_flight
            .as_ref()
            .is_some_and(|flight| flight.generation == reply.generation);
        if !current {
            return Step::None;
        }
        let Some(flight) = self.in_flight.take() else {
            return Step::None;
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
                    let text = io.final_digest_notice(true);
                    io.info(text, Duration::from_secs(10));
                }
                self.digest_ready(&markdown, flight.is_final, worker, io)
            }
            Err(error) => self.digest_failed(&error, flight.is_final, worker, io),
        }
    }

    fn on_digest_failed(
        &mut self,
        failed: DigestFailed,
        worker: &dyn Worker,
        io: &mut dyn SessionIo,
    ) -> Step {
        if failed.generation != self.generation || self.in_flight.is_some() {
            return Step::None;
        }
        self.flow.digest_running = false;
        self.digest_failed(&failed.error, failed.is_final, worker, io)
    }

    fn digest_ready(
        &mut self,
        markdown: &str,
        is_final: bool,
        worker: &dyn Worker,
        io: &mut dyn SessionIo,
    ) -> Step {
        self.status.digest = markdown.to_string();
        self.status.digest_version += 1;
        self.status.buf_count = self.controller.digest_state.buffer.len();
        self.status.last_digest_at = Some(Instant::now());
        self.status.activity.clear();
        self.report_persist_error(io);
        if is_final {
            self.flow.final_done = true;
            self.run_label_task(io);
        }
        self.continue_pending_quit(worker, io)
    }

    fn digest_failed(
        &mut self,
        error: &str,
        is_final: bool,
        worker: &dyn Worker,
        io: &mut dyn SessionIo,
    ) -> Step {
        if is_final && !self.flow.pending_quit {
            let text = io.final_digest_notice(false);
            io.error(text, Duration::from_secs(10));
        }
        // Allow a retry after a failed final.
        self.flow.final_started = false;
        self.status.activity.clear();
        io.error(
            format!("Digest failed: {error}{}", see_debug_log()),
            Duration::from_secs(8),
        );
        self.continue_pending_quit(worker, io)
    }

    /// True when the session has notes to finish and AI notes are on.
    pub fn needs_final_digest(&self) -> bool {
        if !self.missing_digest.is_empty() || self.flow.final_done {
            return false;
        }
        let state = &self.controller.digest_state;
        !state.buffer.is_empty() || state.digest_count > 0
    }

    fn ai_off(models: &[String], feature: &str, io: &mut dyn SessionIo) -> bool {
        if models.is_empty() {
            return false;
        }
        io.warn(
            format!(
                "{feature} are off for this session: {} failed the start check. The transcript is still recorded.",
                models.join(", ")
            ),
            Duration::from_secs(6),
        );
        true
    }

    pub fn force_digest(&mut self, io: &mut dyn SessionIo) {
        if Self::ai_off(&self.missing_digest, "AI notes", io) {
            return;
        }
        if self.flow.digest_running {
            io.warn("Digest already in progress", Duration::from_secs(3));
            return;
        }
        if self.controller.digest_state.buffer.is_empty() {
            io.info("Nothing in buffer to digest yet", Duration::from_secs(3));
            return;
        }
        self.start_digest(false, io);
    }

    // --- Quick actions and the label ---

    /// The number of quick actions the template defines.
    pub fn quick_action_count(&self) -> usize {
        self.controller.template().quick_actions.len()
    }

    /// Run quick action `key` ("1" to "5").
    pub fn quick_action(&mut self, key: &str, io: &mut dyn SessionIo) {
        if Self::ai_off(&self.missing_interactive, "Quick actions", io) {
            return;
        }
        if self.flow.digest_running {
            io.warn("Digest in progress, please wait", Duration::from_secs(3));
            return;
        }
        if self.flow.query_running {
            io.warn(
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
        io.tasks().spawn_or(
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

    fn on_query_reply(&mut self, reply: QueryReply) -> Step {
        self.flow.query_running = false;
        let (body, title, is_error) = match reply.result {
            None => return Step::None,
            Some(Ok((text, label))) => (text, label, false),
            Some(Err(error)) => {
                error!(target: "ltn.app", "Quick action failed: {error}");
                (format!("Error: {error}"), reply.label, true)
            }
        };
        self.status.activity.clear();
        Step::ShowQuery {
            title,
            body,
            is_error,
        }
    }

    fn run_label_task(&mut self, io: &mut dyn SessionIo) {
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
        io.tasks()
            .spawn_or(async move { LabelReply(job.await) }, |_| LabelReply(None));
    }

    fn on_label_reply(
        &mut self,
        reply: LabelReply,
        worker: &mut dyn Worker,
        io: &mut dyn SessionIo,
    ) -> Step {
        if let Some(label) = reply.0
            && self.label.is_empty()
        {
            self.apply_label(&label, worker, io);
        }
        self.flow.label_running = false;
        self.continue_pending_quit(worker, io)
    }

    /// Rename the folder to `<timestamp>_<label>` and keep the label.
    pub fn apply_label(&mut self, label: &str, worker: &mut dyn Worker, io: &mut dyn SessionIo) {
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
                io.error(
                    format!("Session folder {new_name} already exists"),
                    Duration::from_secs(8),
                );
                return;
            }
            if let Err(e) = worker.move_folder(&self.output_dir, &new_path) {
                let text = OsError::from_io(&e, Some(&self.output_dir), Some(&new_path));
                error!(target: "ltn.app", "Could not rename {}: {text}", self.output_dir.display());
                io.error(
                    format!("Could not rename the session folder: {text}"),
                    Duration::from_secs(8),
                );
                return;
            }
            self.output_dir = new_path.clone();
            self.controller.relocate(new_path);
        }
        self.label = safe.clone();
        io.info(format!("Session: {safe}"), Duration::from_secs(3));
    }

    // --- Quit ---

    /// After q: wait for the worker, finish the notes once, let the label land, then close.
    pub fn continue_pending_quit(&mut self, worker: &dyn Worker, io: &mut dyn SessionIo) -> Step {
        match self
            .flow
            .quit_step(self.needs_final_digest(), worker.worker_busy())
        {
            QuitStep::Wait => Step::None,
            QuitStep::RunFinal => {
                self.run_final_digest(io);
                Step::None
            }
            QuitStep::Close => self.close_session(),
        }
    }

    /// Leave the session once.
    pub fn close_session(&mut self) -> Step {
        if self.flow.session_closed {
            return Step::None;
        }
        self.flow.session_closed = true;
        self.flow.pending_quit = false;
        Step::Close
    }

    /// A task result for the session. Gives the payload back when it is not one.
    pub fn task_result(
        &mut self,
        payload: Payload,
        worker: &mut dyn Worker,
        io: &mut dyn SessionIo,
    ) -> Result<Step, Payload> {
        let payload = match payload.downcast::<DigestReply>() {
            Ok(reply) => return Ok(self.on_digest_reply(*reply, worker, io)),
            Err(payload) => payload,
        };
        let payload = match payload.downcast::<DigestFailed>() {
            Ok(failed) => return Ok(self.on_digest_failed(*failed, worker, io)),
            Err(payload) => payload,
        };
        let payload = match payload.downcast::<QueryReply>() {
            Ok(reply) => return Ok(self.on_query_reply(*reply)),
            Err(payload) => payload,
        };
        match payload.downcast::<LabelReply>() {
            Ok(reply) => Ok(self.on_label_reply(*reply, worker, io)),
            Err(payload) => Err(payload),
        }
    }

    /// Abort the running tasks when the session goes away.
    pub fn abort_tasks(&mut self) {
        if let Some(flight) = self.in_flight.take() {
            flight.task.abort();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
