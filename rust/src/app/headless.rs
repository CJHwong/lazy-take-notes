//! Headless sessions: the workers and the controller without a TUI.
//!
//! A worker thread sends events into a channel. One task receives them and is the
//! only one that touches the controller, as in the TUI. Output is files only.
//! `.status.json` shows progress to the `status` command. stderr carries the
//! banner, milestones and summary; stdout carries only the session folder.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

#[cfg(unix)]
use tokio::signal::unix::{SignalKind, signal};
use tokio::sync::Notify;
use tokio::sync::mpsc::UnboundedReceiver;
use tracing::{info, warn};

use crate::adapters::audio::mixed::MixedAudioSource;
use crate::adapters::session_controller::SessionController;
use crate::adapters::session_status::{STATUS_FILE, write_status};
use crate::app::events::{Cancel, Event, Poster, WorkerState};
use crate::app::workers::diarization::diarize_snapshot;
use crate::domain::session_files::{DEBUG_LOG, NOTES, SPEAKER_TRANSCRIPT, TRANSCRIPT};
use crate::domain::session_status::{self as st, SessionStatus};
#[cfg(unix)]
use crate::ports::audio_source::AudioSource as _; // set_mic_muted on the mixer
use crate::ports::diarizer::Diarizer;

pub const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(60);
/// After a stop request, a stuck worker gets this long, so Ctrl-C always returns control.
pub const STOP_GRACE: Duration = Duration::from_secs(30);
const WORKER_JOIN: Duration = Duration::from_secs(5);

/// Milestone lines, plus one status line rewritten in place on a TTY.
///
/// A milestone first commits the shown status line, so it never overwrites live
/// status. Off a TTY the status line is left out, to keep logs clean.
pub struct StatusWriter {
    out: Box<dyn Write + Send>,
    tty: bool,
    pending: bool,
}

impl StatusWriter {
    pub fn new(out: Box<dyn Write + Send>, tty: bool) -> Self {
        Self {
            out,
            tty,
            pending: false,
        }
    }

    pub fn stderr() -> Self {
        use std::io::IsTerminal;
        Self::new(Box::new(std::io::stderr()), std::io::stderr().is_terminal())
    }

    pub fn line(&mut self, message: &str) {
        if self.tty && self.pending {
            let _ = self.out.write_all(b"\n");
            self.pending = false;
        }
        let _ = writeln!(self.out, "{message}");
        let _ = self.out.flush();
    }

    pub fn status(&mut self, message: &str) {
        if !self.tty {
            return;
        }
        let _ = write!(self.out, "\r{message}\x1b[K");
        self.pending = true;
        let _ = self.out.flush();
    }

    pub fn finish(&mut self) {
        if self.tty && self.pending {
            let _ = self.out.write_all(b"\n");
            let _ = self.out.flush();
            self.pending = false;
        }
    }
}

type Writer = Arc<Mutex<StatusWriter>>;

fn report(writer: &Writer, message: &str) {
    writer
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .line(message);
}

pub fn now_iso() -> String {
    chrono::Local::now().format("%Y-%m-%dT%H:%M:%S").to_string()
}

fn iso_elapsed(start: &str, end: &str) -> f64 {
    let parse = |text: &str| chrono::NaiveDateTime::parse_from_str(text, "%Y-%m-%dT%H:%M:%S");
    match (parse(start), parse(end)) {
        (Ok(start), Ok(end)) => (end - start).as_seconds_f64(),
        _ => 0.0,
    }
}

fn clock(seconds: f64) -> String {
    let total = seconds.max(0.0) as u64;
    format!("{:02}:{:02}", total / 60, total % 60)
}

fn plural(count: u64) -> &'static str {
    if count == 1 { "" } else { "s" }
}

pub fn heartbeat_line(activity: &str, elapsed: f64, segments: u64, digests: u64) -> String {
    format!(
        "⏺ {activity} | {} | {segments} seg | {digests} digest{}",
        clock(elapsed),
        plural(digests)
    )
}

pub fn banner_lines(activity: &str, out_dir: &Path, pid: u32, show_mute: bool) -> Vec<String> {
    let mut lines = vec![
        format!("▶ {activity} → {}", out_dir.display()),
        format!("  stop:  Ctrl-C  (or kill -INT {pid})"),
    ];
    if show_mute {
        lines.push(format!("  mute:  kill -USR1 {pid}"));
    }
    lines.push("  check: lazy-take-notes status".into());
    lines
}

pub fn summary_lines(out_dir: &Path, status: &SessionStatus) -> Vec<String> {
    let elapsed = iso_elapsed(&status.started_at, &status.updated_at);
    let mut lines = vec![format!(
        "■ {} | {} | {} seg | {} digest{}",
        status.state,
        clock(elapsed),
        status.segment_count,
        status.digest_count,
        plural(status.digest_count)
    )];
    if status.segment_count > 0 {
        lines.push(format!(
            "  transcript: {}",
            out_dir.join(TRANSCRIPT.name).display()
        ));
    }
    if status.digest_count > 0 {
        lines.push(format!(
            "  notes:      {}",
            out_dir.join(NOTES.name).display()
        ));
    }
    if !status.error.is_empty() {
        lines.push(format!("  error:      {}", status.error));
    }
    lines
}

/// The worker thread's body: given the way out and the stop flag, it runs to the end.
pub type Launcher = Box<dyn FnOnce(Poster, Cancel) + Send>;

pub struct HeadlessSession {
    controller: SessionController,
    output_dir: PathBuf,
    activity: &'static str,
    /// False after a failed AI check: transcript only, as in the TUI.
    ai_notes: bool,
    /// The mixer, for mute on SIGUSR1. None for a file.
    mic: Option<Arc<MixedAudioSource>>,
    writer: Writer,
    status: SessionStatus,
    cancel: Cancel,
    stop_requested: Arc<Notify>,
    /// (segments, digests) for the heartbeat task; the pump owns the controller.
    counts: Arc<[AtomicU64; 2]>,
    last_download_bucket: Option<u32>,
    last_warning: Option<String>,
    /// The worker reached model_ready or recording.
    started: bool,
    /// Labels the speakers once the recording ends, with its timeout.
    diarizer: Option<(Arc<dyn Diarizer>, Duration)>,
}

impl HeadlessSession {
    pub fn new(
        controller: SessionController,
        output_dir: PathBuf,
        activity: &'static str,
        ai_notes: bool,
        mic: Option<Arc<MixedAudioSource>>,
        writer: StatusWriter,
    ) -> Self {
        let now = now_iso();
        Self {
            controller,
            output_dir,
            activity,
            ai_notes,
            mic,
            writer: Arc::new(Mutex::new(writer)),
            status: SessionStatus {
                state: st::STARTING.into(),
                pid: std::process::id(),
                started_at: now.clone(),
                updated_at: now,
                segment_count: 0,
                digest_count: 0,
                error: String::new(),
            },
            cancel: Cancel::default(),
            stop_requested: Arc::new(Notify::new()),
            counts: Arc::default(),
            last_download_bucket: None,
            last_warning: None,
            started: false,
            diarizer: None,
        }
    }

    pub fn status(&self) -> &SessionStatus {
        &self.status
    }

    /// A stopper the caller can use, like the signal handlers do.
    pub fn stopper(&self) -> Stopper {
        Stopper {
            cancel: self.cancel.clone(),
            notify: self.stop_requested.clone(),
            writer: self.writer.clone(),
            ai_notes: self.ai_notes,
        }
    }

    fn save(&mut self, state: Option<&str>, error: Option<&str>) {
        if let Some(state) = state {
            self.status.state = state.into();
        }
        if let Some(error) = error {
            self.status.error = error.into();
        }
        self.status.segment_count = self.controller.all_segments.len() as u64;
        self.status.digest_count = u64::from(self.controller.digest_state.digest_count);
        self.status.updated_at = now_iso();
        self.counts[0].store(self.status.segment_count, Ordering::Relaxed);
        self.counts[1].store(self.status.digest_count, Ordering::Relaxed);
        if let Err(e) = write_status(&self.output_dir, &self.status) {
            warn!(target: "ltn.headless", "could not write {STATUS_FILE}: {e}");
        }
    }

    fn report(&self, message: &str) {
        report(&self.writer, message);
    }

    /// Run until the worker stops, then write the final notes.
    pub async fn run(&mut self, launcher: Launcher) {
        let (sender, mut events) = tokio::sync::mpsc::unbounded_channel();
        let signals = self.watch_signals();
        self.save(None, None);
        let (poster, cancel) = (Poster::new(sender), self.cancel.clone());
        let worker = std::thread::Builder::new()
            .name("ltn-headless-worker".into())
            .spawn(move || launcher(poster, cancel));
        let beat = self.heartbeat();
        match worker {
            Ok(worker) => {
                self.pump(&mut events).await;
                join_for_a_while(worker).await;
            }
            Err(e) => {
                self.report(&format!("✗ error: {e}"));
                self.save(Some(st::ERROR), Some(&e.to_string()));
            }
        }
        self.finalize().await;
        beat.abort();
        if let Some(signals) = signals {
            signals.abort();
        }
        self.writer
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .finish();
    }

    /// SIGINT and SIGTERM stop; SIGUSR1 toggles the mic. SIGHUP is left alone:
    /// `nohup` ignores it on purpose to keep the recording going.
    #[cfg(unix)]
    fn watch_signals(&self) -> Option<tokio::task::JoinHandle<()>> {
        let mut interrupt = signal(SignalKind::interrupt()).ok()?;
        let mut terminate = signal(SignalKind::terminate()).ok()?;
        let mut user1 = match &self.mic {
            Some(_) => signal(SignalKind::user_defined1()).ok(),
            None => None,
        };
        let stopper = self.stopper();
        let mic = self.mic.clone();
        let writer = self.writer.clone();
        Some(tokio::spawn(async move {
            loop {
                let toggle = async {
                    match user1.as_mut() {
                        Some(user1) => user1.recv().await,
                        None => std::future::pending().await,
                    }
                };
                tokio::select! {
                    _ = interrupt.recv() => stopper.stop(),
                    _ = terminate.recv() => stopper.stop(),
                    _ = toggle => {
                        if let Some(mic) = &mic {
                            let muted = !mic.mic_muted();
                            mic.set_mic_muted(muted);
                            report(&writer, if muted { "⏸ mic muted" } else { "▶ mic unmuted" });
                        }
                    }
                }
            }
        }))
    }

    /// Ctrl-C stops a headless session on Windows. There is no SIGUSR1 mute.
    #[cfg(windows)]
    fn watch_signals(&self) -> Option<tokio::task::JoinHandle<()>> {
        let stopper = self.stopper();
        Some(tokio::spawn(async move {
            loop {
                if tokio::signal::ctrl_c().await.is_err() {
                    break;
                }
                stopper.stop();
            }
        }))
    }

    fn heartbeat(&self) -> tokio::task::JoinHandle<()> {
        let writer = self.writer.clone();
        let activity = self.activity;
        let started_at = self.status.started_at.clone();
        let counts = self.counts.clone();
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(HEARTBEAT_INTERVAL).await;
                let segments = counts[0].load(Ordering::Relaxed);
                let digests = counts[1].load(Ordering::Relaxed);
                let line = heartbeat_line(
                    activity,
                    iso_elapsed(&started_at, &now_iso()),
                    segments,
                    digests,
                );
                writer
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .status(&line);
            }
        })
    }

    async fn next_event(&self, events: &mut UnboundedReceiver<Event>) -> Option<Event> {
        loop {
            if self.cancel.is_cancelled() {
                return match tokio::time::timeout(STOP_GRACE, events.recv()).await {
                    Ok(event) => event,
                    Err(_) => {
                        self.report("⏹ worker did not stop in time; ending session");
                        None
                    }
                };
            }
            tokio::select! {
                event = events.recv() => return event,
                // A stop during the wait starts the grace period now.
                () = self.stop_requested.notified() => {}
            }
        }
    }

    async fn pump(&mut self, events: &mut UnboundedReceiver<Event>) {
        while let Some(event) = self.next_event(events).await {
            match event {
                Event::TranscriptChunk(segments) => self.on_chunk(&segments).await,
                Event::WorkerStatus { state, error } => {
                    if !self.on_status(state, &error) {
                        return;
                    }
                }
                Event::ModelDownloadProgress {
                    percent,
                    model_name,
                } => {
                    self.report_download(percent, &model_name);
                    self.save(Some(st::DOWNLOADING), None);
                }
                // Levels, health and speaker turns carry no headless state.
                _ => {}
            }
        }
    }

    /// Returns false when the session should stop.
    fn on_status(&mut self, state: WorkerState, error: &str) -> bool {
        match state {
            WorkerState::Stopped => return false,
            WorkerState::Error => return !self.handle_error(error),
            WorkerState::ModelReady | WorkerState::Recording => self.started = true,
            _ => {}
        }
        if state == WorkerState::Warning {
            if self.last_warning.as_deref() != Some(error) {
                self.report(&format!("⚠ {error}"));
                self.last_warning = Some(error.to_string());
            }
        } else {
            self.report_state(state);
        }
        let mapped = match state {
            WorkerState::LoadingModel => Some(st::LOADING_MODEL),
            WorkerState::ModelReady | WorkerState::Recording | WorkerState::Warning => {
                Some(st::RECORDING)
            }
            _ => None,
        };
        self.save(mapped, None);
        true
    }

    /// A worker error before the start is fatal. After it, the worker keeps recording.
    fn handle_error(&mut self, error: &str) -> bool {
        if self.cancel.is_cancelled() {
            info!(target: "ltn.headless", "worker error during shutdown: {error}");
            return true;
        }
        if !self.started {
            self.report(&format!("✗ error: {error}"));
            self.save(Some(st::ERROR), Some(error));
            return true;
        }
        self.report(&format!("✗ transcription error (continuing): {error}"));
        false
    }

    fn report_state(&self, state: WorkerState) {
        match state {
            WorkerState::LoadingModel => self.report("… loading model"),
            WorkerState::Recording => self.report(&format!("✓ model ready, {}", self.activity)),
            _ => {}
        }
    }

    fn report_download(&mut self, percent: u32, model_name: &str) {
        let bucket = percent / 20;
        if self.last_download_bucket != Some(bucket) {
            self.last_download_bucket = Some(bucket);
            self.report(&format!("… downloading {model_name} {percent}%"));
        }
    }

    async fn on_chunk(&mut self, segments: &[crate::domain::transcript::TranscriptSegment]) {
        self.last_warning = None; // audio flows again; a later loss warns afresh
        let should_digest = self.controller.on_transcript_segments(segments);
        if should_digest && self.ai_notes {
            self.digest(false).await;
        } else {
            self.save(Some(st::RECORDING), None);
        }
    }

    async fn digest(&mut self, is_final: bool) {
        self.save(Some(st::DIGESTING), None);
        match self.controller.run_digest(is_final).await {
            Ok(_) => {
                let number = self.controller.digest_state.digest_count;
                self.report(&format!("✓ notes.md updated (digest #{number})"));
                self.save(Some(st::RECORDING), None);
            }
            Err(error) => {
                let error = if error.is_empty() {
                    "digest failed".to_string()
                } else {
                    error
                };
                self.report(&format!("✗ digest failed: {error}"));
                self.save(Some(st::RECORDING), Some(&error));
            }
        }
    }

    /// Label the speakers after the recording, before the final notes.
    pub fn with_diarizer(mut self, diarizer: Arc<dyn Diarizer>, timeout: Duration) -> Self {
        self.diarizer = Some((diarizer, timeout));
        self
    }

    async fn finalize(&mut self) {
        if self.status.state == st::ERROR {
            return;
        }
        self.label_speakers().await;
        let state = &self.controller.digest_state;
        if self.ai_notes && (!state.buffer.is_empty() || state.digest_count > 0) {
            self.digest(true).await;
        }
        self.save(Some(st::STOPPED), None);
    }
}

impl HeadlessSession {
    /// One speaker pass over the whole recording, so the final notes carry the
    /// labels, as the TUI's last pass at stop. A failed pass keeps the plain notes.
    async fn label_speakers(&mut self) {
        let Some((diarizer, timeout)) = self.diarizer.take() else {
            return;
        };
        if self.controller.all_segments.is_empty() {
            return;
        }
        self.report("… labeling speakers");
        let audio = self.output_dir.join("recording.wav");
        let pass = tokio::task::spawn_blocking(move || {
            diarize_snapshot(diarizer.as_ref(), &audio, timeout)
        })
        .await
        .unwrap_or_else(|e| Err(format!("the speaker pass stopped: {e}")));
        let turns = match pass {
            Ok(turns) => turns,
            Err(e) => {
                warn!(target: "ltn.diarize", "diarization pass failed: {e}");
                self.report(&format!("⚠ speaker labels failed: {e}"));
                return;
            }
        };
        let speakers: std::collections::BTreeSet<u32> = turns.iter().map(|t| t.speaker).collect();
        info!(target: "ltn.diarize", "diarization pass: {} turns, {} speakers", turns.len(), speakers.len());
        self.controller.set_speaker_turns(turns);
        let files = [
            ("speakers.json", self.controller.speakers_json()),
            (SPEAKER_TRANSCRIPT, self.controller.speaker_transcript()),
        ];
        for (name, text) in files {
            if let Err(e) = std::fs::write(self.output_dir.join(name), text) {
                self.report(&format!("⚠ could not save {name}: {e}"));
                return;
            }
        }
        self.report(&format!(
            "✓ {SPEAKER_TRANSCRIPT} written ({} speakers)",
            speakers.len()
        ));
    }
}

/// Wait up to 5 s for the worker thread. A stuck one is left behind.
async fn join_for_a_while(worker: std::thread::JoinHandle<()>) {
    let deadline = tokio::time::Instant::now() + WORKER_JOIN;
    while !worker.is_finished() && tokio::time::Instant::now() < deadline {
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    if worker.is_finished() {
        let _ = worker.join();
    }
}

/// Ends a session the way SIGINT does.
#[derive(Clone)]
pub struct Stopper {
    cancel: Cancel,
    notify: Arc<Notify>,
    writer: Writer,
    ai_notes: bool,
}

impl Stopper {
    pub fn stop(&self) {
        if !self.cancel.is_cancelled() {
            let then = if self.ai_notes {
                ", then the final digest"
            } else {
                ""
            };
            report(
                &self.writer,
                &format!("⏹ stopping… finishing the current chunk{then}"),
            );
        }
        self.cancel.cancel();
        self.notify.notify_one();
    }
}

/// Remove a session folder that holds only the status file and the debug log.
/// The debug log is no record to recover: stderr already has the error, and
/// the same file run again writes the log again. Returns whether the folder is
/// gone. A folder that cannot be removed is kept.
pub fn drop_empty_session(out_dir: &Path) -> bool {
    let Ok(entries) = std::fs::read_dir(out_dir) else {
        return false;
    };
    let files: Vec<PathBuf> = entries.filter_map(Result::ok).map(|e| e.path()).collect();
    let named = |path: &PathBuf, name: &str| path.file_name().is_some_and(|n| n == name);
    let has_status = files.iter().any(|f| named(f, STATUS_FILE));
    let only_status_and_log = files
        .iter()
        .all(|f| named(f, STATUS_FILE) || named(f, DEBUG_LOG.name));
    if !(has_status && only_status_and_log) {
        return false;
    }
    let removed = files
        .iter()
        .try_for_each(std::fs::remove_file)
        .and_then(|()| std::fs::remove_dir(out_dir));
    if let Err(e) = &removed {
        warn!(target: "ltn.headless", "could not remove {}: {e}", out_dir.display());
    }
    removed.is_ok()
}
