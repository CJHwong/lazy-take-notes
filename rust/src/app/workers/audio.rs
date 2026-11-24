//! The live audio loop: audio source in, transcript chunks and status events out.
//!
//! The loop must never end the recording on its own. A failed chunk, a failed
//! transcription or a failing disk is reported, and the loop goes on. Only the
//! cancel flag or a source that is exhausted for good stops it.

use std::collections::BTreeSet;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use tracing::{debug, error, info, warn};

use crate::adapters::audio::session_folder::SessionFolder;
use crate::adapters::audio::wav::{RecorderQueue, start_processed_recorder};
use crate::app::events::{AudioHealthReport, Cancel, Event, Poster, WorkerState};
use crate::app::workers::transcribe_thread::TranscribeThread;
use crate::domain::SAMPLE_RATE;
use crate::domain::transcript::TranscriptSegment;
use crate::ports::audio_source::AudioSource;
use crate::ports::transcriber::{TranscribeError, Transcriber};
use crate::usecases::transcribe_audio::{TranscribeAudio, TranscribeSettings};

/// A chunk in flight may take this long before shutdown or a pause gives up on it.
const IN_FLIGHT_WAIT: Duration = Duration::from_secs(120);
const READ_TIMEOUT: Duration = Duration::from_millis(100);
const LEVEL_EVERY: Duration = Duration::from_millis(100);
const SHUTDOWN_DRAIN: Duration = Duration::from_millis(500);
const RECORDER_JOIN: Duration = Duration::from_secs(5);
const NO_DATA_LOG: Duration = Duration::from_secs(5);

/// Timings a test shortens.
#[derive(Debug, Clone)]
pub struct LoopTiming {
    /// A missing capture grant gives the tap zeros with no error: say so once after this.
    pub no_signal_warn: Duration,
    /// Loop failures in a row for this long mean nothing is being recorded.
    pub loop_alert: Duration,
    pub stats_every: Duration,
}

impl Default for LoopTiming {
    fn default() -> Self {
        Self {
            no_signal_warn: Duration::from_secs(15),
            loop_alert: Duration::from_secs(1),
            stats_every: Duration::from_secs(30),
        }
    }
}

pub struct AudioWorkerConfig {
    pub model_path: String,
    pub language: String,
    pub settings: TranscribeSettings,
    pub recognition_hints: Vec<String>,
    /// Where the WAV goes. None records no audio.
    pub folder: Option<Arc<SessionFolder>>,
    pub save_audio: bool,
    pub timing: LoopTiming,
}

/// Run until cancelled. Returns every segment it transcribed.
pub fn run_audio_worker(
    poster: &Poster,
    cancel: &Cancel,
    paused: &AtomicBool,
    config: AudioWorkerConfig,
    transcriber: Box<dyn Transcriber>,
    source: Arc<dyn AudioSource>,
) -> Vec<TranscriptSegment> {
    let unsaved = UnsavedAudio::new(config.save_audio, poster.clone());
    poster.post(Event::status(WorkerState::LoadingModel));
    let transcriber = load_model(transcriber, &config.model_path, poster, &unsaved);
    poster.post(Event::status(WorkerState::ModelReady));
    let transcription = match TranscribeThread::spawn(transcriber) {
        Ok(thread) => thread,
        Err(e) => {
            error!(target: "ltn.audio", "Could not start the transcription thread: {e}");
            poster.post(Event::status_error(WorkerState::Error, e.to_string()));
            poster.post(Event::status(WorkerState::Stopped));
            return Vec::new();
        }
    };
    let recorder = config
        .folder
        .clone()
        .and_then(|folder| start_recorder(folder, &unsaved));
    let use_case = TranscribeAudio::new(
        config.language.clone(),
        config.settings,
        config.recognition_hints.clone(),
    );
    let mut audio = AudioLoop::new(poster, source, use_case, transcription, unsaved, &config);
    if let Some((queue, _)) = &recorder {
        audio.recorder = Some(queue.clone());
    }
    poster.post(Event::status(WorkerState::Recording));
    if let Err(e) = audio.record(cancel, paused) {
        error!(target: "ltn.audio", "Audio source error: {e}");
        audio.unsaved.keep_because("the audio loop failed");
        poster.post(Event::status_error(WorkerState::Error, e));
    }
    let segments = std::mem::take(&mut audio.all_segments);
    drop(audio); // closes the transcription thread's job channel
    if let Some((queue, writer)) = recorder {
        stop_recorder(&queue, writer);
    }
    poster.post(Event::status(WorkerState::Stopped));
    segments
}

/// A speech model that fails to load records audio only: the audio is the record of the meeting.
fn load_model(
    mut transcriber: Box<dyn Transcriber>,
    model_path: &str,
    poster: &Poster,
    unsaved: &UnsavedAudio,
) -> Box<dyn Transcriber> {
    let Err(e) = transcriber.load_model(model_path) else {
        return transcriber;
    };
    error!(target: "ltn.audio", "Failed to load transcription model; recording audio only: {e}");
    transcriber.close();
    poster.post(Event::status_error(
        WorkerState::Warning,
        format!(
            "Speech model failed to load: {e}. Recording audio only; run transcribe on the recording later."
        ),
    ));
    unsaved.keep_because("the speech model failed to load");
    Box::new(NoTranscription)
}

fn start_recorder(
    folder: Arc<SessionFolder>,
    unsaved: &UnsavedAudio,
) -> Option<(RecorderQueue, JoinHandle<()>)> {
    let notice_target = unsaved.clone();
    let keep = unsaved.keep.clone();
    let started = start_processed_recorder(
        folder,
        SAMPLE_RATE as u32,
        Some(Box::new(move |message| notice_target.notice(message))),
        Some(Box::new(move || keep.load(Ordering::SeqCst))),
    );
    match started {
        Ok(recorder) => Some(recorder),
        Err(e) => {
            warn!(target: "ltn.audio", "Could not start the recording writer: {e}");
            None
        }
    }
}

/// Close the WAV. A stalled disk may hold the writer; give up on it after a while.
fn stop_recorder(queue: &RecorderQueue, writer: JoinHandle<()>) {
    let _ = queue.send(None);
    let deadline = Instant::now() + RECORDER_JOIN;
    while !writer.is_finished() && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(20));
    }
    if writer.is_finished() {
        let _ = writer.join();
    } else {
        warn!(target: "ltn.audio", "The recording writer did not finish in time");
    }
}

/// Stands in for a speech model that failed to load, so the audio still records.
struct NoTranscription;

impl Transcriber for NoTranscription {
    fn load_model(&mut self, _model_path: &str) -> Result<(), TranscribeError> {
        Ok(())
    }

    fn transcribe(
        &mut self,
        _audio: &[f32],
        _language: &str,
        _hints: &[String],
    ) -> Result<Vec<TranscriptSegment>, TranscribeError> {
        Ok(Vec::new())
    }
}

/// With save_audio off, the audio is still written, then deleted at a clean end.
/// A failure keeps it: the audio is then the only full record of the meeting.
#[derive(Clone)]
struct UnsavedAudio {
    keep: Arc<AtomicBool>,
    poster: Poster,
}

impl UnsavedAudio {
    fn new(save_audio: bool, poster: Poster) -> Self {
        Self {
            keep: Arc::new(AtomicBool::new(save_audio)),
            poster,
        }
    }

    fn keep_because(&self, reason: &str) {
        if self.keep.swap(true, Ordering::SeqCst) {
            return;
        }
        warn!(target: "ltn.audio", "save_audio is off, but {reason}; keeping the session audio");
        self.poster.post(Event::status_error(
            WorkerState::Warning,
            format!(
                "save_audio is off, but {reason}. The audio of this meeting is kept in recording.wav, so you can transcribe it later."
            ),
        ));
    }

    /// Disk notices about audio the user turned off would only confuse; the log keeps them.
    fn notice(&self, message: String) {
        if self.keep.load(Ordering::SeqCst) {
            self.poster
                .post(Event::status_error(WorkerState::Warning, message));
        }
    }
}

/// Turns failures in a row into one lasting alert, and clears it.
struct LoopHealth {
    alert_after: Duration,
    failing_since: Option<Instant>,
    failures: u64,
    alerted: bool,
}

impl LoopHealth {
    fn failed(&mut self, error: &str, reported: &Reported, poster: &Poster) {
        let now = Instant::now();
        self.failures += 1;
        let since = *self.failing_since.get_or_insert_with(|| {
            error!(target: "ltn.audio", "Audio loop error; recording continues: {error}");
            poster.post(Event::status_error(
                WorkerState::Warning,
                format!("Audio processing error: {error}. Recording continues."),
            ));
            now
        });
        if self.failures > 1 && self.failures.is_multiple_of(100) {
            error!(target: "ltn.audio", "Audio loop still failing ({} in a row): {error}", self.failures);
        }
        if !self.alerted && now.duration_since(since) >= self.alert_after {
            self.alerted = true;
            poster.post(reported.message("error", error));
        }
    }

    fn succeeded(&mut self, reported: &Reported, poster: &Poster) {
        if self.failing_since.take().is_none() {
            return;
        }
        info!(target: "ltn.audio", "Audio loop recovered after {} failure(s)", self.failures);
        self.failures = 0;
        if std::mem::take(&mut self.alerted) {
            poster.post(reported.message("ok", ""));
        }
    }
}

/// The last health the screen was told: (mic, system, notice).
#[derive(Default, Clone, PartialEq)]
struct Reported(Option<(String, String, String)>);

impl Reported {
    fn message(&self, processing: &str, detail: &str) -> Event {
        let (mic, system, notice) = self
            .0
            .clone()
            .unwrap_or_else(|| ("ok".into(), "ok".into(), String::new()));
        Event::AudioHealth(AudioHealthReport {
            mic,
            system,
            detail: detail.to_string(),
            notice,
            processing: processing.to_string(),
        })
    }
}

/// Counters for the stats line every 30 s.
#[derive(Default)]
struct Stats {
    rms_sum: f64,
    rms_count: u64,
    zero_samples: u64,
    total_samples: u64,
    transcriptions: u64,
}

struct AudioLoop<'a> {
    poster: &'a Poster,
    source: Arc<dyn AudioSource>,
    use_case: TranscribeAudio,
    transcription: TranscribeThread,
    unsaved: UnsavedAudio,
    recorder: Option<RecorderQueue>,
    timing: LoopTiming,
    language: String,
    all_segments: Vec<TranscriptSegment>,
    /// (wall start, first chunk) of the chunk in flight.
    pending: Option<(f64, bool)>,
    samples_fed: usize,
    errors_seen: BTreeSet<String>,
    reported: Reported,
    loop_health: LoopHealth,
    started: Instant,
    last_data: Instant,
    no_data_logged: Option<Instant>,
    level_window: Vec<f32>,
    last_level: Option<Instant>,
    no_signal_warned: bool,
    stats: Stats,
    stats_since: Instant,
    was_paused: bool,
}

impl<'a> AudioLoop<'a> {
    fn new(
        poster: &'a Poster,
        source: Arc<dyn AudioSource>,
        use_case: TranscribeAudio,
        transcription: TranscribeThread,
        unsaved: UnsavedAudio,
        config: &AudioWorkerConfig,
    ) -> Self {
        let now = Instant::now();
        Self {
            poster,
            source,
            use_case,
            transcription,
            unsaved,
            recorder: None,
            timing: config.timing.clone(),
            language: config.language.clone(),
            all_segments: Vec::new(),
            pending: None,
            samples_fed: 0,
            errors_seen: BTreeSet::new(),
            reported: Reported::default(),
            loop_health: LoopHealth {
                alert_after: config.timing.loop_alert,
                failing_since: None,
                failures: 0,
                alerted: false,
            },
            started: now,
            last_data: now,
            no_data_logged: None,
            level_window: Vec::new(),
            last_level: None,
            no_signal_warned: false,
            stats: Stats::default(),
            stats_since: now,
            was_paused: false,
        }
    }

    /// Open, loop until cancelled, flush, close. An error here ends the session's audio.
    fn record(&mut self, cancel: &Cancel, paused: &AtomicBool) -> Result<(), String> {
        self.source
            .open(SAMPLE_RATE as u32, 1)
            .map_err(|e| e.to_string())?;
        info!(target: "ltn.audio", "Audio source opened");
        let now = Instant::now();
        (self.started, self.last_data, self.stats_since) = (now, now, now);
        let result = self.run(cancel, paused).and_then(|()| self.finish());
        if let Err(e) = self.source.close() {
            warn!(target: "ltn.audio", "Audio source close failed: {e}");
        }
        info!(
            target: "ltn.audio",
            "Audio source closed: {} total samples, {} segments",
            self.samples_fed,
            self.all_segments.len()
        );
        result
    }

    fn run(&mut self, cancel: &Cancel, paused: &AtomicBool) -> Result<(), String> {
        while !cancel.is_cancelled() {
            let step = catch_unwind(AssertUnwindSafe(|| self.step(paused)))
                .unwrap_or_else(|panic| Err(panic_text(&panic)));
            match step {
                Ok(Flow::Continue) => self.loop_health.succeeded(&self.reported, self.poster),
                Ok(Flow::Exhausted) => return Ok(()),
                Err(e) => {
                    // One bad chunk must not end the meeting recording.
                    self.loop_health.failed(&e, &self.reported, self.poster);
                    self.unsaved.keep_because("the audio loop hit an error");
                    std::thread::sleep(Duration::from_millis(50));
                }
            }
        }
        Ok(())
    }

    /// One read and what follows. An error fails this chunk only.
    fn step(&mut self, paused: &AtomicBool) -> Result<Flow, String> {
        if paused.load(Ordering::SeqCst) {
            self.pause_step();
            return Ok(Flow::Continue);
        }
        self.was_paused = false;
        self.collect();
        let data = self.source.read(READ_TIMEOUT).map_err(|e| e.to_string())?;
        self.report_sources();
        match data {
            Some(data) => self.take(data),
            None => return Ok(self.no_data()),
        }
        Ok(Flow::Continue)
    }

    fn pause_step(&mut self) {
        if !self.was_paused {
            self.flush_for_pause();
            // Audio that arrived during the flush belongs to the pause.
            self.source.drain();
            self.use_case.reset_buffer();
            self.was_paused = true;
        }
        // Keep draining, so the source's queues do not grow.
        self.source.drain();
        self.collect();
        std::thread::sleep(READ_TIMEOUT);
        self.last_data = Instant::now(); // a pause is not silence
    }

    /// Commit the buffered audio, so the last sentence before a pause is not lost.
    fn flush_for_pause(&mut self) {
        if self.pending.is_some() {
            match self.transcription.wait(IN_FLIGHT_WAIT) {
                Some(reply) => self.apply(reply),
                None => {
                    error!(target: "ltn.audio", "Flushing in-flight transcription on pause failed: timed out");
                    self.unsaved.keep_because("a transcription failed");
                    self.pending = None;
                }
            }
        }
        let Some(chunk) = self.use_case.prepare_buffer() else {
            return;
        };
        if chunk.audio.is_empty() {
            return;
        }
        debug!(target: "ltn.audio", "pause-flush transcribing {} samples", chunk.audio.len());
        self.poster
            .post(Event::TranscriptionStatus { active: true });
        let reply =
            self.transcription
                .remote()
                .transcribe(&chunk.audio, &self.language, &chunk.hints);
        match reply {
            Ok(raw) => {
                let segments =
                    self.use_case
                        .apply_result(raw, chunk.wall_start, chunk.is_first_chunk, true);
                self.publish(segments);
            }
            Err(e) => {
                error!(target: "ltn.audio", "Pause flush transcription failed: {e}");
                self.unsaved.keep_because("a transcription failed");
            }
        }
        self.poster
            .post(Event::TranscriptionStatus { active: false });
    }

    /// Pick up a finished chunk, if any.
    fn collect(&mut self) {
        if self.pending.is_none() {
            return;
        }
        if let Some(reply) = self.transcription.try_take() {
            self.apply(reply);
        }
    }

    fn apply(&mut self, reply: Result<Vec<TranscriptSegment>, TranscribeError>) {
        let Some((wall_start, is_first)) = self.pending.take() else {
            return;
        };
        match reply {
            Ok(raw) => {
                let segments = self.use_case.apply_result(raw, wall_start, is_first, false);
                debug!(target: "ltn.audio", "transcription complete: {} segments", segments.len());
                self.publish(segments);
            }
            Err(e) => {
                error!(target: "ltn.audio", "Transcription failed: {e}");
                self.unsaved.keep_because("a transcription failed");
                self.poster
                    .post(Event::status_error(WorkerState::Error, e.to_string()));
            }
        }
        self.poster
            .post(Event::TranscriptionStatus { active: false });
    }

    fn publish(&mut self, segments: Vec<TranscriptSegment>) {
        if segments.is_empty() {
            return;
        }
        self.all_segments.extend_from_slice(&segments);
        self.poster.post(Event::TranscriptChunk(segments));
    }

    /// Log new source failures, and post the health when it changes.
    fn report_sources(&mut self) {
        let errors: BTreeSet<String> = self.source.source_errors().into_iter().collect();
        let health = self.source.health();
        for new in errors.difference(&self.errors_seen) {
            warn!(target: "ltn.audio", "{new}");
            if health.is_none() {
                self.poster
                    .post(Event::status_error(WorkerState::Warning, new.clone()));
            }
        }
        let Some(health) = health else {
            self.errors_seen = errors;
            return;
        };
        let current = (
            health.mic.as_str().to_string(),
            health.system.as_str().to_string(),
            self.source.device_notice(),
        );
        if self.reported.0.as_ref() != Some(&current) {
            info!(target: "ltn.audio", "audio health: mic={} system={} {}", current.0, current.1, current.2);
            let detail = errors.iter().cloned().collect::<Vec<_>>().join("; ");
            self.reported = Reported(Some(current));
            let event = self.reported.message("ok", &detail);
            self.poster.post(event);
        }
        self.errors_seen = errors;
    }

    fn no_data(&mut self) -> Flow {
        if self.source.exhausted() {
            let elapsed = self.started.elapsed().as_secs_f64();
            error!(
                target: "ltn.audio",
                "Audio stream exhausted after {elapsed:.1}s ({} samples captured, {} segments transcribed)",
                self.samples_fed,
                self.all_segments.len()
            );
            self.poster.post(Event::status_error(
                WorkerState::Error,
                format!("Audio stream stopped unexpectedly after {elapsed:.0}s"),
            ));
            return Flow::Exhausted;
        }
        let gap = self.last_data.elapsed();
        let quiet_log = self
            .no_data_logged
            .is_none_or(|at| at.elapsed() >= NO_DATA_LOG);
        if gap >= NO_DATA_LOG && quiet_log {
            warn!(
                target: "ltn.audio",
                "No audio data for {:.1}s (total_samples={}, segments={})",
                gap.as_secs_f64(),
                self.samples_fed,
                self.all_segments.len()
            );
            self.no_data_logged = Some(Instant::now());
        }
        Flow::Continue
    }

    fn take(&mut self, data: Vec<f32>) {
        self.last_data = Instant::now();
        if let Some(recorder) = &self.recorder {
            let _ = recorder.send(Some(data.clone()));
        }
        self.samples_fed += data.len();
        self.use_case
            .set_session_offset(self.samples_fed as f64 / SAMPLE_RATE as f64);
        self.use_case.feed_audio(&data);
        self.count(&data);
        self.post_level(&data);
        self.warn_no_signal();
        self.log_stats();
        self.start_transcription();
    }

    fn count(&mut self, data: &[f32]) {
        let energy: f64 = data.iter().map(|&s| f64::from(s) * f64::from(s)).sum();
        self.stats.rms_sum += (energy / data.len().max(1) as f64).sqrt();
        self.stats.rms_count += 1;
        self.stats.zero_samples += data.iter().filter(|&&s| s == 0.0).count() as u64;
        self.stats.total_samples += data.len() as u64;
    }

    fn post_level(&mut self, data: &[f32]) {
        self.level_window.extend_from_slice(data);
        if self.last_level.is_some_and(|at| at.elapsed() < LEVEL_EVERY) {
            return;
        }
        let window = std::mem::take(&mut self.level_window);
        let energy: f64 = window.iter().map(|&s| f64::from(s) * f64::from(s)).sum();
        let rms = (energy / window.len().max(1) as f64).sqrt() as f32;
        let rms = if rms.is_finite() { rms } else { 0.0 };
        self.poster.post(Event::AudioLevel { rms });
        self.last_level = Some(Instant::now());
    }

    /// Name the capture permission once, if no system sound arrived in the first seconds.
    fn warn_no_signal(&mut self) {
        if self.no_signal_warned || self.started.elapsed() < self.timing.no_signal_warn {
            return;
        }
        // The permission dialog is open: the user is being asked already.
        if self.source.awaiting_permission() {
            return;
        }
        self.no_signal_warned = true;
        // A source that proves the tap with a test tone shows SYS LOST instead.
        if self.source.verifies_system_audio() || self.source.system_signal_seen() {
            return;
        }
        let message = format!(
            "No system audio received in the first {}s. If you expect it, grant \"System Audio Recording Only\" to this app in System Settings > Privacy & Security > Screen & System Audio Recording.",
            self.timing.no_signal_warn.as_secs()
        );
        warn!(target: "ltn.audio", "{message}");
        self.poster
            .post(Event::status_error(WorkerState::Warning, message));
    }

    fn log_stats(&mut self) {
        if self.stats_since.elapsed() < self.timing.stats_every {
            return;
        }
        let elapsed = self.started.elapsed().as_secs_f64();
        let stats = &self.stats;
        let rms = if stats.rms_count > 0 {
            stats.rms_sum / stats.rms_count as f64
        } else {
            0.0
        };
        let zero_pct = if stats.total_samples > 0 {
            stats.zero_samples as f64 / stats.total_samples as f64 * 100.0
        } else {
            0.0
        };
        // Positive drift: the loop is behind real time, so audio waits in a queue.
        let expected = (elapsed * SAMPLE_RATE as f64) as i64;
        let drift = (expected - self.samples_fed as i64) as f64 / SAMPLE_RATE as f64;
        let queues = self.source.queue_sizes();
        let size = |pick: fn(&crate::ports::audio_source::QueueSizes) -> usize| {
            queues.as_ref().map_or(-1, |q| pick(q) as i64)
        };
        let whole = elapsed as u64;
        info!(
            target: "ltn.audio",
            "Audio stats [{}:{:02}]: {} samples, rms={rms:.3}, zero={zero_pct:.1}%, transcriptions={}, drift={drift:+.2}s, mic_q={}, sys_q={}, sys_buf={}",
            whole / 60,
            whole % 60,
            self.samples_fed,
            stats.transcriptions,
            size(|q| q.mic_blocks),
            size(|q| q.system_blocks),
            size(|q| q.system_buffered),
        );
        self.stats_since = Instant::now();
    }

    fn start_transcription(&mut self) {
        if self.pending.is_some() || !self.use_case.should_trigger() {
            return;
        }
        let Some(chunk) = self.use_case.prepare_buffer() else {
            return;
        };
        debug!(
            target: "ltn.audio",
            "transcription triggered: buffer={} samples, first_chunk={}",
            chunk.audio.len(),
            chunk.is_first_chunk
        );
        self.stats.transcriptions += 1;
        self.pending = Some((chunk.wall_start, chunk.is_first_chunk));
        self.poster
            .post(Event::TranscriptionStatus { active: true });
        self.transcription
            .submit(chunk.audio, &self.language, chunk.hints);
    }

    /// Collect the chunk in flight, take the last audio, and transcribe the rest.
    fn finish(&mut self) -> Result<(), String> {
        if self.pending.is_some() {
            match self.transcription.wait(IN_FLIGHT_WAIT) {
                Some(reply) => self.apply(reply),
                None => {
                    error!(target: "ltn.audio", "In-flight transcription at shutdown: timed out");
                    self.unsaved.keep_because("a transcription failed");
                    self.pending = None;
                }
            }
        }
        let deadline = Instant::now() + SHUTDOWN_DRAIN;
        while Instant::now() < deadline {
            let Ok(Some(data)) = self.source.read(READ_TIMEOUT) else {
                break;
            };
            if let Some(recorder) = &self.recorder {
                let _ = recorder.send(Some(data.clone()));
            }
            self.samples_fed += data.len();
            self.use_case.feed_audio(&data);
        }
        self.use_case
            .set_session_offset(self.samples_fed as f64 / SAMPLE_RATE as f64);
        self.poster
            .post(Event::TranscriptionStatus { active: true });
        let flushed = self.use_case.flush(&mut self.transcription.remote());
        self.poster
            .post(Event::TranscriptionStatus { active: false });
        self.publish(flushed.map_err(|e| e.to_string())?);
        Ok(())
    }
}

enum Flow {
    Continue,
    Exhausted,
}

/// The message a panic carried: a `&str` or a `String`, as `panic!` makes them.
pub fn panic_text(panic: &Box<dyn std::any::Any + Send>) -> String {
    if let Some(text) = panic.downcast_ref::<&str>() {
        return (*text).to_string();
    }
    panic
        .downcast_ref::<String>()
        .cloned()
        .unwrap_or_else(|| "panic".into())
}
