//! The microphone and system audio, mixed into one stream.
//!
//! A failed source never blocks its peer. Silence is valid audio; only a
//! delivery failure starts recovery, and a lost source is retried with backoff
//! for as long as the stream is open: a headset that reconnects after a minute
//! rejoins the same recording. Gain, mute and the peak limiter apply after the
//! two sources are read.
//!
//! Health: `Lost` while a source is being recovered. `Silent` when the raw mic
//! sends exact zeros, which a real mic never does; a restart cannot fix that
//! (input volume 0, a virtual device), so it is reported only. The system tap
//! also sends exact zeros when nothing plays, so after a silent stretch the test
//! tone plays an inaudible tone; a tap that does not return it is restarted.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use tracing::{info, warn};

use super::buffer::AudioBuffer;
use super::mic_boost::MicBoost;
use crate::ports::audio_source::{AudioError, AudioSource, QueueSizes};
pub use crate::ports::audio_source::{Health, SourceHealth};

pub type TestTone = Box<dyn Fn() -> bool + Send + Sync>;

/// The most the mic boost raises a quiet mic. Python's constructor default.
pub const MAX_MIC_GAIN: f64 = 10.0;

/// The timing rules, as fields so the tests can run them in milliseconds.
#[derive(Debug, Clone)]
pub struct MixTiming {
    pub restart_delays: Vec<Duration>,
    /// Exact zeros from the mic this long: reported as silent.
    pub zero_alarm: Duration,
    /// Exact zeros from the system tap this long: play the test tone.
    pub tone_after: Duration,
    /// No signal this long after the tone: the tap is deaf.
    pub tone_wait: Duration,
    /// A continuous source that delivers nothing this long is lost.
    pub dead_audio: Duration,
    /// How long `close` waits for each reader thread.
    pub join_wait: Duration,
}

impl Default for MixTiming {
    fn default() -> Self {
        let seconds = Duration::from_secs_f64;
        Self {
            restart_delays: vec![seconds(0.3), seconds(1.0), seconds(2.0), seconds(5.0)],
            zero_alarm: seconds(3.0),
            tone_after: seconds(30.0),
            tone_wait: seconds(1.5),
            dead_audio: seconds(3.0),
            join_wait: seconds(2.0),
        }
    }
}

/// A flag that threads can wait on, like Python's `threading.Event`.
#[derive(Default)]
struct Event {
    set: Mutex<bool>,
    changed: Condvar,
}

impl Event {
    fn set(&self) {
        *self.set.lock().expect("event lock") = true;
        self.changed.notify_all();
    }

    fn clear(&self) {
        *self.set.lock().expect("event lock") = false;
    }

    fn is_set(&self) -> bool {
        *self.set.lock().expect("event lock")
    }

    /// Wait until set or until `timeout`. Returns whether it is set.
    fn wait(&self, timeout: Duration) -> bool {
        let guard = self.set.lock().expect("event lock");
        let (guard, _) = self
            .changed
            .wait_timeout_while(guard, timeout, |set| !*set)
            .expect("event lock");
        *guard
    }
}

#[derive(Default)]
struct Status {
    /// `(is_mic, message)` in the order the failures began, as the UI lists them.
    errors: Vec<(bool, String)>,
    silent_mic: bool,
    /// Set when the tap swallowed the test tone. Only a real sample clears it,
    /// because a restarted tap delivers zeros that look healthy.
    deaf_system: bool,
    last_delivery: [Option<Instant>; 2],
}

impl Status {
    fn has_error(&self, is_mic: bool) -> bool {
        self.errors.iter().any(|(key, _)| *key == is_mic)
    }
}

/// What only the caller of `read` touches.
struct MixState {
    system_buffer: Vec<f32>,
    seen_generation: u64,
    boost: MicBoost,
    dropped_system_frames: u64,
    last_drop_log: Option<Instant>,
}

struct Inner {
    mic: Arc<dyn AudioSource>,
    system: Arc<dyn AudioSource>,
    mic_queue: AudioBuffer,
    system_queue: AudioBuffer,
    sample_rate: u32,
    timing: MixTiming,
    test_tone: Option<TestTone>,
    stop: Event,
    ready: Event,
    status: Mutex<Status>,
    /// One lock per source (index: `is_mic`), so a reopen never races a drain or close.
    source_locks: [Mutex<()>; 2],
    system_generation: AtomicU64,
    mic_muted: AtomicBool,
    running: AtomicBool,
    last_mic_rms: Mutex<f64>,
}

struct Reader {
    handle: JoinHandle<()>,
}

pub struct MixedAudioSource {
    inner: Arc<Inner>,
    mix: Mutex<MixState>,
    readers: Mutex<Vec<Reader>>,
}

fn label(is_mic: bool) -> &'static str {
    if is_mic { "mic" } else { "system" }
}

impl MixedAudioSource {
    pub fn new(
        mic: Arc<dyn AudioSource>,
        system: Arc<dyn AudioSource>,
        silence_threshold: f64,
        max_mic_gain: f64,
        test_tone: Option<TestTone>,
        timing: MixTiming,
    ) -> Self {
        let sample_rate = 16_000;
        let limit = (sample_rate / 4) as usize;
        let inner = Inner {
            mic,
            system,
            mic_queue: AudioBuffer::new(limit, sample_rate, "mixed microphone"),
            system_queue: AudioBuffer::new(limit, sample_rate, "mixed system"),
            sample_rate,
            timing,
            test_tone,
            stop: Event::default(),
            ready: Event::default(),
            status: Mutex::default(),
            source_locks: [Mutex::new(()), Mutex::new(())],
            system_generation: AtomicU64::new(0),
            mic_muted: AtomicBool::new(false),
            running: AtomicBool::new(false),
            last_mic_rms: Mutex::new(0.0),
        };
        // The boost aims at the level of speech frames, which sits about twice the average level.
        let mix = MixState {
            system_buffer: Vec::new(),
            seen_generation: 0,
            boost: MicBoost::new(silence_threshold * 5.0, max_mic_gain, sample_rate),
            dropped_system_frames: 0,
            last_drop_log: None,
        };
        Self {
            inner: Arc::new(inner),
            mix: Mutex::new(mix),
            readers: Mutex::default(),
        }
    }

    /// The mic level of the newest block, before gain and mute.
    pub fn last_mic_rms(&self) -> f64 {
        *self.inner.last_mic_rms.lock().expect("rms lock")
    }

    pub fn mic_muted(&self) -> bool {
        self.inner.mic_muted.load(Ordering::Relaxed)
    }

    fn take_microphone(&self, timeout: Duration, system_buffered: bool) -> Option<Vec<f32>> {
        let inner = &self.inner;
        let deadline = Instant::now() + timeout;
        loop {
            inner.ready.clear();
            let blocks = inner.mic_queue.take_all();
            if !blocks.is_empty() {
                return Some(blocks.concat());
            }
            let microphone_failed = inner.status.lock().expect("status lock").has_error(true);
            let system_pending = !inner.system_queue.is_empty() || system_buffered;
            if system_pending && (microphone_failed || !inner.running.load(Ordering::Relaxed)) {
                return None;
            }
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() || inner.stop.is_set() {
                return None;
            }
            inner.ready.wait(remaining);
        }
    }

    fn take_system(&self, mix: &mut MixState) {
        let inner = &self.inner;
        let generation = inner.system_generation.load(Ordering::Relaxed);
        if mix.seen_generation != generation {
            mix.system_buffer.clear();
            mix.seen_generation = generation;
        }
        for block in inner.system_queue.take_all() {
            mix.system_buffer.extend(block);
        }
        let limit = (inner.sample_rate / 4) as usize;
        let discarded = mix.system_buffer.len().saturating_sub(limit);
        if discarded > 0 {
            mix.dropped_system_frames += discarded as u64;
            if mix
                .last_drop_log
                .is_none_or(|at| at.elapsed() >= Duration::from_secs(1))
            {
                let seconds = mix.dropped_system_frames as f64 / f64::from(inner.sample_rate);
                warn!(target: "ltn.audio.mixed", "system audio buffer dropped {} old samples ({seconds:.3}s)", mix.dropped_system_frames);
                mix.last_drop_log = Some(Instant::now());
            }
            mix.system_buffer.drain(..discarded);
        }
    }

    fn join_readers(&self) -> Vec<AudioError> {
        let mut errors = Vec::new();
        let mut readers = self.readers.lock().expect("readers lock");
        let deadline = Instant::now() + self.inner.timing.join_wait;
        let mut alive = Vec::new();
        for reader in readers.drain(..) {
            while !reader.handle.is_finished() && Instant::now() < deadline {
                thread::sleep(Duration::from_millis(5));
            }
            if reader.handle.is_finished() {
                let _ = reader.handle.join();
            } else {
                errors.push(AudioError("Audio reader did not stop".into()));
                alive.push(reader);
            }
        }
        *readers = alive;
        self.inner
            .running
            .store(!readers.is_empty(), Ordering::Relaxed);
        errors
    }
}

/// A block scaled down when its peak passes 0.99, so the mix never clips.
fn limit_peak(mut samples: Vec<f32>) -> Vec<f32> {
    let peak = samples.iter().fold(0.0f32, |peak, s| peak.max(s.abs()));
    if peak > 0.99 {
        let scale = 0.99 / peak;
        samples.iter_mut().for_each(|s| *s *= scale);
    }
    samples
}

fn rms(samples: &[f32]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let dot: f32 = samples.iter().map(|s| s * s).sum();
    (f64::from(dot) / samples.len() as f64).sqrt()
}

/// Per-reader state for an exact-zero stretch and a pending test tone.
#[derive(Default)]
struct ZeroWatch {
    zero_since: Option<Instant>,
    tone_sent_at: Option<Instant>,
}

impl Inner {
    fn source(&self, is_mic: bool) -> &Arc<dyn AudioSource> {
        if is_mic { &self.mic } else { &self.system }
    }

    fn queue(&self, is_mic: bool) -> &AudioBuffer {
        if is_mic {
            &self.mic_queue
        } else {
            &self.system_queue
        }
    }

    /// A reader stuck inside its source reports nothing, so judge it by delivery time.
    fn stalled(&self, status: &Status, is_mic: bool) -> bool {
        let source = self.source(is_mic);
        if !self.running.load(Ordering::Relaxed)
            || !source.continuous_capture()
            || source.awaiting_permission()
        {
            return false;
        }
        status.last_delivery[usize::from(is_mic)]
            .is_some_and(|at| at.elapsed() > self.timing.dead_audio)
    }

    fn set_error(&self, is_mic: bool, error: Option<&str>) {
        let name = if is_mic { "Microphone" } else { "System audio" };
        {
            let mut status = self.status.lock().expect("status lock");
            let slot = status.errors.iter().position(|(key, _)| *key == is_mic);
            match (error, slot) {
                (None, Some(index)) => {
                    status.errors.remove(index);
                }
                (Some(error), Some(index)) => status.errors[index].1 = format!("{name}: {error}"),
                (Some(error), None) => status.errors.push((is_mic, format!("{name}: {error}"))),
                (None, None) => {}
            }
        }
        self.ready.set();
    }

    fn open_source(&self, is_mic: bool) -> Option<AudioError> {
        let error = self.source(is_mic).open(self.sample_rate, 1).err()?;
        warn!(target: "ltn.audio.mixed", "{} audio source failed to open; retrying: {error}", label(is_mic));
        self.set_error(is_mic, Some(&error.0));
        Some(error)
    }

    fn reader(self: Arc<Self>, is_mic: bool, mut open_error: Option<AudioError>) {
        let mut last_delivery = Instant::now();
        let mut watch = ZeroWatch::default();
        while !self.stop.is_set() {
            let attempt = match open_error.take() {
                Some(error) => Err(error),
                None => self.read_source(is_mic, last_delivery).and_then(|block| {
                    if let Some(block) = &block {
                        self.watch_signal(&mut watch, block, is_mic)?;
                    }
                    Ok(block)
                }),
            };
            let block = match attempt {
                Ok(block) => block,
                Err(error) => {
                    watch = ZeroWatch::default();
                    match self.recover(is_mic, error) {
                        Some(block) => Some(block),
                        None => return,
                    }
                }
            };
            let Some(block) = block.filter(|b| !b.is_empty()) else {
                self.stop.wait(Duration::from_millis(5));
                continue;
            };
            if self.stop.is_set() {
                return;
            }
            last_delivery = Instant::now();
            self.status.lock().expect("status lock").last_delivery[usize::from(is_mic)] =
                Some(last_delivery);
            self.set_error(is_mic, None);
            self.queue(is_mic).put(&block);
            self.ready.set();
        }
    }

    fn read_source(
        &self,
        is_mic: bool,
        last_delivery: Instant,
    ) -> Result<Option<Vec<f32>>, AudioError> {
        let source = self.source(is_mic);
        if let Some(block) = source
            .read(Duration::from_millis(50))?
            .filter(|b| !b.is_empty())
        {
            return Ok(Some(block));
        }
        if source.exhausted() {
            return Err(AudioError("capture stream ended unexpectedly".into()));
        }
        if source.awaiting_permission() {
            return Ok(None);
        }
        if source.continuous_capture() && last_delivery.elapsed() >= self.timing.dead_audio {
            return Err(AudioError(
                "capture stopped delivering audio buffers".into(),
            ));
        }
        Ok(None)
    }

    /// Track exact-zero stretches. Fails when the system tap swallowed the test tone.
    fn watch_signal(
        &self,
        watch: &mut ZeroWatch,
        block: &[f32],
        is_mic: bool,
    ) -> Result<(), AudioError> {
        let now = Instant::now();
        if block.iter().any(|&s| s != 0.0) {
            if watch.tone_sent_at.is_some() {
                info!(target: "ltn.audio.mixed", "system audio confirmed: signal after the test tone");
            }
            *watch = ZeroWatch::default();
            if is_mic {
                self.set_silent_mic(false);
            } else {
                self.set_deaf_system(false);
            }
            return Ok(());
        }
        let silent_for = now - *watch.zero_since.get_or_insert(now);
        if is_mic {
            if silent_for >= self.timing.zero_alarm {
                self.set_silent_mic(true);
            }
            return Ok(());
        }
        if self.test_tone.is_none() {
            return Ok(());
        }
        match watch.tone_sent_at {
            None => {
                if silent_for >= self.timing.tone_after && self.play_tone() {
                    info!(target: "ltn.audio.mixed", "system audio silent for {:.0}s; test tone played", silent_for.as_secs_f64());
                    watch.tone_sent_at = Some(Instant::now());
                }
                Ok(())
            }
            Some(sent) if now.saturating_duration_since(sent) >= self.timing.tone_wait => {
                self.set_deaf_system(true);
                Err(AudioError(
                    "system audio did not pick up the test tone".into(),
                ))
            }
            Some(_) => Ok(()),
        }
    }

    fn play_tone(&self) -> bool {
        self.test_tone.as_ref().is_some_and(|tone| tone())
    }

    fn set_silent_mic(&self, silent: bool) {
        let changed = {
            let mut status = self.status.lock().expect("status lock");
            let changed = status.silent_mic != silent;
            status.silent_mic = silent;
            changed
        };
        if changed {
            if silent {
                warn!(target: "ltn.audio.mixed", "microphone sends exact zeros");
            } else {
                info!(target: "ltn.audio.mixed", "microphone signal is back");
            }
            self.ready.set();
        }
    }

    fn set_deaf_system(&self, deaf: bool) {
        let changed = {
            let mut status = self.status.lock().expect("status lock");
            let changed = status.deaf_system != deaf;
            status.deaf_system = deaf;
            changed
        };
        if changed && !deaf {
            info!(target: "ltn.audio.mixed", "system audio picked up signal again");
        }
    }

    fn restart_delay(&self, attempt: usize) -> Duration {
        let delays = &self.timing.restart_delays;
        delays[attempt.min(delays.len() - 1)]
    }

    /// Reopen a failed source with backoff until it delivers or the stream closes.
    fn recover(&self, is_mic: bool, mut error: AudioError) -> Option<Vec<f32>> {
        self.set_error(is_mic, Some(&error.0));
        let mut attempt = 0;
        while !self.stop.wait(self.restart_delay(attempt)) {
            attempt += 1;
            warn!(target: "ltn.audio.mixed", "restarting {} audio source (attempt {attempt}): {error}", label(is_mic));
            match self
                .reopen(is_mic)
                .and_then(|()| self.wait_for_first_block(is_mic))
            {
                Ok(Some(block)) => {
                    info!(target: "ltn.audio.mixed", "{} audio source recovered after {attempt} attempt(s)", label(is_mic));
                    return Some(block);
                }
                Ok(None) => {
                    error = AudioError("capture did not deliver a buffer after restart".into())
                }
                Err(reopen_error) => error = reopen_error,
            }
            self.set_error(is_mic, Some(&error.0));
        }
        None
    }

    fn reopen(&self, is_mic: bool) -> Result<(), AudioError> {
        let _guard = self.source_locks[usize::from(is_mic)]
            .lock()
            .expect("source lock");
        if self.stop.is_set() {
            return Ok(());
        }
        let source = self.source(is_mic);
        source.close()?;
        self.queue(is_mic).clear();
        if !is_mic {
            self.system_generation.fetch_add(1, Ordering::Relaxed);
        }
        source.open(self.sample_rate, 1)
    }

    fn wait_for_first_block(&self, is_mic: bool) -> Result<Option<Vec<f32>>, AudioError> {
        let source = self.source(is_mic);
        let wait = self
            .timing
            .dead_audio
            .clamp(Duration::from_millis(50), Duration::from_secs(3));
        let deadline = Instant::now() + wait;
        while !self.stop.is_set() && Instant::now() < deadline {
            if let Some(block) = source
                .read(Duration::from_millis(50))?
                .filter(|b| !b.is_empty())
            {
                return Ok(Some(block));
            }
            if source.exhausted() {
                return Err(AudioError("capture stream ended during restart".into()));
            }
            self.stop.wait(Duration::from_millis(5));
        }
        Ok(None)
    }

    fn close_sources(&self) -> Vec<AudioError> {
        let mut errors = Vec::new();
        for is_mic in [true, false] {
            let _guard = self.source_locks[usize::from(is_mic)]
                .lock()
                .expect("source lock");
            if let Err(error) = self.source(is_mic).close() {
                warn!(target: "ltn.audio.mixed", "failed to close {} audio source: {error}", label(is_mic));
                errors.push(error);
            }
        }
        errors
    }
}

impl AudioSource for MixedAudioSource {
    fn open(&self, sample_rate: u32, channels: u16) -> Result<(), AudioError> {
        if !self.readers.lock().expect("readers lock").is_empty() {
            return Err(AudioError("Mixed audio source is already open".into()));
        }
        let inner = &self.inner;
        if sample_rate != inner.sample_rate || channels != 1 {
            return Err(AudioError("MixedAudioSource requires 16000 Hz mono".into()));
        }
        {
            let mut status = inner.status.lock().expect("status lock");
            *status = Status {
                last_delivery: [Some(Instant::now()); 2],
                ..Status::default()
            };
        }
        inner.mic_queue.reset();
        inner.system_queue.reset();
        self.mix.lock().expect("mix lock").dropped_system_frames = 0;
        self.drain_buffers();
        // A source that fails to open is retried like a lost one. The other source
        // records from the first second; the recording must not wait for both.
        let mic_error = inner.open_source(true);
        let system_error = inner.open_source(false);
        inner.stop.clear();
        inner.running.store(true, Ordering::Relaxed);
        let mut readers = self.readers.lock().expect("readers lock");
        for (is_mic, open_error) in [(true, mic_error), (false, system_error)] {
            let worker = inner.clone();
            let name = format!("ltn-audio-{}", label(is_mic));
            let handle = thread::Builder::new()
                .name(name)
                .spawn(move || worker.reader(is_mic, open_error));
            match handle {
                Ok(handle) => readers.push(Reader { handle }),
                Err(error) => {
                    drop(readers);
                    inner.stop.set();
                    inner.close_sources();
                    self.join_readers();
                    return Err(AudioError(format!(
                        "could not start an audio reader: {error}"
                    )));
                }
            }
        }
        Ok(())
    }

    fn read(&self, timeout: Duration) -> Result<Option<Vec<f32>>, AudioError> {
        let mut mix = self.mix.lock().expect("mix lock");
        let mic = self.take_microphone(timeout, !mix.system_buffer.is_empty());
        self.take_system(&mut mix);
        let Some(mic) = mic else {
            *self.inner.last_mic_rms.lock().expect("rms lock") = 0.0;
            if mix.system_buffer.is_empty() {
                return Ok(None);
            }
            return Ok(Some(limit_peak(std::mem::take(&mut mix.system_buffer))));
        };
        *self.inner.last_mic_rms.lock().expect("rms lock") = rms(&mic);
        // Exactly the mic's length of system audio, zero-padded, so both cover the same time.
        let system = if mix.system_buffer.is_empty() {
            None
        } else {
            let take = mic.len().min(mix.system_buffer.len());
            let mut window: Vec<f32> = mix.system_buffer.drain(..take).collect();
            window.resize(mic.len(), 0.0);
            Some(window)
        };
        let silence = vec![0.0; mic.len()];
        let mut mic = mix.boost.apply(&mic, system.as_deref().unwrap_or(&silence));
        // The readers keep running while muted, so the stream state survives.
        if self.inner.mic_muted.load(Ordering::Relaxed) {
            mic.iter_mut().for_each(|s| *s *= 0.0);
        }
        Ok(Some(match system {
            None => limit_peak(mic),
            Some(system) => limit_peak(mic.iter().zip(&system).map(|(m, s)| m + s).collect()),
        }))
    }

    fn drain(&self) {
        self.drain_buffers();
        for is_mic in [true, false] {
            let _guard = self.inner.source_locks[usize::from(is_mic)]
                .lock()
                .expect("source lock");
            self.inner.source(is_mic).drain();
        }
    }

    fn close(&self) -> Result<(), AudioError> {
        self.inner.stop.set();
        self.inner.ready.set();
        let mut errors = self.inner.close_sources();
        errors.extend(self.join_readers());
        self.inner.mic_queue.clear();
        self.inner.system_queue.clear();
        self.mix.lock().expect("mix lock").system_buffer.clear();
        errors.into_iter().next().map_or(Ok(()), Err)
    }

    fn set_mic_muted(&self, muted: bool) {
        self.inner.mic_muted.store(muted, Ordering::Relaxed);
    }

    fn system_signal_seen(&self) -> bool {
        self.inner.system.system_signal_seen()
    }

    fn awaiting_permission(&self) -> bool {
        self.inner.system.awaiting_permission()
    }

    fn device_notice(&self) -> String {
        self.inner.mic.device_notice()
    }

    fn health(&self) -> Option<Health> {
        let inner = &self.inner;
        let status = inner.status.lock().expect("status lock");
        let mic_lost = status.has_error(true) || inner.stalled(&status, true);
        let system_lost =
            status.has_error(false) || status.deaf_system || inner.stalled(&status, false);
        let mic = if mic_lost {
            SourceHealth::Lost
        } else if status.silent_mic {
            SourceHealth::Silent
        } else {
            SourceHealth::Ok
        };
        Some(Health {
            mic,
            system: if system_lost {
                SourceHealth::Lost
            } else {
                SourceHealth::Ok
            },
        })
    }

    /// Current delivery failures, including recovery in progress.
    fn source_errors(&self) -> Vec<String> {
        self.inner
            .status
            .lock()
            .expect("status lock")
            .errors
            .iter()
            .map(|(_, message)| message.clone())
            .collect()
    }

    /// Whether a silent tap is proven with the test tone, so silence alone says nothing.
    fn verifies_system_audio(&self) -> bool {
        self.inner.test_tone.is_some() && self.inner.system.continuous_capture()
    }

    fn queue_sizes(&self) -> Option<QueueSizes> {
        Some(QueueSizes {
            mic_blocks: self.inner.mic_queue.len(),
            system_blocks: self.inner.system_queue.len(),
            system_buffered: self.mix.lock().expect("mix lock").system_buffer.len(),
        })
    }

    fn dropped_frames(&self) -> u64 {
        let mixed = self.mix.lock().expect("mix lock").dropped_system_frames;
        self.inner.mic_queue.dropped_frames() + self.inner.system_queue.dropped_frames() + mixed
    }
}

impl MixedAudioSource {
    fn drain_buffers(&self) {
        self.inner.mic_queue.clear();
        self.inner.system_queue.clear();
        let mut mix = self.mix.lock().expect("mix lock");
        mix.system_buffer.clear();
        mix.boost.reset();
        *self.inner.last_mic_rms.lock().expect("rms lock") = 0.0;
    }
}

impl Drop for MixedAudioSource {
    fn drop(&mut self) {
        self.inner.stop.set();
        self.inner.ready.set();
    }
}
