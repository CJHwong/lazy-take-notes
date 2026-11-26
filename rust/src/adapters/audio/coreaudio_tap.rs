//! System audio from the native Core Audio process tap (`coreaudio-tap`, a Swift helper).
//!
//! The helper writes raw float32 16 kHz mono to stdout and diagnostics to
//! stderr. Its audio, its diagnostics and its lifetime are tracked apart.

use std::collections::VecDeque;
use std::io::{BufRead, BufReader, Read};
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use tracing::{debug, error, info, warn};

use super::buffer::AudioBuffer;
use crate::ports::audio_source::{AudioError, AudioSource};

const SAMPLE_RATE: u32 = 16_000;
const BYTES_PER_SAMPLE: usize = 4;
const CHUNK_FRAMES: usize = SAMPLE_RATE as usize / 10;
const TAIL_LINES: usize = 20;

/// The helper next to the executable, where a release puts it. A debug build
/// falls back to the copy in this repo; a release build carries no local path.
pub fn tap_binary() -> PathBuf {
    let beside = std::env::current_exe()
        .ok()
        .and_then(|exe| helper_beside(&exe))
        .unwrap_or_else(|| PathBuf::from("coreaudio-tap"));
    #[cfg(debug_assertions)]
    if !beside.exists() {
        return PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../src/lazy_take_notes/_native/bin/coreaudio-tap");
    }
    beside
}

/// The helper beside the real executable. A link to it (Homebrew puts one on
/// PATH) names the link's own folder, so the link is resolved first.
fn helper_beside(exe: &Path) -> Option<PathBuf> {
    let real = std::fs::canonicalize(exe).unwrap_or_else(|_| exe.to_path_buf());
    real.parent().map(|dir| dir.join("coreaudio-tap"))
}

/// State of one helper run. A new open makes a new one, so a late reader of
/// the old run can never write into the new run's queue.
struct Run {
    queue: AudioBuffer,
    stop: AtomicBool,
    exhausted: AtomicBool,
    stderr_done: AtomicBool,
    signal_seen: AtomicBool,
    awaiting_permission: AtomicBool,
    error: Mutex<Option<String>>,
    tail: Mutex<VecDeque<String>>,
}

impl Run {
    fn new() -> Self {
        Self {
            queue: AudioBuffer::new(SAMPLE_RATE as usize, SAMPLE_RATE, "system tap"),
            stop: AtomicBool::new(false),
            exhausted: AtomicBool::new(false),
            stderr_done: AtomicBool::new(false),
            signal_seen: AtomicBool::new(false),
            awaiting_permission: AtomicBool::new(false),
            error: Mutex::new(None),
            tail: Mutex::new(VecDeque::new()),
        }
    }

    fn stopped(&self) -> bool {
        self.stop.load(Ordering::SeqCst)
    }

    fn set_error(&self, message: String) {
        *self.error.lock().expect("tap error lock") = Some(message);
    }

    fn record_diagnostic(&self, raw: &[u8]) {
        let line = String::from_utf8_lossy(raw).trim_end().to_string();
        if !line.is_empty() {
            let mut tail = self.tail.lock().expect("tap tail lock");
            if tail.len() == TAIL_LINES {
                tail.pop_front();
            }
            tail.push_back(line.clone());
            drop(tail);
            info!(target: "ltn.audio.coreaudio", "coreaudio-tap: {line}");
        }
        if line.contains("waiting for permission") {
            self.awaiting_permission.store(true, Ordering::SeqCst);
        } else if line.contains("permission granted") || line.ends_with("capturing") {
            self.awaiting_permission.store(false, Ordering::SeqCst);
        }
    }

    fn enqueue(&self, raw: &[u8]) {
        if raw.is_empty() {
            return;
        }
        let samples: Vec<f32> = raw
            .as_chunks::<BYTES_PER_SAMPLE>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        if !self.signal_seen.load(Ordering::Relaxed) && samples.iter().any(|&s| s != 0.0) {
            self.signal_seen.store(true, Ordering::Relaxed);
        }
        self.queue.put(&samples);
    }
}

struct Helper {
    child: Arc<Mutex<Child>>,
    stdin: Option<ChildStdin>,
    reader: JoinHandle<()>,
    stderr_reader: JoinHandle<()>,
}

pub struct CoreAudioTapSource {
    binary: PathBuf,
    ensure_permission: bool,
    run: Mutex<Arc<Run>>,
    helper: Mutex<Option<Helper>>,
}

fn wait_with_timeout(child: &Mutex<Child>, timeout: Duration) -> Option<i32> {
    let deadline = Instant::now() + timeout;
    loop {
        match child.lock().expect("tap child lock").try_wait() {
            // A signal death has no code; Python reports it as the negative signal number.
            Ok(Some(status)) => {
                return Some(status.code().unwrap_or_else(|| {
                    -std::os::unix::process::ExitStatusExt::signal(&status).unwrap_or(0)
                }));
            }
            Ok(None) => {}
            Err(e) => {
                debug!(target: "ltn.audio.coreaudio", "coreaudio-tap wait failed: {e}");
                return None;
            }
        }
        if Instant::now() >= deadline {
            return None;
        }
        thread::sleep(Duration::from_millis(10));
    }
}

impl CoreAudioTapSource {
    pub fn new(ensure_permission: bool) -> Self {
        Self::with_binary(tap_binary(), ensure_permission)
    }

    pub fn with_binary(binary: PathBuf, ensure_permission: bool) -> Self {
        Self {
            binary,
            ensure_permission,
            run: Mutex::new(Arc::new(Run::new())),
            helper: Mutex::default(),
        }
    }

    fn run(&self) -> Arc<Run> {
        self.run.lock().expect("tap run lock").clone()
    }

    /// The pid of the running helper, for diagnostics and fault tests.
    pub fn pid(&self) -> Option<u32> {
        self.helper
            .lock()
            .expect("tap helper lock")
            .as_ref()
            .map(|h| h.child.lock().expect("tap child lock").id())
    }

    fn read_audio(run: &Run, mut stdout: impl Read, child: &Mutex<Child>) {
        let mut block = vec![0u8; CHUNK_FRAMES * BYTES_PER_SAMPLE];
        let mut remainder: Vec<u8> = Vec::new();
        let failure = loop {
            if run.stopped() {
                break None;
            }
            // Fill the whole block, as Python's buffered read does, so blocks stay 100 ms.
            let mut filled = 0;
            let mut ended = false;
            while filled < block.len() {
                match stdout.read(&mut block[filled..]) {
                    Ok(0) => {
                        ended = true;
                        break;
                    }
                    Ok(count) => filled += count,
                    Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
                    Err(e) => return Self::reader_failed(run, child, &e),
                }
            }
            remainder.extend_from_slice(&block[..filled]);
            let usable = remainder.len() - remainder.len() % BYTES_PER_SAMPLE;
            run.enqueue(&remainder[..usable]);
            remainder.drain(..usable);
            if ended {
                break (!remainder.is_empty())
                    .then(|| "coreaudio-tap ended with a partial float32 sample".to_string());
            }
        };
        if let Some(message) = failure
            && !run.stopped()
        {
            run.set_error(message);
        }
        if !run.stopped() {
            Self::record_exit(run, child);
        }
    }

    fn reader_failed(run: &Run, child: &Mutex<Child>, failure: &std::io::Error) {
        if run.stopped() {
            return;
        }
        run.set_error(format!("coreaudio-tap audio pipe failed: {failure}"));
        Self::record_exit(run, child);
    }

    fn record_exit(run: &Run, child: &Mutex<Child>) {
        let code = wait_with_timeout(child, Duration::from_millis(500));
        if code.is_none() {
            run.set_error("coreaudio-tap closed audio output while still running".into());
        }
        if let Some(code) = code.filter(|&c| c != 0) {
            let deadline = Instant::now() + Duration::from_millis(200);
            while !run.stderr_done.load(Ordering::SeqCst) && Instant::now() < deadline {
                thread::sleep(Duration::from_millis(5));
            }
            let diagnostic = run
                .tail
                .lock()
                .expect("tap tail lock")
                .iter()
                .cloned()
                .collect::<Vec<_>>()
                .join("\n");
            let message = format!("coreaudio-tap exited with code {code}");
            run.set_error(if diagnostic.is_empty() {
                message
            } else {
                format!("{message}: {diagnostic}")
            });
        }
        run.exhausted.store(true, Ordering::SeqCst);
        match run.error.lock().expect("tap error lock").as_deref() {
            Some(failure) => {
                warn!(target: "ltn.audio.coreaudio", "coreaudio-tap audio output ended: {failure}")
            }
            None => warn!(target: "ltn.audio.coreaudio", "coreaudio-tap audio output ended"),
        }
    }

    fn stop_process(child: &Mutex<Child>, stdin: Option<ChildStdin>) {
        drop(stdin);
        if wait_with_timeout(child, Duration::from_secs(1)).is_some() {
            return;
        }
        warn!(target: "ltn.audio.coreaudio", "coreaudio-tap did not stop after stdin closed; terminating");
        let pid = child.lock().expect("tap child lock").id();
        // SAFETY: kill(2) on our own child's pid; the child is not reaped yet, so the pid is still ours.
        unsafe { libc::kill(pid as libc::pid_t, libc::SIGTERM) };
        if wait_with_timeout(child, Duration::from_secs(1)).is_some() {
            return;
        }
        warn!(target: "ltn.audio.coreaudio", "coreaudio-tap ignored termination; killing");
        if let Err(e) = child.lock().expect("tap child lock").kill() {
            debug!(target: "ltn.audio.coreaudio", "coreaudio-tap exited before kill: {e}");
        }
        wait_with_timeout(child, Duration::from_secs(1));
    }

    fn join(reader: JoinHandle<()>) -> Result<(), AudioError> {
        let deadline = Instant::now() + Duration::from_secs(2);
        while !reader.is_finished() && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(5));
        }
        if !reader.is_finished() {
            return Err(AudioError(
                "coreaudio-tap reader did not stop after helper exit".into(),
            ));
        }
        let _ = reader.join();
        Ok(())
    }
}

impl AudioSource for CoreAudioTapSource {
    fn open(&self, sample_rate: u32, channels: u16) -> Result<(), AudioError> {
        if !cfg!(target_os = "macos") {
            return Err(AudioError("CoreAudioTapSource is macOS only".into()));
        }
        if sample_rate != SAMPLE_RATE || channels != 1 {
            return Err(AudioError(
                "CoreAudioTapSource requires 16000 Hz mono output".into(),
            ));
        }
        if !self.binary.exists() {
            return Err(AudioError(format!(
                "Native binary not found: {}\nRun: bash scripts/build_native.sh",
                self.binary.display()
            )));
        }
        self.close()?;
        let run = Arc::new(Run::new());
        *self.run.lock().expect("tap run lock") = run.clone();
        let mut command = Command::new(&self.binary);
        if self.ensure_permission || std::env::var("LTN_KIOSK").as_deref() == Ok("1") {
            command.arg("--ensure-permission");
        }
        command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        // SAFETY: setsid is async-signal-safe; this matches Python's start_new_session=True.
        unsafe {
            command.pre_exec(|| {
                libc::setsid();
                Ok(())
            });
        }
        let mut child = command.spawn().map_err(|e| AudioError(e.to_string()))?;
        info!(target: "ltn.audio.coreaudio", "coreaudio-tap started (pid={}, binary={})", child.id(), self.binary.display());
        let (stdin, stdout, stderr) = (
            child.stdin.take(),
            child.stdout.take().expect("piped"),
            child.stderr.take().expect("piped"),
        );
        let child = Arc::new(Mutex::new(child));
        let stderr_run = run.clone();
        let stderr_reader = thread::Builder::new().name("ltn-tap-stderr".into()).spawn(move || {
            let mut lines = BufReader::new(stderr);
            let mut line = Vec::new();
            loop {
                line.clear();
                match lines.read_until(b'\n', &mut line) {
                    Ok(0) => break,
                    Ok(_) => stderr_run.record_diagnostic(&line),
                    Err(e) => {
                        if !stderr_run.stopped() {
                            error!(target: "ltn.audio.coreaudio", "coreaudio-tap diagnostic pipe failed: {e}");
                        }
                        break;
                    }
                }
            }
            stderr_run.stderr_done.store(true, Ordering::SeqCst);
        });
        let (reader_run, reader_child) = (run.clone(), child.clone());
        let reader = thread::Builder::new()
            .name("ltn-tap-audio".into())
            .spawn(move || Self::read_audio(&reader_run, stdout, &reader_child));
        match (stderr_reader, reader) {
            (Ok(stderr_reader), Ok(reader)) => {
                *self.helper.lock().expect("tap helper lock") = Some(Helper {
                    child,
                    stdin,
                    reader,
                    stderr_reader,
                });
                Ok(())
            }
            (stderr_reader, reader) => {
                run.stop.store(true, Ordering::SeqCst);
                Self::stop_process(&child, stdin);
                let failure = stderr_reader
                    .err()
                    .or(reader.err())
                    .expect("one spawn failed");
                Err(AudioError(format!(
                    "could not start a coreaudio-tap reader: {failure}"
                )))
            }
        }
    }

    fn read(&self, timeout: Duration) -> Result<Option<Vec<f32>>, AudioError> {
        let run = self.run();
        if let Some(block) = run.queue.get(timeout) {
            return Ok(Some(block));
        }
        match run.error.lock().expect("tap error lock").clone() {
            Some(message) => Err(AudioError(message)),
            None => Ok(None),
        }
    }

    fn drain(&self) {
        self.run().queue.clear();
    }

    fn close(&self) -> Result<(), AudioError> {
        let run = self.run();
        run.stop.store(true, Ordering::SeqCst);
        let helper = self.helper.lock().expect("tap helper lock").take();
        let mut result = Ok(());
        if let Some(helper) = helper {
            Self::stop_process(&helper.child, helper.stdin);
            result = Self::join(helper.reader).and(Self::join(helper.stderr_reader));
        }
        run.queue.clear();
        result
    }

    fn exhausted(&self) -> bool {
        self.run().exhausted.load(Ordering::SeqCst)
    }

    fn continuous_capture(&self) -> bool {
        true
    }

    fn awaiting_permission(&self) -> bool {
        self.run().awaiting_permission.load(Ordering::SeqCst)
    }

    fn system_signal_seen(&self) -> bool {
        self.run().signal_seen.load(Ordering::Relaxed)
    }

    fn dropped_frames(&self) -> u64 {
        self.run().queue.dropped_frames()
    }
}

impl Drop for CoreAudioTapSource {
    fn drop(&mut self) {
        if let Err(e) = self.close() {
            warn!(target: "ltn.audio.coreaudio", "coreaudio-tap close on drop: {e}");
        }
    }
}

#[cfg(test)]
#[path = "coreaudio_tap_tests.rs"]
mod tests;
