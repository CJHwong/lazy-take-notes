//! Ported from tests/l4_frameworks_and_drivers/workers/test_file_transcription_worker.py.

mod common;

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use lazy_take_notes::app::events::{Cancel, Event, Poster, WorkerState};
use lazy_take_notes::app::workers::file_transcription::{FileJob, run_file_transcription};
use lazy_take_notes::domain::errors::ModelResolutionError;
use lazy_take_notes::domain::transcript::TranscriptSegment;
use lazy_take_notes::ports::model_resolver::ModelResolver;
use lazy_take_notes::ports::transcriber::{TranscribeError, Transcriber};
use lazy_take_notes::usecases::transcribe_audio::{SpeechGate, TranscribeSettings};

use WorkerState::{Error, LoadingModel, ModelReady, Recording, Stopped};

/// A 16 kHz mono 16-bit WAV of steady sound, which ffmpeg decodes.
fn wav(dir: &Path, seconds: usize) -> PathBuf {
    let samples = seconds * 16_000;
    let mut bytes = Vec::new();
    bytes.extend(b"RIFF");
    bytes.extend(&(36 + samples as u32 * 2).to_le_bytes());
    bytes.extend(b"WAVEfmt ");
    bytes.extend(&16u32.to_le_bytes());
    for value in [1u16, 1] {
        bytes.extend(&value.to_le_bytes());
    }
    bytes.extend(&16_000u32.to_le_bytes());
    bytes.extend(&32_000u32.to_le_bytes());
    for value in [2u16, 16] {
        bytes.extend(&value.to_le_bytes());
    }
    bytes.extend(b"data");
    bytes.extend(&(samples as u32 * 2).to_le_bytes());
    for _ in 0..samples {
        bytes.extend(&3_277i16.to_le_bytes()); // 0.1
    }
    let path = dir.join("talk.wav");
    std::fs::write(&path, bytes).unwrap();
    path
}

struct Resolver(Result<String, String>);

impl ModelResolver for Resolver {
    fn resolve(&self, _name: &str) -> Result<String, ModelResolutionError> {
        self.0.clone().map_err(ModelResolutionError)
    }
}

/// One new line per chunk. Fails to load, or fails from call `fail_from` on.
#[derive(Default)]
struct Whisper {
    fail_load: bool,
    fail_from: Option<usize>,
    calls: usize,
    closed: std::sync::Arc<Mutex<bool>>,
}

impl Transcriber for Whisper {
    fn load_model(&mut self, _path: &str) -> Result<(), TranscribeError> {
        if self.fail_load {
            return Err(TranscribeError("bad model file".into()));
        }
        Ok(())
    }

    fn transcribe(
        &mut self,
        _audio: &[f32],
        _language: &str,
        _hints: &[String],
    ) -> Result<Vec<TranscriptSegment>, TranscribeError> {
        self.calls += 1;
        if self.fail_from.is_some_and(|from| self.calls >= from) {
            return Err(TranscribeError("whisper crashed".into()));
        }
        Ok(vec![TranscriptSegment::new(
            format!("line {}", self.calls),
            0.0,
            0.5,
        )])
    }

    fn close(&mut self) {
        *self.closed.lock().unwrap() = true;
    }
}

struct Outcome {
    segments: Vec<TranscriptSegment>,
    states: Vec<(WorkerState, String)>,
    texts: Vec<String>,
}

fn run(path: &Path, resolver: Resolver, whisper: Whisper, cancel: Cancel) -> Outcome {
    let (sender, mut receiver) = tokio::sync::mpsc::unbounded_channel();
    let job = FileJob {
        audio_path: path,
        model_name: "breeze-q8",
        language: "en",
        settings: TranscribeSettings {
            chunk_duration: 1.0,
            overlap: 0.0,
            gate: SpeechGate::LevelOnly,
            ..TranscribeSettings::default()
        },
        recognition_hints: Vec::new(),
    };
    let segments = run_file_transcription(
        &Poster::new(sender),
        &cancel,
        job,
        &resolver,
        Box::new(whisper),
    );
    let (mut states, mut texts) = (Vec::new(), Vec::new());
    while let Ok(event) = receiver.try_recv() {
        match event {
            Event::WorkerStatus { state, error } => states.push((state, error)),
            Event::TranscriptChunk(lines) => texts.extend(lines.into_iter().map(|l| l.text)),
            _ => {}
        }
    }
    Outcome {
        segments,
        states,
        texts,
    }
}

fn ok() -> Resolver {
    Resolver(Ok("/models/m.bin".into()))
}

fn names(states: &[(WorkerState, String)]) -> Vec<WorkerState> {
    states.iter().map(|(state, _)| *state).collect()
}

#[test]
fn a_file_is_transcribed_chunk_by_chunk() {
    let dir = tempfile::tempdir().unwrap();
    let outcome = run(
        &wav(dir.path(), 3),
        ok(),
        Whisper::default(),
        Cancel::default(),
    );
    assert_eq!(outcome.texts, ["line 1", "line 2", "line 3"]);
    assert_eq!(outcome.segments.len(), 3);
    assert_eq!(
        names(&outcome.states),
        [LoadingModel, ModelReady, Recording, Stopped]
    );
}

#[test]
fn a_missing_file_posts_an_error() {
    let outcome = run(
        Path::new("/nowhere/a.wav"),
        ok(),
        Whisper::default(),
        Cancel::default(),
    );
    assert_eq!(
        outcome.states,
        [(Error, "Audio file not found: /nowhere/a.wav".into())]
    );
}

#[test]
fn a_model_that_does_not_resolve_posts_an_error() {
    let dir = tempfile::tempdir().unwrap();
    let resolver = Resolver(Err("Unknown model 'x'".into()));
    let outcome = run(
        &wav(dir.path(), 1),
        resolver,
        Whisper::default(),
        Cancel::default(),
    );
    assert_eq!(
        outcome.states,
        [
            (LoadingModel, String::new()),
            (Error, "Unknown model 'x'".into())
        ]
    );
}

#[test]
fn a_model_that_fails_to_load_posts_an_error_and_closes() {
    let dir = tempfile::tempdir().unwrap();
    let whisper = Whisper {
        fail_load: true,
        ..Whisper::default()
    };
    let closed = whisper.closed.clone();
    let outcome = run(&wav(dir.path(), 1), ok(), whisper, Cancel::default());
    assert_eq!(
        outcome.states.last().unwrap(),
        &(Error, "bad model file".into())
    );
    assert!(*closed.lock().unwrap());
}

#[test]
fn a_cancelled_run_transcribes_nothing_more() {
    let dir = tempfile::tempdir().unwrap();
    let cancel = Cancel::default();
    cancel.cancel();
    let outcome = run(&wav(dir.path(), 3), ok(), Whisper::default(), cancel);
    assert!(outcome.texts.is_empty());
    assert_eq!(
        names(&outcome.states),
        [LoadingModel, ModelReady, Recording, Stopped]
    );
}

#[test]
fn a_failure_mid_file_keeps_the_earlier_lines() {
    let dir = tempfile::tempdir().unwrap();
    let whisper = Whisper {
        fail_from: Some(2),
        ..Whisper::default()
    };
    let closed = whisper.closed.clone();
    let outcome = run(&wav(dir.path(), 3), ok(), whisper, Cancel::default());
    assert_eq!(outcome.texts, ["line 1"]);
    assert_eq!(outcome.segments.len(), 1);
    assert_eq!(
        outcome.states.last().unwrap(),
        &(Error, "whisper crashed".into())
    );
    assert!(*closed.lock().unwrap());
}
