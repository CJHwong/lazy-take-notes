//! Ported from tests/l4_frameworks_and_drivers/workers/test_audio_worker.py.
//! The WAV writer's disk faults are tested in adapters/audio/wav_tests.rs.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use lazy_take_notes::adapters::audio::session_folder::SessionFolder;
use lazy_take_notes::app::events::{AudioHealthReport, Cancel, Event, Poster, WorkerState};
use lazy_take_notes::app::workers::audio::{AudioWorkerConfig, LoopTiming, run_audio_worker};
use lazy_take_notes::domain::transcript::TranscriptSegment;
use lazy_take_notes::ports::audio_source::{AudioError, AudioSource, Health, SourceHealth};
use lazy_take_notes::ports::transcriber::{TranscribeError, Transcriber};
use lazy_take_notes::usecases::transcribe_audio::{SpeechGate, TranscribeSettings};

/// What the fake source does on each read.
enum Step {
    Data(Vec<f32>),
    Nothing,
    Fail(&'static str),
    Pause(bool),
    Health(Option<Health>, Vec<&'static str>),
}

#[derive(Default)]
struct Flags {
    open_error: Option<&'static str>,
    exhausted: bool,
    awaiting_permission: bool,
    system_signal_seen: bool,
    verifies: bool,
    notice: &'static str,
}

/// Plays a script of reads, then cancels the worker.
struct ScriptSource {
    steps: Mutex<VecDeque<Step>>,
    flags: Flags,
    health: Mutex<(Option<Health>, Vec<String>)>,
    cancel: Cancel,
    paused: Arc<AtomicBool>,
    opened: AtomicUsize,
    closed: AtomicUsize,
}

impl ScriptSource {
    fn new(steps: Vec<Step>, flags: Flags, cancel: &Cancel, paused: &Arc<AtomicBool>) -> Self {
        Self {
            steps: Mutex::new(steps.into()),
            flags,
            health: Mutex::default(),
            cancel: cancel.clone(),
            paused: paused.clone(),
            opened: AtomicUsize::new(0),
            closed: AtomicUsize::new(0),
        }
    }
}

impl AudioSource for ScriptSource {
    fn open(&self, _rate: u32, _channels: u16) -> Result<(), AudioError> {
        self.opened.fetch_add(1, Ordering::SeqCst);
        match self.flags.open_error {
            Some(message) => Err(AudioError(message.into())),
            None => Ok(()),
        }
    }

    fn read(&self, _timeout: Duration) -> Result<Option<Vec<f32>>, AudioError> {
        let step = self.steps.lock().unwrap().pop_front();
        match step {
            None => {
                self.cancel.cancel();
                Ok(None)
            }
            Some(Step::Data(data)) => Ok(Some(data)),
            Some(Step::Nothing) => Ok(None),
            Some(Step::Fail(message)) => Err(AudioError(message.into())),
            Some(Step::Pause(on)) => {
                self.paused.store(on, Ordering::SeqCst);
                Ok(None)
            }
            Some(Step::Health(health, errors)) => {
                *self.health.lock().unwrap() =
                    (health, errors.into_iter().map(String::from).collect());
                Ok(None)
            }
        }
    }

    /// A paused loop drains instead of reading, so a drain plays the next pause step.
    fn drain(&self) {
        let mut steps = self.steps.lock().unwrap();
        match steps.front() {
            Some(Step::Pause(on)) => {
                self.paused.store(*on, Ordering::SeqCst);
                steps.pop_front();
            }
            Some(Step::Nothing) => {
                steps.pop_front();
            }
            None => self.cancel.cancel(),
            _ => {}
        }
    }

    fn close(&self) -> Result<(), AudioError> {
        self.closed.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    fn exhausted(&self) -> bool {
        self.flags.exhausted && self.steps.lock().unwrap().is_empty()
    }

    fn awaiting_permission(&self) -> bool {
        self.flags.awaiting_permission
    }

    fn system_signal_seen(&self) -> bool {
        self.flags.system_signal_seen
    }

    fn verifies_system_audio(&self) -> bool {
        self.flags.verifies
    }

    fn device_notice(&self) -> String {
        self.flags.notice.into()
    }

    fn health(&self) -> Option<Health> {
        self.health.lock().unwrap().0
    }

    fn source_errors(&self) -> Vec<String> {
        self.health.lock().unwrap().1.clone()
    }
}

/// Returns one new line per chunk. Can fail to load, fail each chunk, or run slowly.
#[derive(Default, Clone)]
struct Whisper {
    fail_load: bool,
    fail_chunks: bool,
    delay_ms: u64,
    /// The sample count of each call, shared with the test.
    calls: Arc<Mutex<Vec<usize>>>,
}

impl Transcriber for Whisper {
    fn load_model(&mut self, _path: &str) -> Result<(), TranscribeError> {
        if self.fail_load {
            return Err(TranscribeError("no such model".into()));
        }
        Ok(())
    }

    fn transcribe(
        &mut self,
        audio: &[f32],
        _language: &str,
        _hints: &[String],
    ) -> Result<Vec<TranscriptSegment>, TranscribeError> {
        std::thread::sleep(Duration::from_millis(self.delay_ms));
        let mut calls = self.calls.lock().unwrap();
        calls.push(audio.len());
        if self.fail_chunks {
            return Err(TranscribeError("whisper crashed".into()));
        }
        Ok(vec![TranscriptSegment::new(
            format!("line {}", calls.len()),
            0.0,
            0.5,
        )])
    }
}

/// One second of steady sound: a whole chunk at a 1 s chunk length.
fn second() -> Step {
    Step::Data(vec![0.1; 16_000])
}

struct Run {
    events: Vec<Event>,
    segments: Vec<TranscriptSegment>,
    source: Arc<ScriptSource>,
    calls: Arc<Mutex<Vec<usize>>>,
}

impl Run {
    fn states(&self) -> Vec<(WorkerState, String)> {
        self.events
            .iter()
            .filter_map(|event| match event {
                Event::WorkerStatus { state, error } => Some((*state, error.clone())),
                _ => None,
            })
            .collect()
    }

    fn texts(&self) -> Vec<String> {
        self.events
            .iter()
            .filter_map(|event| match event {
                Event::TranscriptChunk(segments) => {
                    Some(segments.iter().map(|s| s.text.clone()).collect::<Vec<_>>())
                }
                _ => None,
            })
            .flatten()
            .collect()
    }

    fn health(&self) -> Vec<AudioHealthReport> {
        self.events
            .iter()
            .filter_map(|event| match event {
                Event::AudioHealth(report) => Some(report.clone()),
                _ => None,
            })
            .collect()
    }

    fn warnings(&self) -> Vec<String> {
        self.states()
            .into_iter()
            .filter(|(state, _)| *state == WorkerState::Warning)
            .map(|(_, error)| error)
            .collect()
    }
}

struct Options {
    whisper: Whisper,
    flags: Flags,
    folder: Option<Arc<SessionFolder>>,
    save_audio: bool,
    timing: LoopTiming,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            whisper: Whisper::default(),
            flags: Flags {
                system_signal_seen: true,
                ..Flags::default()
            },
            folder: None,
            save_audio: true,
            timing: LoopTiming::default(),
        }
    }
}

fn run(steps: Vec<Step>, options: Options) -> Run {
    let cancel = Cancel::default();
    let paused = Arc::new(AtomicBool::new(false));
    let source = Arc::new(ScriptSource::new(steps, options.flags, &cancel, &paused));
    let (sender, mut receiver) = tokio::sync::mpsc::unbounded_channel();
    let calls = options.whisper.calls.clone();
    let config = AudioWorkerConfig {
        model_path: "model.bin".into(),
        language: "en".into(),
        settings: TranscribeSettings {
            chunk_duration: 1.0,
            overlap: 0.0,
            silence_threshold: 0.01,
            pause_duration: 0.5,
            gate: SpeechGate::LevelOnly,
        },
        recognition_hints: Vec::new(),
        folder: options.folder,
        save_audio: options.save_audio,
        timing: options.timing,
    };
    let segments = run_audio_worker(
        &Poster::new(sender),
        &cancel,
        &paused,
        config,
        Box::new(options.whisper),
        source.clone(),
    );
    let mut events = Vec::new();
    while let Ok(event) = receiver.try_recv() {
        events.push(event);
    }
    Run {
        events,
        segments,
        source,
        calls,
    }
}

use WorkerState::{Error, LoadingModel, ModelReady, Recording, Stopped};

#[test]
fn an_immediate_cancel_opens_and_closes_the_source() {
    let result = run(vec![], Options::default());
    let states: Vec<WorkerState> = result.states().into_iter().map(|(s, _)| s).collect();
    assert_eq!(states, [LoadingModel, ModelReady, Recording, Stopped]);
    assert_eq!(result.source.opened.load(Ordering::SeqCst), 1);
    assert_eq!(result.source.closed.load(Ordering::SeqCst), 1);
}

#[test]
fn chunks_are_transcribed_and_the_rest_flushed_at_the_end() {
    let options = Options {
        whisper: Whisper {
            delay_ms: 50,
            ..Whisper::default()
        },
        ..Options::default()
    };
    // The first chunk is in flight while two more seconds arrive: the final flush sends them.
    let steps = vec![second(), second(), second()];
    let result = run(steps, options);
    assert_eq!(result.texts(), ["line 1", "line 2"]);
    assert_eq!(*result.calls.lock().unwrap(), [16_000, 32_000]);
    assert_eq!(result.segments.len(), 2);
    let active: Vec<bool> = result
        .events
        .iter()
        .filter_map(|event| match event {
            Event::TranscriptionStatus { active } => Some(*active),
            _ => None,
        })
        .collect();
    assert_eq!(active, [true, false, true, false]);
}

#[test]
fn a_chunk_in_flight_at_shutdown_is_collected() {
    let options = Options {
        whisper: Whisper {
            delay_ms: 300,
            ..Whisper::default()
        },
        ..Options::default()
    };
    let result = run(vec![second()], options);
    assert_eq!(result.texts(), ["line 1"]);
}

#[test]
fn a_pause_flushes_the_buffer_and_drops_the_audio_after_it() {
    let steps = vec![
        Step::Data(vec![0.1; 8_000]),
        Step::Pause(true),
        Step::Nothing,
        Step::Pause(false),
    ];
    let result = run(steps, Options::default());
    assert_eq!(result.texts(), ["line 1"]);
    assert_eq!(*result.calls.lock().unwrap(), [8_000]);
}

#[test]
fn a_pause_before_any_audio_transcribes_nothing() {
    let result = run(
        vec![Step::Pause(true), Step::Pause(false)],
        Options::default(),
    );
    assert!(result.texts().is_empty());
    assert!(result.calls.lock().unwrap().is_empty());
}

#[test]
fn a_source_that_fails_to_open_reports_and_stops() {
    let options = Options {
        flags: Flags {
            open_error: Some("device busy"),
            ..Flags::default()
        },
        ..Options::default()
    };
    let result = run(vec![], options);
    assert_eq!(
        result.states()[3..],
        [(Error, "device busy".into()), (Stopped, String::new())]
    );
    assert_eq!(result.source.closed.load(Ordering::SeqCst), 0);
}

#[test]
fn a_failed_transcription_posts_an_error_and_recording_goes_on() {
    let options = Options {
        whisper: Whisper {
            fail_chunks: true,
            ..Whisper::default()
        },
        ..Options::default()
    };
    let result = run(vec![second(), second(), second()], options);
    let errors: Vec<String> = result
        .states()
        .into_iter()
        .filter(|(state, _)| *state == Error)
        .map(|(_, error)| error)
        .collect();
    assert_eq!(errors, ["whisper crashed", "whisper crashed"]);
    assert_eq!(result.states().last().unwrap().0, Stopped);
}

#[test]
fn an_exhausted_source_ends_the_loop_with_an_error() {
    let options = Options {
        flags: Flags {
            exhausted: true,
            system_signal_seen: true,
            ..Flags::default()
        },
        ..Options::default()
    };
    let result = run(vec![Step::Nothing], options);
    assert!(
        result
            .states()
            .contains(&(Error, "Audio stream stopped unexpectedly after 0s".into()))
    );
}

#[test]
fn nan_audio_posts_a_zero_level() {
    let result = run(vec![Step::Data(vec![f32::NAN; 512])], Options::default());
    assert!(result.events.contains(&Event::AudioLevel { rms: 0.0 }));
}

#[test]
fn a_failing_read_warns_once_and_the_recording_goes_on() {
    let steps = vec![Step::Fail("bad chunk"), Step::Fail("bad chunk"), second()];
    let result = run(steps, Options::default());
    assert_eq!(
        result.warnings(),
        ["Audio processing error: bad chunk. Recording continues."]
    );
    assert_eq!(result.texts(), ["line 1"]);
}

#[test]
fn a_loop_that_keeps_failing_raises_an_alert_and_clears_it() {
    let options = Options {
        timing: LoopTiming {
            loop_alert: Duration::from_millis(100),
            ..LoopTiming::default()
        },
        ..Options::default()
    };
    let steps = vec![
        Step::Fail("bad chunk"),
        Step::Fail("bad chunk"),
        Step::Fail("bad chunk"),
        Step::Fail("bad chunk"),
        Step::Fail("bad chunk"),
        Step::Nothing,
    ];
    let result = run(steps, options);
    let processing: Vec<String> = result.health().into_iter().map(|h| h.processing).collect();
    // Five failures 50 ms apart pass the 100 ms alert; the next good read clears it.
    assert_eq!(processing, ["error", "ok"]);
    assert_eq!(result.health()[0].detail, "bad chunk");
}

#[test]
fn each_health_change_is_posted_once_with_sorted_errors() {
    let lost = Health {
        mic: SourceHealth::Lost,
        system: SourceHealth::Ok,
    };
    let ok = Health {
        mic: SourceHealth::Ok,
        system: SourceHealth::Ok,
    };
    let steps = vec![
        Step::Health(Some(ok), vec![]),
        Step::Nothing,
        Step::Health(Some(lost), vec!["mic: b", "mic: a"]),
        Step::Nothing,
        Step::Health(Some(ok), vec![]),
    ];
    let options = Options {
        flags: Flags {
            notice: "Microphone 'X' is unavailable; using Y",
            system_signal_seen: true,
            ..Flags::default()
        },
        ..Options::default()
    };
    let result = run(steps, options);
    let seen: Vec<(String, String, String)> = result
        .health()
        .into_iter()
        .map(|h| (h.mic, h.detail, h.notice))
        .collect();
    let notice = "Microphone 'X' is unavailable; using Y".to_string();
    assert_eq!(
        seen,
        [
            ("ok".into(), String::new(), notice.clone()),
            ("lost".into(), "mic: a; mic: b".into(), notice.clone()),
            ("ok".into(), String::new(), notice),
        ]
    );
    // A source with health reports through it, not through warnings.
    assert!(result.warnings().is_empty());
}

#[test]
fn a_source_without_health_posts_its_errors_as_warnings() {
    let result = run(
        vec![
            Step::Health(None, vec!["system: tap exited"]),
            Step::Nothing,
        ],
        Options::default(),
    );
    assert_eq!(result.warnings(), ["system: tap exited"]);
}

fn no_signal_warnings(flags: Flags) -> Vec<String> {
    let options = Options {
        flags,
        timing: LoopTiming {
            no_signal_warn: Duration::ZERO,
            ..LoopTiming::default()
        },
        ..Options::default()
    };
    run(
        vec![Step::Data(vec![0.0; 512]), Step::Data(vec![0.0; 512])],
        options,
    )
    .warnings()
}

#[test]
fn the_permission_hint_shows_once_without_system_sound() {
    let warnings = no_signal_warnings(Flags::default());
    assert_eq!(warnings.len(), 1);
    assert!(warnings[0].starts_with("No system audio received in the first 0s."));
}

#[test]
fn no_permission_hint_when_it_is_not_due() {
    for flags in [
        Flags {
            system_signal_seen: true,
            ..Flags::default()
        },
        Flags {
            awaiting_permission: true,
            ..Flags::default()
        },
        Flags {
            verifies: true,
            ..Flags::default()
        },
    ] {
        assert!(no_signal_warnings(flags).is_empty());
    }
}

fn recorded(folder: &tempfile::TempDir) -> bool {
    folder.path().join("recording.wav").exists()
}

fn with_folder(folder: &tempfile::TempDir, save_audio: bool, whisper: Whisper) -> Options {
    Options {
        whisper,
        folder: Some(Arc::new(SessionFolder::new(folder.path()))),
        save_audio,
        ..Options::default()
    }
}

#[test]
fn a_clean_session_with_save_audio_off_deletes_the_audio_quietly() {
    let folder = tempfile::tempdir().unwrap();
    let result = run(
        vec![second()],
        with_folder(&folder, false, Whisper::default()),
    );
    assert!(!recorded(&folder));
    assert!(result.warnings().is_empty());
}

#[test]
fn save_audio_on_keeps_the_audio_without_a_notice() {
    let folder = tempfile::tempdir().unwrap();
    let result = run(
        vec![second()],
        with_folder(&folder, true, Whisper::default()),
    );
    assert!(recorded(&folder));
    assert!(result.warnings().is_empty());
}

#[test]
fn a_model_that_fails_to_load_still_records_and_keeps_the_audio() {
    let folder = tempfile::tempdir().unwrap();
    let whisper = Whisper {
        fail_load: true,
        ..Whisper::default()
    };
    let result = run(vec![second()], with_folder(&folder, false, whisper));
    assert!(recorded(&folder));
    assert_eq!(
        result.warnings(),
        [
            "Speech model failed to load: no such model. Recording audio only; run transcribe on the recording later.",
            "save_audio is off, but the speech model failed to load. The audio of this meeting is kept in recording.wav, so you can transcribe it later.",
        ]
    );
    assert!(result.texts().is_empty());
    assert!(result.calls.lock().unwrap().is_empty());
}

#[test]
fn a_failed_transcription_keeps_the_audio() {
    let folder = tempfile::tempdir().unwrap();
    let whisper = Whisper {
        fail_chunks: true,
        ..Whisper::default()
    };
    run(vec![second()], with_folder(&folder, false, whisper));
    assert!(recorded(&folder));
}
