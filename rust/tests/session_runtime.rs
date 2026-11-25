//! The session runtime without any screen: a fake front end collects the
//! notices, and each test hands the task results back by hand.

mod common;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use common::{FakeLlm, FakePersistence, Reply, built_in_template, segment};
use lazy_take_notes::adapters::audio::session_folder::SessionFolder;
use lazy_take_notes::adapters::session_controller::SessionController;
use lazy_take_notes::app::events::{Event, WorkerState};
use lazy_take_notes::app::session::file::FileRuntime;
use lazy_take_notes::app::session::record::{RecordRuntime, RecordWorkers};
use lazy_take_notes::app::session::{SessionIo, SessionRuntime, Severity, Step};
use lazy_take_notes::app::tasks::{Payload, Tasks};
use lazy_take_notes::config::build_app_config;
use lazy_take_notes::ports::audio_source::{AudioError, AudioSource};

const DIGEST: &str = "## Topic\nSome content";

/// A front end that only remembers: notices, task results and worker events.
struct FakeIo {
    runtime: tokio::runtime::Runtime,
    notices: Vec<(String, Severity)>,
    results: Arc<Mutex<Vec<Payload>>>,
    events: Arc<Mutex<Vec<Event>>>,
    exited: bool,
}

impl FakeIo {
    fn new() -> Self {
        Self {
            runtime: tokio::runtime::Builder::new_multi_thread()
                .worker_threads(2)
                .enable_all()
                .build()
                .unwrap(),
            notices: Vec::new(),
            results: Arc::default(),
            events: Arc::default(),
            exited: false,
        }
    }

    fn texts(&self) -> Vec<&str> {
        self.notices.iter().map(|(text, _)| text.as_str()).collect()
    }

    /// The next task result, waiting up to 5 s for it.
    fn next_result(&self) -> Payload {
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            if let Some(payload) = self.results.lock().unwrap().pop() {
                return payload;
            }
            assert!(Instant::now() < deadline, "no task result");
            std::thread::sleep(Duration::from_millis(5));
        }
    }
}

impl SessionIo for FakeIo {
    fn notify(&mut self, text: String, severity: Severity, _timeout: Duration) {
        self.notices.push((text, severity));
    }

    fn tasks(&self) -> Tasks {
        let (results, events) = (self.results.clone(), self.events.clone());
        Tasks::new(
            self.runtime.handle().clone(),
            Arc::new(move |payload| results.lock().unwrap().push(payload)),
            Arc::new(move |event| {
                events.lock().unwrap().push(event);
                true
            }),
        )
    }

    fn exit(&mut self) {
        self.exited = true;
    }
}

fn session(dir: &std::path::Path) -> SessionRuntime {
    let raw = yaml_serde::from_str("output: {auto_label: false}").unwrap();
    let controller = SessionController::new(
        build_app_config(&raw).unwrap(),
        built_in_template("default_en"),
        Arc::new(FakeLlm::new(Reply::Text(DIGEST.into(), 10))),
        Box::new(FakePersistence::new().0),
    );
    SessionRuntime::new(
        controller,
        dir.join("2026-11-19_120000"),
        "",
        vec![],
        vec![],
    )
}

#[test]
fn a_finished_file_gets_its_final_notes_then_q_closes() {
    let dir = tempfile::tempdir().unwrap();
    let mut io = FakeIo::new();
    let mut file = FileRuntime::new(session(dir.path()), Box::new(|_poster, _cancel| {}));
    file.mount(&mut io);
    let chunk = Event::TranscriptChunk(vec![segment("hello", 0.0, 1.0)]);
    assert_eq!(file.on_event(chunk, &mut io), Ok(Step::None));
    assert_eq!(
        file.session.status.digest_version, 0,
        "a file waits for its end"
    );

    let stopped = Event::status(WorkerState::Stopped);
    assert_eq!(file.on_event(stopped, &mut io), Ok(Step::None));
    assert_eq!(file.session.status.state, "stopped");
    assert_eq!(file.session.status.activity, "Final digest...");

    // The worker thread's own end comes back too; the session gives it back.
    let step = loop {
        let result = io.next_result();
        if let Ok(step) = file.task_result(result, &mut io) {
            break step;
        }
    };
    assert_eq!(step, Step::None);
    assert_eq!(file.session.status.digest, DIGEST);
    assert_eq!(file.session.status.digest_version, 1);
    assert!(file.session.status.activity.is_empty());
    assert!(io.texts().contains(&"Final digest ready. Press q to quit."));
    assert_eq!(file.quit(&mut io), Step::Close);
}

#[test]
fn a_file_session_gives_back_the_events_it_does_not_use() {
    let dir = tempfile::tempdir().unwrap();
    let mut io = FakeIo::new();
    let mut file = FileRuntime::new(session(dir.path()), Box::new(|_poster, _cancel| {}));
    let event = Event::AudioLevel { rms: 0.5 };
    assert_eq!(file.on_event(event.clone(), &mut io), Err(event));
}

#[test]
fn a_digest_with_an_empty_buffer_only_says_so() {
    let dir = tempfile::tempdir().unwrap();
    let mut io = FakeIo::new();
    let mut runtime = session(dir.path());
    runtime.force_digest(&mut io);
    assert_eq!(io.texts(), ["Nothing in buffer to digest yet"]);
    assert!(!runtime.flow.digest_running);
}

#[test]
fn a_download_shows_its_notice_once_per_download() {
    let dir = tempfile::tempdir().unwrap();
    let mut runtime = session(dir.path());
    let shown = Step::ShowDownload {
        model: "breeze".into(),
    };
    assert_eq!(runtime.on_download_progress(10, "breeze"), shown);
    assert_eq!(runtime.on_download_progress(50, "breeze"), Step::None);
    assert_eq!(runtime.status.download_percent, 50);
    runtime.download_closed();
    assert_eq!(runtime.on_download_progress(1, "breeze"), shown);
}

#[derive(Default)]
struct MuteProbe {
    muted: AtomicBool,
}

impl AudioSource for MuteProbe {
    fn open(&self, _rate: u32, _channels: u16) -> Result<(), AudioError> {
        Ok(())
    }

    fn read(&self, _timeout: Duration) -> Result<Option<Vec<f32>>, AudioError> {
        Ok(None)
    }

    fn close(&self) -> Result<(), AudioError> {
        Ok(())
    }

    fn set_mic_muted(&self, muted: bool) {
        self.muted.store(muted, Ordering::SeqCst);
    }
}

fn recording(dir: &std::path::Path) -> (RecordRuntime, Arc<MuteProbe>, Arc<AtomicBool>) {
    let runtime = session(dir);
    std::fs::create_dir_all(&runtime.output_dir).unwrap();
    let source = Arc::new(MuteProbe::default());
    let paused = Arc::new(AtomicBool::new(false));
    let workers = RecordWorkers {
        audio: Box::new(|_poster, _cancel| {}),
        diarization: None,
        paused: paused.clone(),
        source: source.clone(),
        folder: Arc::new(SessionFolder::new(runtime.output_dir.clone())),
        mic_muted: false,
        consent_noticed: dir.join(".consent_noticed"),
        startup_warning: Some("check the audio".into()),
    };
    (RecordRuntime::new(runtime, workers), source, paused)
}

#[test]
fn a_recording_pauses_mutes_and_stops_without_a_screen() {
    let dir = tempfile::tempdir().unwrap();
    let mut io = FakeIo::new();
    let (mut record, source, paused) = recording(dir.path());
    record.mount(&mut io);
    assert_eq!(io.texts(), ["check the audio"]);
    record.on_event(Event::status(WorkerState::Recording), &mut io);
    assert_eq!(record.session.status.state, "recording");
    assert_eq!(record.quit(&mut io), Step::None, "q waits for s");

    record.toggle_mic(&mut io);
    assert!(source.muted.load(Ordering::SeqCst));
    assert!(record.session.status.mic_muted);

    record.toggle_pause(&mut io);
    assert!(paused.load(Ordering::SeqCst));
    assert_eq!(record.session.status.state, "paused");
    record.toggle_mic(&mut io);
    assert!(
        io.texts().contains(&"Muting unavailable while paused"),
        "{:?}",
        io.texts()
    );

    record.stop_recording(&mut io);
    let status = &record.session.status;
    assert_eq!(status.state, "stopped");
    assert!(status.stopped && !status.recording && !status.paused);
    assert!(status.context_locked);
    assert!(!status.mic_muted);
}

#[test]
fn a_shutdown_exits_once_the_audio_is_flushed() {
    let dir = tempfile::tempdir().unwrap();
    let mut io = FakeIo::new();
    let (mut record, _source, _paused) = recording(dir.path());
    record.on_event(Event::status(WorkerState::Recording), &mut io);
    assert!(record.stop_for_shutdown(), "the worker still flushes");
    record.on_event(Event::status(WorkerState::Stopped), &mut io);
    assert!(io.exited);
}

#[test]
fn capture_health_sets_the_alert_and_says_what_changed() {
    let dir = tempfile::tempdir().unwrap();
    let mut io = FakeIo::new();
    let (mut record, _source, _paused) = recording(dir.path());
    let report = |mic: &str| {
        Event::AudioHealth(lazy_take_notes::app::events::AudioHealthReport {
            mic: mic.into(),
            system: "ok".into(),
            detail: String::new(),
            notice: String::new(),
            processing: "ok".into(),
        })
    };
    record.on_event(report("lost"), &mut io);
    assert_eq!(record.session.status.audio_alert, "MIC LOST");
    record.on_event(report("ok"), &mut io);
    assert!(record.session.status.audio_alert.is_empty());
    assert_eq!(
        io.texts(),
        [
            "Microphone lost. Reconnecting; recording continues.",
            "Microphone is back."
        ]
    );
}
