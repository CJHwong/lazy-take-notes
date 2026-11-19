//! The record screen against fake workers, a fake LLM and a fake disk, through a real host.

mod common;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use common::{FakeLlm, FakePersistence, Reply, built_in_template, segment};
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use lazy_take_notes::adapters::audio::session_folder::SessionFolder;
use lazy_take_notes::adapters::session_controller::SessionController;
use lazy_take_notes::app::events::{AudioHealthReport, Cancel, Event, WorkerState};
use lazy_take_notes::config::build_app_config;
use lazy_take_notes::domain::speaker_turn::SpeakerTurn;
use lazy_take_notes::ports::audio_source::{AudioError, AudioSource};
use lazy_take_notes::tui::host::Host;
use lazy_take_notes::tui::screens::record::{RecordScreen, RecordWorkers};
use lazy_take_notes::tui::screens::session::Session;

/// Remembers the last mute the screen set.
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

struct Run {
    host: Host,
    source: Arc<MuteProbe>,
    paused: Arc<AtomicBool>,
    cancel: Arc<Mutex<Option<Cancel>>>,
    dir: tempfile::TempDir,
    _runtime: tokio::runtime::Runtime,
}

fn start(consent_seen: bool) -> Run {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();
    let raw = yaml_serde::from_str("output: {auto_label: false}").unwrap();
    let config = build_app_config(&raw).unwrap();
    let (persistence, _saved) = FakePersistence::new();
    let controller = SessionController::new(
        config,
        built_in_template("default_en"),
        Arc::new(FakeLlm::new(Reply::Text("## Notes".into(), 10))),
        Box::new(persistence),
    );
    let dir = tempfile::tempdir().unwrap();
    let out_dir = dir.path().join("2026-11-19_120000");
    std::fs::create_dir_all(&out_dir).unwrap();
    let session = Session::new(controller, out_dir.clone(), "", vec![], vec![]);
    let consent = dir.path().join(".consent_noticed");
    if consent_seen {
        std::fs::write(&consent, "").unwrap();
    }
    let source = Arc::new(MuteProbe::default());
    let paused = Arc::new(AtomicBool::new(false));
    let cancel = Arc::new(Mutex::new(None));
    let seen = cancel.clone();
    let workers = RecordWorkers {
        // The worker keeps its cancel so a test can see the stop; each test posts events.
        audio: Box::new(move |_poster, cancel| {
            *seen.lock().unwrap() = Some(cancel);
        }),
        diarization: None,
        paused: paused.clone(),
        source: source.clone(),
        folder: Arc::new(SessionFolder::new(out_dir)),
        mic_muted: false,
        consent_noticed: consent,
        startup_warning: None,
    };
    let mut host = Host::new("textual-dark", runtime.handle().clone());
    host.push(Box::new(RecordScreen::new(session, 0.01, workers)));
    let mut run = Run {
        host,
        source,
        paused,
        cancel,
        dir,
        _runtime: runtime,
    };
    run.settle();
    run
}

impl Run {
    fn post(&mut self, event: Event) {
        self.host.ctx().poster().post(event);
        self.settle();
    }

    fn press(&mut self, name: &str) {
        let code = match name {
            "space" => KeyCode::Char(' '),
            name => KeyCode::Char(name.chars().next().unwrap()),
        };
        self.host.key(&KeyEvent::new(code, KeyModifiers::NONE));
        self.settle();
    }

    fn settle(&mut self) {
        let mut quiet_since = Instant::now();
        let deadline = Instant::now() + Duration::from_secs(5);
        while Instant::now() < deadline && quiet_since.elapsed() < Duration::from_millis(150) {
            if self.host.pump() > 0 {
                quiet_since = Instant::now();
            }
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    fn toasts(&mut self) -> Vec<String> {
        self.host.ctx().toast_texts()
    }

    fn recording(&mut self) {
        self.post(Event::status(WorkerState::Recording));
    }

    fn cancelled(&self) -> bool {
        self.cancel
            .lock()
            .unwrap()
            .as_ref()
            .is_some_and(Cancel::is_cancelled)
    }

    fn session_dir(&self) -> std::path::PathBuf {
        self.dir.path().join("2026-11-19_120000")
    }
}

fn health(mic: &str, system: &str, processing: &str) -> Event {
    Event::AudioHealth(AudioHealthReport {
        mic: mic.into(),
        system: system.into(),
        detail: "device gone".into(),
        notice: String::new(),
        processing: processing.into(),
    })
}

#[test]
fn the_consent_notice_shows_until_n_stops_it() {
    let mut run = start(false);
    assert_eq!(run.host.depth(), 2, "the notice is on top of the session");
    run.press("n");
    assert_eq!(run.host.depth(), 1);
    assert!(run.dir.path().join(".consent_noticed").exists());
    let mut seen = start(true);
    assert_eq!(seen.host.depth(), 1);
    seen.press("x");
    assert_eq!(seen.host.depth(), 1);
}

#[test]
fn q_does_nothing_while_recording_and_s_stops_the_worker() {
    let mut run = start(true);
    run.recording();
    run.press("q");
    assert!(!run.host.finished());
    assert!(!run.cancelled());
    run.press("s");
    assert!(run.cancelled());
    assert!(
        run.toasts()
            .contains(&"Recording stopped. You can still browse and run quick actions.".into())
    );
}

#[test]
fn space_pauses_and_resumes_and_mute_waits_for_resume() {
    let mut run = start(true);
    run.recording();
    run.press("space");
    assert!(run.paused.load(Ordering::SeqCst));
    run.press("m");
    assert!(!run.source.muted.load(Ordering::SeqCst));
    assert!(
        run.toasts()
            .contains(&"Muting unavailable while paused".into())
    );
    run.press("space");
    assert!(!run.paused.load(Ordering::SeqCst));
    run.press("m");
    assert!(run.source.muted.load(Ordering::SeqCst));
    run.press("m");
    assert!(!run.source.muted.load(Ordering::SeqCst));
}

#[test]
fn health_changes_toast_once_each_and_recovery_says_so() {
    let mut run = start(true);
    run.recording();
    run.post(health("lost", "ok", "ok"));
    run.post(health("lost", "ok", "ok"));
    run.post(health("ok", "ok", "error"));
    run.post(health("ok", "ok", "ok"));
    let toasts = run.toasts();
    let expected = [
        "Microphone lost (device gone). Reconnecting; recording continues.",
        "Microphone is back.",
        "Audio processing keeps failing (device gone). Nothing is recorded until it recovers.",
        "Audio processing recovered.",
    ];
    for text in expected {
        assert_eq!(
            toasts.iter().filter(|toast| *toast == text).count(),
            1,
            "{text}: {toasts:?}"
        );
    }
}

#[test]
fn speaker_turns_label_the_lines_and_are_saved() {
    let mut run = start(true);
    run.recording();
    run.post(Event::TranscriptChunk(vec![segment("hello", 0.0, 2.0)]));
    run.post(Event::SpeakerTurns(vec![SpeakerTurn {
        start: 0.0,
        end: 2.0,
        speaker: 0,
    }]));
    let json = std::fs::read_to_string(run.session_dir().join("speakers.json")).unwrap();
    assert!(json.contains("\"speaker\": 0"), "{json}");
    let labeled =
        std::fs::read_to_string(run.session_dir().join("transcript_speakers.txt")).unwrap();
    assert!(labeled.ends_with("S0: hello\n"), "{labeled}");
    // A new line is labeled and saved at once.
    run.post(Event::TranscriptChunk(vec![segment("again", 1.0, 2.0)]));
    let labeled =
        std::fs::read_to_string(run.session_dir().join("transcript_speakers.txt")).unwrap();
    assert!(labeled.ends_with("S0: again\n"), "{labeled}");
}

#[test]
fn a_worker_that_stops_on_its_own_runs_the_final_digest() {
    let mut run = start(true);
    run.recording();
    run.post(Event::TranscriptChunk(vec![segment("hello", 0.0, 2.0)]));
    run.post(Event::status(WorkerState::Stopped));
    assert!(!run.host.finished());
    // The final digest ran: q now closes the screen.
    run.press("q");
    run.settle();
    assert!(run.host.finished(), "{:?}", run.toasts());
}
