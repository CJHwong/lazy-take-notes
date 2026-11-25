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
use lazy_take_notes::app::session::{Step, Worker};
use lazy_take_notes::config::build_app_config;
use lazy_take_notes::domain::speaker_turn::SpeakerTurn;
use lazy_take_notes::ports::audio_source::{AudioError, AudioSource};
use lazy_take_notes::tui::host::Host;
use lazy_take_notes::tui::screens::record::{DiarizationLauncher, RecordScreen, RecordWorkers};
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
    start_with(consent_seen, false, None)
}

/// The audio worker's body when a test makes it crash.
fn raising_worker_body() {
    panic!("worker bug");
}

fn start_with(
    consent_seen: bool,
    audio_panics: bool,
    diarization: Option<DiarizationLauncher>,
) -> Run {
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
            if audio_panics {
                raising_worker_body();
            }
        }),
        diarization,
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
            "tab" => KeyCode::Tab,
            "left" => KeyCode::Left,
            "right" => KeyCode::Right,
            "escape" => KeyCode::Esc,
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

#[test]
fn the_stop_waits_for_the_last_speaker_pass_before_the_final_notes() {
    // The fake worker idles until the stop asks for the last pass, then labels S3.
    let diarization: DiarizationLauncher = Box::new(|poster, cancel, finish, _now| {
        while !finish.is_cancelled() && !cancel.is_cancelled() {
            std::thread::sleep(Duration::from_millis(5));
        }
        std::thread::sleep(Duration::from_millis(300));
        poster.post(Event::SpeakerTurns(vec![SpeakerTurn {
            start: 0.0,
            end: 2.0,
            speaker: 3,
        }]));
        poster.post(Event::DiarizationFinished);
    });
    let mut run = start_with(true, false, Some(diarization));
    run.recording();
    run.post(Event::TranscriptChunk(vec![segment("hello", 0.0, 2.0)]));
    run.press("s");
    run.press("q");
    // The audio is flushed, but the speakers are not labeled yet: q waits.
    run.post(Event::status(WorkerState::Stopped));
    assert!(
        run.toasts()
            .iter()
            .any(|t| t.contains("Labeling the speakers")),
        "{:?}",
        run.toasts()
    );
    let deadline = Instant::now() + Duration::from_secs(5);
    while !run.host.finished() && Instant::now() < deadline {
        run.settle();
    }
    assert!(run.host.finished(), "{:?}", run.toasts());
    let labeled =
        std::fs::read_to_string(run.session_dir().join("transcript_speakers.txt")).unwrap();
    assert!(labeled.contains("S3: hello"), "{labeled}");
}

#[test]
fn a_pause_asks_for_a_speaker_pass_at_once() {
    // The fake worker labels S3 only when the screen asks for a pass now.
    let diarization: DiarizationLauncher = Box::new(|poster, cancel, _finish, now| {
        while !now.take() && !cancel.is_cancelled() {
            std::thread::sleep(Duration::from_millis(5));
        }
        poster.post(Event::SpeakerTurns(vec![SpeakerTurn {
            start: 0.0,
            end: 2.0,
            speaker: 3,
        }]));
    });
    let mut run = start_with(true, false, Some(diarization));
    run.recording();
    run.post(Event::TranscriptChunk(vec![segment("hello", 0.0, 2.0)]));
    std::thread::sleep(Duration::from_millis(100));
    run.settle();
    assert!(
        !run.screen_text().contains("S3: hello"),
        "no pass before the pause"
    );
    run.press("space");
    let deadline = Instant::now() + Duration::from_secs(5);
    while !run.screen_text().contains("S3: hello") && Instant::now() < deadline {
        run.settle();
    }
    assert!(
        run.screen_text().contains("S3: hello"),
        "{}",
        run.screen_text()
    );
}

// --- Named tests for the faults of test_record_app_faults.py ---

impl Run {
    fn enter(&mut self) {
        self.host
            .key(&KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        self.settle();
    }

    fn type_text(&mut self, text: &str) {
        for ch in text.chars() {
            self.press(&ch.to_string());
        }
    }

    fn screen_text(&mut self) -> String {
        let mut terminal =
            ratatui::Terminal::new(ratatui::backend::TestBackend::new(120, 40)).unwrap();
        terminal.draw(|frame| self.host.draw(frame)).unwrap();
        let buffer = terminal.backend().buffer();
        (0..buffer.area.height)
            .map(|y| {
                (0..buffer.area.width)
                    .map(|x| buffer[(x, y)].symbol())
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("\n")
    }
}

const MARKUP_LIKE: &str = "[/INST] [bold]not bold[/bold] {brace}";

#[test]
fn markup_like_text_is_shown_verbatim() {
    let mut run = start(true);
    run.recording();
    run.post(Event::TranscriptChunk(vec![segment(MARKUP_LIKE, 0.0, 2.0)]));
    assert!(
        run.screen_text().contains(MARKUP_LIKE),
        "{}",
        run.screen_text()
    );
}

#[test]
fn markup_like_text_survives_a_speaker_redraw() {
    let mut run = start(true);
    run.recording();
    run.post(Event::TranscriptChunk(vec![segment(MARKUP_LIKE, 0.0, 2.0)]));
    run.post(Event::SpeakerTurns(vec![SpeakerTurn {
        start: 0.0,
        end: 2.0,
        speaker: 0,
    }]));
    assert!(
        run.screen_text().contains(&format!("S0: {MARKUP_LIKE}")),
        "{}",
        run.screen_text()
    );
    let labeled =
        std::fs::read_to_string(run.session_dir().join("transcript_speakers.txt")).unwrap();
    assert!(
        labeled.ends_with(&format!("S0: {MARKUP_LIKE}\n")),
        "{labeled}"
    );
}

#[test]
fn a_failed_speakers_write_keeps_the_session() {
    let mut run = start(true);
    run.recording();
    // A folder where the file goes makes the write fail.
    std::fs::create_dir(run.session_dir().join("speakers.json")).unwrap();
    run.post(Event::TranscriptChunk(vec![segment("hello", 0.0, 2.0)]));
    run.post(Event::SpeakerTurns(vec![SpeakerTurn {
        start: 0.0,
        end: 2.0,
        speaker: 0,
    }]));
    assert!(
        run.toasts()
            .iter()
            .any(|toast| toast.starts_with("Could not save speakers:")),
        "{:?}",
        run.toasts()
    );
    run.post(Event::TranscriptChunk(vec![segment(
        "still here",
        2.0,
        3.0,
    )]));
    assert!(!run.host.finished());
    assert!(run.screen_text().contains("still here"));
}

#[test]
fn an_existing_target_is_never_overwritten() {
    let mut run = start(true);
    run.recording();
    let taken = run.dir.path().join("2026-11-19_120000_taken");
    std::fs::create_dir(&taken).unwrap();
    std::fs::write(taken.join("notes.md"), "keep me").unwrap();
    run.press("l");
    run.type_text("taken");
    run.enter();
    assert!(
        run.toasts()
            .contains(&"Session folder 2026-11-19_120000_taken already exists".to_string()),
        "{:?}",
        run.toasts()
    );
    assert!(run.session_dir().exists(), "the session keeps its folder");
    assert_eq!(
        std::fs::read_to_string(taken.join("notes.md")).unwrap(),
        "keep me"
    );
}

#[test]
fn a_failed_rename_keeps_the_session() {
    use std::os::unix::fs::PermissionsExt;
    let mut run = start(true);
    run.recording();
    let parent = run.dir.path().to_path_buf();
    std::fs::set_permissions(&parent, std::fs::Permissions::from_mode(0o500)).unwrap();
    run.press("l");
    run.type_text("standup");
    run.enter();
    std::fs::set_permissions(&parent, std::fs::Permissions::from_mode(0o700)).unwrap();
    assert!(
        run.toasts()
            .iter()
            .any(|toast| toast.starts_with("Could not rename the session folder:")),
        "{:?}",
        run.toasts()
    );
    assert!(run.session_dir().exists());
    run.post(Event::TranscriptChunk(vec![segment(
        "after the rename",
        0.0,
        1.0,
    )]));
    assert!(!run.host.finished());
}

#[test]
fn a_lost_mic_stays_on_the_status_bar_until_it_returns() {
    let mut run = start(true);
    run.recording();
    let bar = |run: &mut Run| run.screen_text().lines().last().unwrap_or("").to_string();
    run.post(health("lost", "ok", "ok"));
    assert!(bar(&mut run).contains("MIC LOST"), "{}", bar(&mut run));
    run.post(health("ok", "lost", "ok"));
    let text = bar(&mut run);
    assert!(
        !text.contains("MIC LOST") && text.contains("SYS LOST"),
        "{text}"
    );
    run.post(health("silent", "ok", "ok"));
    let text = bar(&mut run);
    assert!(
        text.contains("MIC SILENT") && !text.contains("SYS LOST"),
        "{text}"
    );
    run.post(health("ok", "ok", "error"));
    assert!(bar(&mut run).contains("AUDIO ERROR"), "{}", bar(&mut run));
    run.post(health("ok", "ok", "ok"));
    let text = bar(&mut run);
    for alert in ["LOST", "SILENT", "AUDIO ERROR"] {
        assert!(!text.contains(alert), "{text}");
    }
}

/// A mode whose worker has finished.
struct Idle;

impl Worker for Idle {
    fn worker_busy(&self) -> bool {
        false
    }
    fn move_folder(
        &mut self,
        _from: &std::path::Path,
        _to: &std::path::Path,
    ) -> std::io::Result<()> {
        Ok(())
    }
}

#[test]
fn it_closes_once_even_when_called_again() {
    // A second q and the label task's end both reach close_session.
    let raw = yaml_serde::from_str("output: {auto_label: false}").unwrap();
    let controller = SessionController::new(
        build_app_config(&raw).unwrap(),
        built_in_template("default_en"),
        Arc::new(FakeLlm::new(Reply::Text("## Notes".into(), 10))),
        Box::new(FakePersistence::new().0),
    );
    let dir = tempfile::tempdir().unwrap();
    let mut session = Session::new(controller, dir.path().to_path_buf(), "", vec![], vec![]);
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let mut host = Host::new("textual-dark", runtime.handle().clone());
    let session = &mut session.runtime;
    assert_eq!(session.close_session(), Step::Close);
    assert_eq!(session.close_session(), Step::None);
    session.flow.pending_quit = true;
    session.flow.final_attempted = true;
    assert_eq!(session.continue_pending_quit(&Idle, host.ctx()), Step::None);
}

#[test]
fn a_worker_crash_logs_the_stack_and_the_session_continues() {
    let logs = tempfile::tempdir().unwrap();
    lazy_take_notes::app::logging::setup_file_logging(logs.path(), true);
    let mut run = start_with(true, true, None);
    run.recording();
    run.press("s");
    assert!(!run.host.finished());
    assert!(
        run.toasts()
            .contains(&"Recording stopped. You can still browse and run quick actions.".into()),
        "{:?}",
        run.toasts()
    );
    // The worker panics on its own thread, maybe after the stop: wait for its line.
    let path = logs.path().join("debug.log");
    let deadline = Instant::now() + Duration::from_secs(5);
    let read = || std::fs::read_to_string(&path).unwrap_or_default();
    while !read().contains("CRITICAL panic in thread audio") && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(10));
    }
    lazy_take_notes::app::logging::close_file_logging();
    let log = read();
    assert!(log.contains("CRITICAL panic in thread audio"), "{log}");
    assert!(log.contains("worker bug"), "{log}");
    assert!(log.contains("raising_worker_body"), "{log}");
}

impl Run {
    /// A left click on the first cell where `text` shows.
    fn click(&mut self, text: &str) {
        let screen = self.screen_text();
        let (row, line) = screen
            .lines()
            .enumerate()
            .find(|(_, line)| line.contains(text))
            .unwrap_or_else(|| panic!("{text:?} is not on screen:\n{screen}"));
        let column = line[..line.find(text).unwrap()].chars().count() as u16;
        let at = |kind| crossterm::event::MouseEvent {
            kind,
            column,
            row: row as u16,
            modifiers: KeyModifiers::NONE,
        };
        let left = crossterm::event::MouseButton::Left;
        self.host
            .mouse(&at(crossterm::event::MouseEventKind::Down(left)));
        self.settle();
    }
}

#[test]
fn a_click_on_a_speaker_chip_names_that_speaker() {
    let mut run = start(true);
    run.recording();
    run.post(Event::TranscriptChunk(vec![segment("hello", 0.0, 2.0)]));
    run.post(Event::SpeakerTurns(vec![SpeakerTurn {
        start: 0.0,
        end: 2.0,
        speaker: 0,
    }]));
    // The first click selects; the second opens the prompt.
    run.click(" S0 2s ");
    assert!(!run.screen_text().contains("Name for S0"));
    run.click(" S0 2s ");
    assert!(
        run.screen_text().contains("Name for S0"),
        "{}",
        run.screen_text()
    );
    run.type_text("Andy");
    run.enter();
    let screen = run.screen_text();
    assert!(screen.contains("Andy: hello"), "{screen}");
    assert!(screen.contains(" Andy 2s "), "{screen}");
    let json = std::fs::read_to_string(run.session_dir().join("speakers.json")).unwrap();
    assert!(json.contains("\"Andy\""), "{json}");
    let labeled =
        std::fs::read_to_string(run.session_dir().join("transcript_speakers.txt")).unwrap();
    assert!(labeled.ends_with("Andy: hello\n"), "{labeled}");
    // The chip is still selected, so one click opens the name to edit.
    run.click(" Andy 2s ");
    assert!(
        run.screen_text().contains("Name for S0"),
        "{}",
        run.screen_text()
    );
}

#[test]
fn eight_speakers_wrap_onto_more_rows_and_each_chip_still_opens_its_prompt() {
    let mut run = start(true);
    run.recording();
    run.post(Event::TranscriptChunk(vec![segment("hello", 0.0, 80.0)]));
    let turns = (0..8)
        .map(|speaker| SpeakerTurn {
            start: f64::from(speaker) * 10.0,
            end: f64::from(speaker) * 10.0 + 9.0,
            speaker,
        })
        .collect();
    run.post(Event::SpeakerTurns(turns));
    let screen = run.screen_text();
    for speaker in 0..8 {
        assert!(
            screen.contains(&format!(" S{speaker} 9s ")),
            "S{speaker}:\n{screen}"
        );
    }
    let rows = screen.lines().filter(|line| line.contains(" 9s ")).count();
    assert!(rows >= 2, "{screen}");
    run.click(" S7 9s ");
    run.click(" S7 9s ");
    assert!(
        run.screen_text().contains("Name for S7"),
        "{}",
        run.screen_text()
    );
}

#[test]
fn a_speaker_pass_shows_on_the_status_bar_while_it_runs() {
    let mut run = start(true);
    run.recording();
    run.post(Event::DiarizationStatus { active: true });
    assert!(
        run.screen_text().contains("Labeling speakers"),
        "{}",
        run.screen_text()
    );
    run.post(Event::DiarizationStatus { active: false });
    assert!(!run.screen_text().contains("Labeling speakers"));
}

fn two_speakers(run: &mut Run) {
    run.recording();
    run.post(Event::TranscriptChunk(vec![
        segment("hello", 0.0, 2.0),
        segment("hi", 3.0, 4.0),
    ]));
    run.post(Event::SpeakerTurns(vec![
        SpeakerTurn {
            start: 0.0,
            end: 2.0,
            speaker: 0,
        },
        SpeakerTurn {
            start: 3.0,
            end: 4.0,
            speaker: 1,
        },
    ]));
}

#[test]
fn the_speaker_bar_is_a_tab_stop_only_once_a_speaker_is_heard() {
    // No speakers: two tabs reach the context box.
    let mut run = start(true);
    run.recording();
    run.press("tab");
    run.press("tab");
    run.type_text("zz");
    assert!(run.screen_text().contains("zz"), "{}", run.screen_text());
    // With speakers, the bar comes between the transcript and the digest.
    let mut run = start(true);
    two_speakers(&mut run);
    run.press("tab");
    run.press("tab");
    run.type_text("zz");
    assert!(!run.screen_text().contains("zz"), "{}", run.screen_text());
    run.press("tab");
    run.type_text("zz");
    assert!(run.screen_text().contains("zz"), "{}", run.screen_text());
}

#[test]
fn the_arrows_choose_a_speaker_and_enter_names_it() {
    let mut run = start(true);
    two_speakers(&mut run);
    // Tab selects the first chip; Right moves on and stops at the last one.
    run.press("tab");
    run.press("right");
    run.press("right");
    run.enter();
    assert!(
        run.screen_text().contains("Name for S1"),
        "{}",
        run.screen_text()
    );
    run.type_text("Bea");
    run.enter();
    let screen = run.screen_text();
    assert!(screen.contains("Bea: hi"), "{screen}");
    assert!(screen.contains(" Bea 1s "), "{screen}");
    // Left goes back to S0.
    run.press("left");
    run.enter();
    assert!(
        run.screen_text().contains("Name for S0"),
        "{}",
        run.screen_text()
    );
}

#[test]
fn escape_on_the_speaker_bar_drops_the_selection() {
    let mut run = start(true);
    two_speakers(&mut run);
    run.press("tab");
    run.press("escape");
    run.enter();
    assert!(
        !run.screen_text().contains("Name for"),
        "{}",
        run.screen_text()
    );
}

#[test]
fn enter_after_a_click_on_a_chip_names_that_speaker() {
    let mut run = start(true);
    two_speakers(&mut run);
    run.click(" S1 1s ");
    run.enter();
    assert!(
        run.screen_text().contains("Name for S1"),
        "{}",
        run.screen_text()
    );
}
