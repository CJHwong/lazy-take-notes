//! The transcribe screen against a fake LLM and a fake disk, through a real host.

mod common;

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use common::{FakeLlm, FakePersistence, Reply, Saved, built_in_template, segment};
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use lazy_take_notes::adapters::session_controller::SessionController;
use lazy_take_notes::app::events::{Event, WorkerState};
use lazy_take_notes::config::build_app_config;
use lazy_take_notes::ports::llm::LlmError;
use lazy_take_notes::tui::host::Host;
use lazy_take_notes::tui::screens::session::Session;
use lazy_take_notes::tui::screens::transcribe::TranscribeScreen;

const DIGEST: &str = "## Topic\nSome content";

struct Run {
    host: Host,
    llm: Arc<FakeLlm>,
    saved: Arc<Mutex<Saved>>,
    _runtime: tokio::runtime::Runtime,
    _dir: tempfile::TempDir,
}

fn start(reply: Reply) -> Run {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();
    let raw = yaml_serde::from_str("output: {auto_label: false}").unwrap();
    let config = build_app_config(&raw).unwrap();
    let llm = Arc::new(FakeLlm::new(reply));
    let (persistence, saved) = FakePersistence::new();
    let controller = SessionController::new(
        config,
        built_in_template("default_en"),
        llm.clone(),
        Box::new(persistence),
    );
    let dir = tempfile::tempdir().unwrap();
    let session = Session::new(
        controller,
        dir.path().join("2026-11-19_120000"),
        "",
        vec![],
        vec![],
    );
    // The worker does nothing: each test posts its events.
    let screen = TranscribeScreen::new(session, 0.01, Box::new(|_poster, _cancel| {}));
    let mut host = Host::new("textual-dark", runtime.handle().clone());
    host.push(Box::new(screen));
    Run {
        host,
        llm,
        saved,
        _runtime: runtime,
        _dir: dir,
    }
}

impl Run {
    fn post(&mut self, event: Event) {
        self.host.ctx().poster().post(event);
        self.settle();
    }

    fn press(&mut self, name: &str) {
        let (code, modifiers) = match name {
            "tab" => (KeyCode::Tab, KeyModifiers::NONE),
            name => (
                KeyCode::Char(name.chars().next().unwrap()),
                KeyModifiers::NONE,
            ),
        };
        self.host.key(&KeyEvent::new(code, modifiers));
        self.settle();
    }

    /// Step the host until nothing has arrived for a short while.
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

    fn lines(&mut self, texts: &[&str]) {
        let segments = texts
            .iter()
            .enumerate()
            .map(|(index, text)| segment(text, index as f64, index as f64 + 1.0))
            .collect();
        self.post(Event::TranscriptChunk(segments));
    }

    fn toasts(&mut self) -> Vec<String> {
        self.host.ctx().toast_texts()
    }

    fn notes(&self) -> Vec<(String, u32)> {
        self.saved.lock().unwrap().digest.clone()
    }
}

#[test]
fn an_unexpected_digest_panic_fails_the_digest_not_the_session() {
    let mut run = start(Reply::Panic("bug in the client"));
    run.post(Event::status(WorkerState::Recording));
    run.lines(&["first line", "second line"]);
    run.press("d");
    assert!(
        run.toasts()
            .iter()
            .any(|toast| toast.contains("bug in the client")),
        "{:?}",
        run.toasts()
    );
    run.llm.set(Reply::Text(DIGEST.into(), 10));
    run.press("d");
    assert!(
        !run.toasts()
            .contains(&"Digest already in progress".to_string()),
        "the panicked digest released the running flag"
    );
    assert_eq!(run.notes(), [(DIGEST.to_string(), 1)]);
}

#[test]
fn an_unexpected_quick_action_panic_shows_an_error_not_a_stuck_query() {
    let mut run = start(Reply::Panic("bug in the client"));
    run.post(Event::status(WorkerState::Recording));
    run.lines(&["first line"]);
    run.press("1");
    run.press("escape");
    run.llm.set(Reply::Text("an answer".into(), 1));
    run.press("1");
    assert!(
        !run.toasts().iter().any(|toast| toast.contains("already")),
        "the panicked query released its flag: {:?}",
        run.toasts()
    );
}

#[test]
fn the_final_digest_replaces_a_running_one_and_drops_its_prompt() {
    let mut run = start(Reply::Hang);
    run.post(Event::status(WorkerState::Recording));
    run.lines(&["first line", "second line"]);
    run.press("d");
    assert_eq!(run.llm.chat_calls.lock().unwrap().len(), 1);
    run.llm.set(Reply::Text(DIGEST.into(), 10));
    run.post(Event::status(WorkerState::Stopped));
    assert_eq!(
        run.notes(),
        [(DIGEST.to_string(), 1)],
        "only the final digest saved notes"
    );
    let sent = run.llm.last_messages();
    assert_eq!(
        sent.len(),
        2,
        "system prompt and the final prompt only: {sent:?}"
    );
    assert!(
        run.toasts()
            .contains(&"Final digest ready. Press q to quit.".to_string())
    );
    run.press("d");
    assert!(
        run.toasts()
            .contains(&"Nothing in buffer to digest yet".to_string()),
        "the replaced digest no longer holds the running flag"
    );
}

#[test]
fn q_during_a_digest_waits_for_it_then_closes() {
    let mut run = start(Reply::Hang);
    run.post(Event::status(WorkerState::Recording));
    run.lines(&["a line"]);
    run.press("d");
    run.post(Event::status(WorkerState::Stopped));
    run.press("q");
    assert!(
        run.toasts()
            .contains(&"Quitting when the digest finishes.".to_string())
    );
    assert_eq!(run.host.depth(), 1, "still waiting for the digest");
    // The final digest hangs, so a second q still waits for it.
    run.press("q");
    assert_eq!(
        run.host.depth(),
        1,
        "a running digest still holds the second q"
    );
}

#[test]
fn a_failed_final_digest_is_tried_once_then_the_screen_closes() {
    let mut run = start(Reply::Fail(LlmError::new("ConnectError", "down")));
    run.post(Event::status(WorkerState::Recording));
    run.lines(&["a line"]);
    run.press("q");
    assert_eq!(run.host.depth(), 1, "the worker still runs");
    run.post(Event::status(WorkerState::Stopped));
    assert_eq!(run.llm.chat_calls.lock().unwrap().len(), 1);
    assert_eq!(run.host.depth(), 0, "closed after the one failed try");
    assert!(
        run.toasts()
            .iter()
            .any(|toast| toast.starts_with("Digest failed: "))
    );
}

#[test]
fn the_end_of_the_file_writes_the_final_notes_on_its_own() {
    let mut run = start(Reply::Text(DIGEST.into(), 10));
    run.post(Event::status(WorkerState::Recording));
    run.lines(&["a line"]);
    run.post(Event::status(WorkerState::Stopped));
    assert_eq!(run.notes().len(), 1);
    assert_eq!(run.host.depth(), 1, "the screen stays until q");
    run.press("q");
    assert_eq!(run.host.depth(), 0);
}

#[test]
fn closing_from_under_a_modal_closes_the_modal_too() {
    let mut run = start(Reply::Text(DIGEST.into(), 10));
    run.post(Event::status(WorkerState::Recording));
    run.press("q");
    run.press("h");
    assert_eq!(run.host.depth(), 2, "help is open");
    run.post(Event::status(WorkerState::Stopped));
    assert_eq!(run.host.depth(), 0);
}

#[test]
fn letters_typed_into_the_context_box_do_not_run_bindings() {
    let mut run = start(Reply::Text(DIGEST.into(), 10));
    run.press("tab");
    run.press("tab");
    for letter in ["a", "q", "h", "s", "1"] {
        run.press(letter);
    }
    assert_eq!(run.host.depth(), 1, "q did not quit and h opened no help");
    assert!(run.toasts().is_empty(), "{:?}", run.toasts());
}
