//! The host's safety net and screen stack.

use std::sync::{Arc, Mutex};

use crossterm::event::{KeyCode, KeyModifiers};
use ratatui::Terminal;
use ratatui::backend::TestBackend;

use super::*;

/// A screen driven by the test: what each key does, and what it saw.
#[derive(Default)]
struct Probe {
    panic_on_key: bool,
    session: bool,
    stops: bool,
    modal: bool,
    hint: Option<String>,
    push_on_key: Option<fn() -> Box<dyn Screen>>,
    pop_with: Option<i32>,
    resumed: Arc<Mutex<Vec<Option<i32>>>>,
}

impl Screen for Probe {
    fn render(&mut self, frame: &mut Frame, area: Rect, _ctx: &Ctx) {
        frame.render_widget(Paragraph::new("probe screen"), area);
    }

    fn key(&mut self, _key: &KeyEvent, _name: &str, _ctx: &mut Ctx) -> Action {
        if self.panic_on_key {
            panic!("boom");
        }
        if let Some(make) = self.push_on_key {
            return Action::Push(make());
        }
        if let Some(value) = self.pop_with {
            return Action::Pop(Some(Box::new(value)));
        }
        Action::None
    }

    fn resumed(&mut self, result: Option<Payload>, _ctx: &mut Ctx) -> Action {
        let value = result
            .and_then(|payload| payload.downcast::<i32>().ok())
            .map(|value| *value);
        self.resumed.lock().unwrap().push(value);
        Action::None
    }

    fn is_modal(&self) -> bool {
        self.modal
    }

    fn hosts_session(&self) -> bool {
        self.session
    }

    fn stop_for_shutdown(&mut self, _ctx: &mut Ctx) -> bool {
        self.stops
    }

    fn quit_hint(&self) -> Option<String> {
        self.hint.clone()
    }
}

fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
}

fn press(host: &mut Host, code: KeyCode, modifiers: KeyModifiers) {
    host.key(&KeyEvent::new(code, modifiers));
}

#[test]
fn a_panicking_handler_shows_a_toast_and_the_session_continues() {
    let runtime = runtime();
    let mut host = Host::new("textual-dark", runtime.handle().clone());
    host.push(Box::new(Probe {
        panic_on_key: true,
        session: true,
        ..Probe::default()
    }));
    press(&mut host, KeyCode::Char('x'), KeyModifiers::NONE);
    assert_eq!(host.depth(), 1);
    assert!(!host.finished());
    let toasts = host.ctx().toast_texts();
    assert_eq!(toasts.len(), 1);
    assert!(
        toasts[0].starts_with("Unexpected error: boom"),
        "{toasts:?}"
    );
    assert!(
        toasts[0].ends_with(". The session continues."),
        "{toasts:?}"
    );
}

#[test]
fn an_error_loop_shows_one_sticky_alert_instead_of_more_toasts() {
    let runtime = runtime();
    let mut host = Host::new("textual-dark", runtime.handle().clone());
    host.push(Box::new(Probe {
        panic_on_key: true,
        ..Probe::default()
    }));
    for _ in 0..10 {
        press(&mut host, KeyCode::Char('x'), KeyModifiers::NONE);
    }
    let toasts = host.ctx().toast_texts();
    assert_eq!(toasts.len(), ERROR_BURST_LIMIT + 1, "{toasts:?}");
    assert!(toasts.last().unwrap().starts_with("Errors keep repeating"));
    assert!(
        toasts
            .last()
            .unwrap()
            .ends_with("The audio recording continues; press s to stop and save.")
    );
    assert!(
        !toasts[0].contains("session continues"),
        "no session, no tail: {toasts:?}"
    );
    terminal::set_quiet_panics(false);
}

#[test]
fn a_popped_result_reaches_the_screen_below() {
    let runtime = runtime();
    let mut host = Host::new("textual-dark", runtime.handle().clone());
    let resumed = Arc::new(Mutex::new(Vec::new()));
    host.push(Box::new(Probe {
        push_on_key: Some(|| {
            Box::new(Probe {
                pop_with: Some(7),
                ..Probe::default()
            })
        }),
        resumed: resumed.clone(),
        ..Probe::default()
    }));
    press(&mut host, KeyCode::Enter, KeyModifiers::NONE);
    assert_eq!(host.depth(), 2);
    press(&mut host, KeyCode::Enter, KeyModifiers::NONE);
    assert_eq!(host.depth(), 1);
    assert_eq!(*resumed.lock().unwrap(), [Some(7)]);
}

#[test]
fn ctrl_c_says_how_to_quit_and_ctrl_q_waits_for_the_audio_flush() {
    let runtime = runtime();
    let mut host = Host::new("textual-dark", runtime.handle().clone());
    host.push(Box::new(Probe::default()));
    press(&mut host, KeyCode::Char('c'), KeyModifiers::CONTROL);
    assert_eq!(host.ctx().toast_texts(), ["Press ctrl+q to quit the app"]);
    host.push(Box::new(Probe {
        hint: Some("Recording: press s to stop and save first".into()),
        stops: true,
        ..Probe::default()
    }));
    press(&mut host, KeyCode::Char('c'), KeyModifiers::CONTROL);
    assert_eq!(
        host.ctx().toast_texts().last().unwrap(),
        "Recording: press s to stop and save first"
    );
    press(&mut host, KeyCode::Char('q'), KeyModifiers::CONTROL);
    assert!(!host.finished(), "waits SHUTDOWN_GRACE for the audio flush");
    let mut idle = Host::new("textual-dark", runtime.handle().clone());
    idle.push(Box::new(Probe::default()));
    press(&mut idle, KeyCode::Char('q'), KeyModifiers::CONTROL);
    assert!(idle.finished());
}

#[test]
fn a_modal_draws_over_the_dimmed_screen_and_toasts_go_on_top() {
    let runtime = runtime();
    let mut host = Host::new("textual-dark", runtime.handle().clone());
    host.push(Box::new(Probe::default()));
    host.push(Box::new(Probe {
        modal: true,
        ..Probe::default()
    }));
    host.ctx().notify("Transcript copied");
    let mut terminal = Terminal::new(TestBackend::new(60, 12)).unwrap();
    terminal.draw(|frame| host.draw(frame)).unwrap();
    let buffer = terminal.backend().buffer().clone();
    let text: String = buffer.content().iter().map(|cell| cell.symbol()).collect();
    assert!(text.contains("probe screen"));
    assert!(text.contains("Transcript copied"));
    // A blank cell keeps the base screen's foreground, blended 60% toward the background.
    let foreground = host.ctx().theme.color("foreground");
    assert_ne!(
        buffer[(30, 0)].fg,
        foreground,
        "the screen below the modal is dimmed"
    );
}

/// Records the keys it gets.
struct KeyLog(Arc<Mutex<Vec<String>>>);

impl Screen for KeyLog {
    fn render(&mut self, _frame: &mut Frame, _area: Rect, _ctx: &Ctx) {}

    fn key(&mut self, _key: &KeyEvent, name: &str, _ctx: &mut Ctx) -> Action {
        self.0.lock().unwrap().push(name.to_string());
        Action::None
    }
}

#[test]
fn tab_closes_a_notice_and_still_moves_the_focus_below() {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let keys = Arc::new(Mutex::new(Vec::new()));
    let mut host = Host::new("textual-dark", runtime.handle().clone());
    host.push(Box::new(KeyLog(keys.clone())));
    let notice = || {
        Box::new(
            crate::tui::screens::modals::MarkdownModal::builtin_templates_notice(
                std::path::PathBuf::from("/nonexistent/flag"),
            ),
        )
    };
    host.push(notice());
    host.key(&KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE));
    assert_eq!(host.depth(), 1);
    assert_eq!(*keys.lock().unwrap(), vec!["tab".to_string()]);
    // Any other key only closes it, as in Python.
    host.push(notice());
    host.key(&KeyEvent::new(KeyCode::Char('a'), KeyModifiers::NONE));
    assert_eq!(host.depth(), 1);
    assert_eq!(keys.lock().unwrap().len(), 1);
}

/// Keeps a sender to post later, and records the numbers it receives.
struct LateSender {
    got: Arc<Mutex<Vec<i32>>>,
    sender: Arc<Mutex<Option<TaskSender>>>,
}

impl Screen for LateSender {
    fn render(&mut self, _frame: &mut Frame, _area: Rect, _ctx: &Ctx) {}

    fn key(&mut self, _key: &KeyEvent, _name: &str, _ctx: &mut Ctx) -> Action {
        Action::Pop(None)
    }

    fn mounted(&mut self, ctx: &mut Ctx) -> Action {
        *self.sender.lock().unwrap() = Some(ctx.task_sender());
        Action::None
    }

    fn message(&mut self, msg: Msg, _ctx: &mut Ctx) -> (Option<Msg>, Action) {
        match msg {
            Msg::Task(payload) => match payload.downcast::<i32>() {
                Ok(value) => {
                    self.got.lock().unwrap().push(*value);
                    (None, Action::None)
                }
                Err(payload) => (Some(Msg::Task(payload)), Action::None),
            },
            other => (Some(other), Action::None),
        }
    }
}

#[test]
fn a_result_from_a_closed_screen_never_reaches_the_next_one() {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let mut host = Host::new("textual-dark", runtime.handle().clone());
    host.push(Box::new(KeyLog(Arc::default())));
    let (old_sender, new_sender) = (Arc::default(), Arc::default());
    let (old_got, new_got) = (
        Arc::new(Mutex::new(Vec::new())),
        Arc::new(Mutex::new(Vec::new())),
    );
    host.push(Box::new(LateSender {
        got: old_got.clone(),
        sender: Arc::clone(&old_sender),
    }));
    host.key(&KeyEvent::new(KeyCode::Char('q'), KeyModifiers::NONE));
    host.push(Box::new(LateSender {
        got: new_got.clone(),
        sender: Arc::clone(&new_sender),
    }));
    old_sender.lock().unwrap().as_ref().unwrap().post(1_i32);
    new_sender.lock().unwrap().as_ref().unwrap().post(2_i32);
    host.pump();
    assert_eq!(*new_got.lock().unwrap(), vec![2]);
    assert!(old_got.lock().unwrap().is_empty());
}

// --- Named tests for the faults of test_record_app_faults.py ---

/// A screen whose message handler panics on the first message, then counts the rest.
#[derive(Default)]
struct FlakyHandler {
    seen: Arc<Mutex<u32>>,
}

impl Screen for FlakyHandler {
    fn render(&mut self, _frame: &mut Frame, _area: Rect, _ctx: &Ctx) {}

    fn key(&mut self, _key: &KeyEvent, _name: &str, _ctx: &mut Ctx) -> Action {
        Action::None
    }

    fn message(&mut self, _msg: Msg, _ctx: &mut Ctx) -> (Option<Msg>, Action) {
        let count = {
            let mut seen = self.seen.lock().unwrap();
            *seen += 1;
            *seen
        };
        if count == 1 {
            panic!("bug in a handler");
        }
        (None, Action::None)
    }
}

/// A host whose exit watchdog counts as armed, so a test signal never ends the test run.
fn host_without_watchdog(runtime: &tokio::runtime::Runtime) -> Host {
    let host = Host::new("textual-dark", runtime.handle().clone());
    host.watchdog_armed.store(true, Ordering::SeqCst);
    host
}

#[test]
fn a_raising_handler_does_not_stop_the_screen_from_handling_messages() {
    let runtime = runtime();
    let mut host = Host::new("textual-dark", runtime.handle().clone());
    let seen = Arc::new(Mutex::new(0));
    host.push(Box::new(FlakyHandler { seen: seen.clone() }));
    host.message(Msg::Task(Box::new(1_u8)));
    host.message(Msg::Task(Box::new(2_u8)));
    assert_eq!(*seen.lock().unwrap(), 2, "the second message still arrived");
    assert!(!host.finished());
}

#[test]
fn a_signal_stops_the_session_and_exits_after_the_audio_flush() {
    let runtime = runtime();
    let mut host = host_without_watchdog(&runtime);
    host.push(Box::new(Probe {
        session: true,
        stops: true,
        ..Probe::default()
    }));
    host.message(Msg::Signal(signal_hook::consts::SIGTERM));
    assert!(!host.finished(), "waits for the audio flush");
    let exit_at = host.ctx.exit_at.expect("an exit is scheduled");
    assert!(exit_at <= Instant::now() + SHUTDOWN_GRACE);
}

#[test]
fn a_signal_exits_even_if_the_audio_worker_never_stops() {
    let runtime = runtime();
    let mut host = host_without_watchdog(&runtime);
    host.push(Box::new(Probe {
        session: true,
        stops: true,
        ..Probe::default()
    }));
    host.message(Msg::Signal(signal_hook::consts::SIGHUP));
    host.ctx.exit_at = Some(Instant::now());
    assert!(
        host.finished(),
        "the grace period ends the app without the worker"
    );
}

#[test]
fn a_signal_with_no_session_exits_at_once() {
    let runtime = runtime();
    let mut host = host_without_watchdog(&runtime);
    host.push(Box::new(Probe::default()));
    host.message(Msg::Signal(signal_hook::consts::SIGTERM));
    assert!(host.finished());
}

#[test]
fn the_exit_watchdog_is_armed_only_once() {
    let armed = AtomicBool::new(true);
    // Already armed: no second watchdog thread starts, so this returns at once.
    arm_watchdog_once(&armed, signal_hook::consts::SIGTERM);
    assert!(armed.load(Ordering::SeqCst));
}

#[test]
fn ctrl_c_with_a_modal_open_still_names_the_stop_key() {
    let runtime = runtime();
    let mut host = Host::new("textual-dark", runtime.handle().clone());
    host.push(Box::new(Probe {
        session: true,
        hint: Some("Recording: press s to stop and save first".into()),
        ..Probe::default()
    }));
    host.push(Box::new(Probe {
        modal: true,
        ..Probe::default()
    }));
    press(&mut host, KeyCode::Char('c'), KeyModifiers::CONTROL);
    assert_eq!(
        host.ctx().toast_texts().last().unwrap(),
        "Recording: press s to stop and save first"
    );
}

#[test]
fn a_thread_never_waits_for_the_loop() {
    // AGENTS.md: check threads post their updates and never wait on the event
    // loop. Here nothing pumps the loop while the thread posts.
    let runtime = runtime();
    let mut host = Host::new("textual-dark", runtime.handle().clone());
    host.push(Box::new(Probe::default()));
    let sender = host.ctx().task_sender();
    let (done, finished) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        for value in 0..10_000 {
            sender.post(value);
        }
        let _ = done.send(());
    });
    assert!(
        finished.recv_timeout(Duration::from_secs(5)).is_ok(),
        "the thread finished without the loop running"
    );
    assert_eq!(host.pump(), 10_000);
}

/// Keeps every `String` task result it receives.
struct Collector(Arc<Mutex<Vec<String>>>);

impl Screen for Collector {
    fn render(&mut self, _frame: &mut Frame, _area: Rect, _ctx: &Ctx) {}

    fn key(&mut self, _key: &KeyEvent, _name: &str, _ctx: &mut Ctx) -> Action {
        Action::None
    }

    fn message(&mut self, msg: Msg, _ctx: &mut Ctx) -> (Option<Msg>, Action) {
        if let Msg::Task(payload) = msg
            && let Ok(text) = payload.downcast::<String>()
        {
            self.0.lock().unwrap().push(*text);
        }
        (None, Action::None)
    }
}

fn wait_for_results(host: &mut Host, results: &Mutex<Vec<String>>, count: usize) {
    let deadline = Instant::now() + Duration::from_secs(5);
    while results.lock().unwrap().len() < count && Instant::now() < deadline {
        host.pump();
        std::thread::sleep(Duration::from_millis(5));
    }
}

#[test]
fn a_thread_that_panics_posts_its_fallback_result() {
    let runtime = runtime();
    let mut host = Host::new("textual-dark", runtime.handle().clone());
    let results = Arc::new(Mutex::new(Vec::new()));
    host.push(Box::new(Collector(results.clone())));
    host.ctx().spawn_thread_or(
        "panics",
        || -> String { panic!("thread boom") },
        |message| format!("fallback: {message}"),
    );
    host.ctx()
        .spawn_thread_or("works", || "done".to_string(), |_| "unused".to_string());
    wait_for_results(&mut host, &results, 2);
    let mut seen = results.lock().unwrap().clone();
    seen.sort();
    assert_eq!(seen, ["done", "fallback: thread boom"]);
    assert!(!host.finished());
}
