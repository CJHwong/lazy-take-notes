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
