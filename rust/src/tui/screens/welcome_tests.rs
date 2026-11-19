//! The welcome picker's keys.

use crossterm::event::{KeyCode, KeyModifiers};

use super::*;
use crate::tui::host::Host;

fn press(host: &mut Host, code: KeyCode) {
    host.key(&KeyEvent::new(code, KeyModifiers::NONE));
}

fn chosen(host: &mut Host) -> Option<&'static str> {
    host.take_result()
        .and_then(|payload| payload.downcast::<ChosenMode>().ok())
        .map(|mode| mode.0)
}

fn host_with(last_mode: Option<&str>) -> (Host, tokio::runtime::Runtime) {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let mut host = Host::new("textual-dark", runtime.handle().clone());
    host.push(Box::new(WelcomePicker::new(last_mode)));
    (host, runtime)
}

#[test]
fn enter_picks_the_highlighted_mode_and_the_last_mode_keeps_its_place() {
    let (mut host, _runtime) = host_with(None);
    press(&mut host, KeyCode::Down);
    press(&mut host, KeyCode::Down);
    press(&mut host, KeyCode::Enter);
    assert_eq!(chosen(&mut host), Some("view"));
    let (mut host, _runtime) = host_with(Some("create-template"));
    press(&mut host, KeyCode::Down);
    press(&mut host, KeyCode::Enter);
    assert_eq!(chosen(&mut host), Some("config"));
}

#[test]
fn escape_and_q_cancel() {
    for code in [KeyCode::Esc, KeyCode::Char('q')] {
        let (mut host, _runtime) = host_with(None);
        press(&mut host, code);
        assert!(host.finished());
        assert_eq!(chosen(&mut host), None);
    }
}
