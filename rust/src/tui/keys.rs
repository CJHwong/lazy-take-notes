//! Keys by Textual's names, and the binding tables each screen declares.
//!
//! The tables are part of the data contract. `tests/fixtures/bindings.json`
//! holds the Python tables; a test checks each Rust table against it.

use crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyModifiers};

/// One row of a screen's merged binding table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Binding {
    pub key: &'static str,
    pub action: &'static str,
    pub description: &'static str,
    pub show: bool,
    pub priority: bool,
}

pub const fn bind(key: &'static str, action: &'static str, description: &'static str) -> Binding {
    Binding {
        key,
        action,
        description,
        show: true,
        priority: false,
    }
}

impl Binding {
    pub const fn hidden(self) -> Self {
        Self {
            show: false,
            ..self
        }
    }

    pub const fn priority(self) -> Self {
        Self {
            priority: true,
            ..self
        }
    }
}

/// Every Textual screen has these, before its own.
pub const SCREEN_BASE: [Binding; 4] = [
    bind("tab", "app.focus_next", "Focus Next").hidden(),
    bind("shift+tab", "app.focus_previous", "Focus Previous").hidden(),
    bind("ctrl+c", "screen.copy_text", "Copy selected text").hidden(),
    bind("super+c", "screen.copy_text", "Copy selected text").hidden(),
];

/// The action bound to `key`, if any.
pub fn action_for(table: &[Binding], key: &str) -> Option<&'static str> {
    table
        .iter()
        .find(|binding| binding.key == key)
        .map(|binding| binding.action)
}

/// A key press by Textual's name, or None for a release or repeat-only event.
/// The key a status bar hint names, as in `[Esc] Cancel` or `[1] Summary`.
/// None for a hint that names no single key (`[↑/↓] Navigate`).
pub fn hint_key(hint: &str) -> Option<KeyEvent> {
    let lower = hint.to_lowercase();
    let code = match lower.as_str() {
        "esc" | "escape" => KeyCode::Esc,
        "enter" => KeyCode::Enter,
        "tab" => KeyCode::Tab,
        "space" => KeyCode::Char(' '),
        _ => {
            let mut chars = lower.chars();
            match (chars.next(), chars.next()) {
                (Some(character), None) if !character.is_whitespace() => KeyCode::Char(character),
                _ => return None,
            }
        }
    };
    Some(KeyEvent::new(code, KeyModifiers::NONE))
}

pub fn key_name(event: &KeyEvent) -> Option<String> {
    if event.kind == KeyEventKind::Release {
        return None;
    }
    let base = match event.code {
        KeyCode::Char(' ') => "space".to_string(),
        KeyCode::Char(character) => character.to_lowercase().collect(),
        KeyCode::Enter => "enter".into(),
        KeyCode::Esc => "escape".into(),
        KeyCode::Tab => "tab".into(),
        KeyCode::BackTab => return Some("shift+tab".into()),
        KeyCode::Backspace => "backspace".into(),
        KeyCode::Delete => "delete".into(),
        KeyCode::Up => "up".into(),
        KeyCode::Down => "down".into(),
        KeyCode::Left => "left".into(),
        KeyCode::Right => "right".into(),
        KeyCode::Home => "home".into(),
        KeyCode::End => "end".into(),
        KeyCode::PageUp => "pageup".into(),
        KeyCode::PageDown => "pagedown".into(),
        KeyCode::Insert => "insert".into(),
        KeyCode::F(number) => format!("f{number}"),
        _ => return None,
    };
    let mut name = String::new();
    let modifiers = event.modifiers;
    if modifiers.contains(KeyModifiers::CONTROL) {
        name.push_str("ctrl+");
    }
    if modifiers.contains(KeyModifiers::ALT) {
        name.push_str("alt+");
    }
    if modifiers.contains(KeyModifiers::SUPER) {
        name.push_str("super+");
    }
    // A shifted letter is its capital, as Textual names it: `shift+` only for the rest.
    match event.code {
        KeyCode::Char(character) if modifiers.contains(KeyModifiers::SHIFT) => name.push(character),
        _ if modifiers.contains(KeyModifiers::SHIFT) => {
            name.push_str("shift+");
            name.push_str(&base);
        }
        _ => name.push_str(&base),
    }
    Some(name)
}

/// The printable character a key types into an input, if it types one.
pub fn typed_char(event: &KeyEvent) -> Option<char> {
    let typing = event.modifiers.difference(KeyModifiers::SHIFT).is_empty();
    match event.code {
        KeyCode::Char(character) if typing && event.kind != KeyEventKind::Release => {
            Some(character)
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn press(code: KeyCode, modifiers: KeyModifiers) -> Option<String> {
        key_name(&KeyEvent::new(code, modifiers))
    }

    #[test]
    fn keys_have_textual_names() {
        assert_eq!(
            press(KeyCode::Char('q'), KeyModifiers::NONE).as_deref(),
            Some("q")
        );
        assert_eq!(
            press(KeyCode::Char('q'), KeyModifiers::CONTROL).as_deref(),
            Some("ctrl+q")
        );
        assert_eq!(
            press(KeyCode::Char('A'), KeyModifiers::SHIFT).as_deref(),
            Some("A")
        );
        assert_eq!(
            press(KeyCode::Char(' '), KeyModifiers::NONE).as_deref(),
            Some("space")
        );
        assert_eq!(
            press(KeyCode::BackTab, KeyModifiers::SHIFT).as_deref(),
            Some("shift+tab")
        );
        assert_eq!(
            press(KeyCode::PageUp, KeyModifiers::CONTROL).as_deref(),
            Some("ctrl+pageup")
        );
        assert_eq!(
            press(KeyCode::Esc, KeyModifiers::NONE).as_deref(),
            Some("escape")
        );
    }
}
