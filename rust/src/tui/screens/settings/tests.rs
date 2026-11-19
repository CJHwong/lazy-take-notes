//! The settings form against Python's: the data at open and the file after Save.
//! `scripts/gen_settings_fixtures.py` records the Python side.

use serde_json::Value as Json;

use super::models::LanguageEditor;
use super::*;
use crate::tui::host::Host;

const FIXTURES: &str = include_str!("../../../../tests/fixtures/settings.json");
/// The fixture script creates this file; the custom path case needs it to exist.
const CUSTOM_MODEL: &str = "/tmp/ltn-settings-fixture/model.bin";

fn to_json(value: &Value) -> Json {
    serde_json::to_value(value).expect("YAML data converts to JSON")
}

fn host() -> (Host, tokio::runtime::Runtime) {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let host = Host::new("textual-dark", runtime.handle().clone());
    (host, runtime)
}

fn screen_for(root: &Path, config: Option<&str>) -> SettingsScreen {
    let paths = Paths::under(root);
    if let Some(text) = config {
        std::fs::create_dir_all(&paths.config_dir).unwrap();
        std::fs::write(paths.config_dir.join("config.yaml"), text).unwrap();
    }
    let store = Arc::new(CachedModelStore::new(&paths));
    SettingsScreen::with(paths, None, store, &[])
}

fn apply(screen: &mut SettingsScreen, edit: &[Json], ctx: &mut Ctx) {
    let kind = edit[0].as_str().unwrap();
    let id = edit[1].as_str().unwrap();
    let page = screen
        .pages
        .iter()
        .position(|page| page.item(id).is_some() || page.items.iter().any(|item| matches!(&item.kind, Kind::Buttons { buttons, .. } if buttons.iter().any(|b| b.id == id))))
        .unwrap_or(0);
    let page = &mut screen.pages[page];
    match kind {
        "input" => page.set_input(id, edit[2].as_str().unwrap()),
        "switch" => page.set_switch(id, edit[2].as_bool().unwrap()),
        "select" => page
            .select_mut(id)
            .unwrap()
            .set_value(edit[2].as_str().unwrap()),
        "textarea" => page
            .text_area_mut(id)
            .unwrap()
            .set_text(edit[2].as_str().unwrap()),
        "press" => {
            screen.pressed(id, ctx);
        }
        "remove_language" => screen.pages[0]
            .custom::<LanguageEditor>("models-language")
            .unwrap()
            .remove(id),
        other => panic!("unknown edit {other}"),
    }
}

#[test]
fn the_form_reads_and_saves_config_yaml_as_python_does() {
    std::fs::create_dir_all(Path::new(CUSTOM_MODEL).parent().unwrap()).unwrap();
    std::fs::write(CUSTOM_MODEL, b"").unwrap();
    let cases: Vec<Json> = serde_json::from_str(FIXTURES).unwrap();
    for case in cases {
        let name = case["name"].as_str().unwrap();
        let root = tempfile::tempdir().unwrap();
        let (mut host, _runtime) = host();
        let ctx = host.ctx();
        let mut screen = screen_for(root.path(), case["config"].as_str());
        screen.mounted(ctx);
        let opened = to_json(&screen.collect());
        assert_eq!(opened, case["opened"], "{name}: the data at open");
        for edit in case["edits"].as_array().unwrap() {
            apply(&mut screen, edit.as_array().unwrap(), ctx);
        }
        assert_eq!(
            screen.has_unsaved_changes(),
            case["unsaved_before_save"].as_bool().unwrap(),
            "{name}: unsaved changes"
        );
        screen.save(ctx);
        let path = root.path().join("config").join("config.yaml");
        let saved = std::fs::read_to_string(&path).ok();
        match case["saved_file"].as_str() {
            None => assert!(
                saved.is_none() || case["config"].is_string(),
                "{name}: nothing is saved"
            ),
            Some(expected) if name == "unreadable_yaml" => {
                // Save refuses a file it cannot read, so the file stays as it was.
                assert_eq!(
                    saved.as_deref(),
                    Some(expected),
                    "{name}: the file is untouched"
                );
            }
            Some(expected) => {
                let saved = saved.unwrap_or_else(|| panic!("{name}: no file saved"));
                let ours: Value = yaml_serde::from_str(&saved).unwrap();
                let theirs: Value = yaml_serde::from_str(expected).unwrap();
                assert_eq!(to_json(&ours), to_json(&theirs), "{name}: the saved file");
                // The key order too: Python writes the form's order.
                let keys = |value: &Value| -> Vec<String> {
                    value
                        .as_mapping()
                        .unwrap()
                        .keys()
                        .map(|k| k.as_str().unwrap().to_string())
                        .collect()
                };
                assert_eq!(keys(&ours), keys(&theirs), "{name}: the key order");
            }
        }
        // The toasts, with the temporary path out of the way.
        let ours: Vec<String> = ctx
            .toast_texts()
            .into_iter()
            .map(|text| text.replace(&path.display().to_string(), "<path>"))
            .collect();
        let theirs: Vec<String> = case["toasts"]
            .as_array()
            .unwrap()
            .iter()
            .map(|text| {
                let text = text.as_str().unwrap();
                match text.strip_prefix("Saved to ") {
                    Some(_) => "Saved to <path>".to_string(),
                    None => text.to_string(),
                }
            })
            .collect();
        if name == "unreadable_yaml" {
            // The parser's own words differ; the sentences around them do not.
            assert!(
                ours[0].starts_with("config.yaml cannot be read ("),
                "{ours:?}"
            );
            assert!(ours[0].ends_with("). Showing defaults. Press [e] to fix the file."));
            assert!(ours[1].starts_with("Cannot save config: "), "{ours:?}");
        } else {
            assert_eq!(ours, theirs, "{name}: the toasts");
        }
    }
}

// --- The mouse ---

use crossterm::event::{KeyCode, KeyModifiers, MouseButton, MouseEvent, MouseEventKind};
use ratatui::Terminal;
use ratatui::backend::TestBackend;

/// Settings on a host, drawn on a test terminal, driven by real mouse events.
struct Driven {
    host: Host,
    terminal: Terminal<TestBackend>,
    _root: tempfile::TempDir,
    _runtime: tokio::runtime::Runtime,
}

impl Driven {
    fn open(width: u16, height: u16) -> Self {
        let root = tempfile::tempdir().unwrap();
        let (mut host, runtime) = host();
        host.push(Box::new(screen_for(root.path(), None)));
        let mut driven = Self {
            host,
            terminal: Terminal::new(TestBackend::new(width, height)).unwrap(),
            _root: root,
            _runtime: runtime,
        };
        driven.draw();
        driven
    }

    fn draw(&mut self) {
        let host = &mut self.host;
        self.terminal.draw(|frame| host.draw(frame)).unwrap();
    }

    fn rows(&self) -> Vec<String> {
        let buffer = self.terminal.backend().buffer();
        (0..buffer.area.height)
            .map(|y| {
                (0..buffer.area.width)
                    .map(|x| buffer[(x, y)].symbol().to_string())
                    .collect()
            })
            .collect()
    }

    /// The cell where `text` first shows.
    fn find(&self, text: &str) -> (u16, u16) {
        let wanted: Vec<char> = text.chars().collect();
        for (y, row) in self.rows().iter().enumerate() {
            let cells: Vec<char> = row.chars().collect();
            let last = cells.len().saturating_sub(wanted.len());
            if let Some(x) = (0..=last).find(|&x| cells[x..x + wanted.len()] == wanted[..]) {
                return (x as u16, y as u16);
            }
        }
        panic!("{text:?} is not on screen:\n{}", self.rows().join("\n"));
    }

    fn shows(&self, text: &str) -> bool {
        self.rows().iter().any(|row| row.contains(text))
    }

    fn mouse(&mut self, kind: MouseEventKind, (column, row): (u16, u16)) {
        self.host.mouse(&MouseEvent {
            kind,
            column,
            row,
            modifiers: KeyModifiers::NONE,
        });
        self.draw();
    }

    fn click(&mut self, text: &str) {
        let at = self.find(text);
        self.mouse(MouseEventKind::Down(MouseButton::Left), at);
    }

    fn press(&mut self, code: KeyCode) {
        self.host.key(&KeyEvent::new(code, KeyModifiers::NONE));
        self.draw();
    }
}

#[test]
fn a_click_on_a_page_row_opens_that_page() {
    let mut driven = Driven::open(100, 30);
    assert!(driven.rows()[0].contains("─ Models "));
    driven.click("Digest");
    assert!(
        driven.rows()[0].contains("─ Digest "),
        "{}",
        driven.rows()[0]
    );
    assert!(driven.shows("When to summarize"));
}

#[test]
fn a_click_opens_a_select_and_a_click_on_an_option_picks_it() {
    let mut driven = Driven::open(100, 30);
    driven.click("Appearance");
    driven.click("textual-dark");
    // The wheel moves the list's highlight down past its first screen, so the list scrolls.
    let list = driven.find("nord");
    for _ in 0..12 {
        driven.mouse(MouseEventKind::ScrollDown, list);
    }
    driven.click("dracula");
    assert!(driven.shows("│ dracula "), "{}", driven.rows().join("\n"));
    assert!(!driven.shows("tokyo-night"), "the list closed");
}

#[test]
fn the_wheel_scrolls_the_page_under_the_pointer() {
    let mut driven = Driven::open(100, 20);
    assert!(driven.rows()[1].contains("Text generation"));
    // Over the page list, the wheel leaves the page where it is.
    let nav = driven.find("Digest");
    driven.mouse(MouseEventKind::ScrollDown, nav);
    assert!(driven.rows()[1].contains("Text generation"));
    let page = driven.find("Provider");
    for _ in 0..3 {
        driven.mouse(MouseEventKind::ScrollDown, page);
    }
    assert!(!driven.rows()[1].contains("Text generation"));
}

#[test]
fn a_click_focuses_a_field_and_the_dialog_buttons_answer_clicks() {
    let mut driven = Driven::open(100, 30);
    driven.click("http://localhost");
    driven.press(KeyCode::End);
    driven.press(KeyCode::Char('x'));
    assert!(driven.shows("11434x"));
    // The first Esc leaves the page for the page list; the second asks.
    driven.press(KeyCode::Esc);
    driven.press(KeyCode::Esc);
    assert!(driven.shows("Unsaved changes"));
    driven.click("Keep editing");
    assert!(!driven.shows("Unsaved changes"));
    driven.press(KeyCode::Esc);
    driven.click("Discard and exit");
    assert!(!driven.shows("Settings"));
}

#[test]
fn a_click_on_a_switch_turns_it_on() {
    let mut driven = Driven::open(100, 30);
    driven.click("Audio");
    let page = driven.find("Microphone");
    for _ in 0..30 {
        driven.mouse(MouseEventKind::ScrollDown, page);
    }
    let (_, row) = driven.find("Label speakers");
    let was = driven.switch_color();
    // The switch sits in the control column, after the 24-column label.
    let (label_x, _) = driven.find("Label speakers");
    driven.mouse(
        MouseEventKind::Down(MouseButton::Left),
        (label_x + super::form::LABEL_WIDTH + 2, row),
    );
    assert_ne!(driven.switch_color(), was);
}

impl Driven {
    /// The thumb's cell color of the speaker switch: its right half is the
    /// success color while on.
    fn switch_color(&self) -> ratatui::style::Color {
        let (x, y) = self.find("Label speakers");
        let buffer = self.terminal.backend().buffer();
        buffer[(x + super::form::LABEL_WIDTH + 4, y)].bg
    }
}
