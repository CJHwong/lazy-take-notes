//! The view screen with a saved session that has speakers.

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers, MouseButton, MouseEvent, MouseEventKind};
use lazy_take_notes::tui::host::Host;
use lazy_take_notes::tui::screens::view::ViewScreen;

struct Viewer {
    host: Host,
    _runtime: tokio::runtime::Runtime,
}

impl Viewer {
    fn open(dir: &std::path::Path) -> Self {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let mut host = Host::new("textual-dark", runtime.handle().clone());
        host.push(Box::new(ViewScreen::new(dir.to_path_buf())));
        Self {
            host,
            _runtime: runtime,
        }
    }

    fn screen(&mut self) -> String {
        let mut terminal =
            ratatui::Terminal::new(ratatui::backend::TestBackend::new(120, 30)).unwrap();
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

    fn click(&mut self, text: &str) {
        let screen = self.screen();
        let (row, line) = screen
            .lines()
            .enumerate()
            .find(|(_, line)| line.contains(text))
            .unwrap_or_else(|| panic!("{text:?} is not on screen:\n{screen}"));
        let column = line[..line.find(text).unwrap()].chars().count() as u16;
        self.host.mouse(&MouseEvent {
            kind: MouseEventKind::Down(MouseButton::Left),
            column,
            row: row as u16,
            modifiers: KeyModifiers::NONE,
        });
    }

    /// The background of the last cell of the line that holds `text`.
    fn background_of(&mut self, text: &str) -> ratatui::style::Color {
        let mut terminal =
            ratatui::Terminal::new(ratatui::backend::TestBackend::new(120, 30)).unwrap();
        terminal.draw(|frame| self.host.draw(frame)).unwrap();
        let buffer = terminal.backend().buffer();
        for y in 0..buffer.area.height {
            let row: String = (0..buffer.area.width)
                .map(|x| buffer[(x, y)].symbol())
                .collect();
            if let Some(at) = row.find(text) {
                let x = (row[..at].chars().count() + text.chars().count() - 1) as u16;
                return buffer[(x, y)].bg;
            }
        }
        panic!("{text:?} is not on screen");
    }

    fn press(&mut self, code: KeyCode) {
        self.host.key(&KeyEvent::new(code, KeyModifiers::NONE));
    }

    /// True when the first cell of `text` is underlined.
    fn underlined(&mut self, text: &str) -> bool {
        let mut terminal =
            ratatui::Terminal::new(ratatui::backend::TestBackend::new(120, 30)).unwrap();
        terminal.draw(|frame| self.host.draw(frame)).unwrap();
        let buffer = terminal.backend().buffer();
        for y in 0..buffer.area.height {
            let row: String = (0..buffer.area.width)
                .map(|x| buffer[(x, y)].symbol())
                .collect();
            if let Some(at) = row.find(text) {
                let x = row[..at].chars().count() as u16;
                return buffer[(x, y)]
                    .modifier
                    .contains(ratatui::style::Modifier::UNDERLINED);
            }
        }
        panic!("{text:?} is not on screen");
    }

    fn type_text(&mut self, text: &str) {
        for ch in text.chars() {
            self.host
                .key(&KeyEvent::new(KeyCode::Char(ch), KeyModifiers::NONE));
        }
        self.host
            .key(&KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
    }
}

fn saved_session() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join("transcript.txt"),
        "[00:00:00] hello\n[00:00:05] hi\n",
    )
    .unwrap();
    std::fs::write(
        dir.path().join("transcript_speakers.txt"),
        "[00:00:00] S1: hello\n[00:00:05] S2: hi\n",
    )
    .unwrap();
    std::fs::write(
        dir.path().join("speakers.json"),
        r#"{"names": {}, "turns": [{"start": 0.0, "end": 4.0, "speaker": 1}, {"start": 5.0, "end": 6.0, "speaker": 2}]}"#,
    )
    .unwrap();
    dir
}

#[test]
fn the_view_shows_the_labels_and_a_chip_click_renames_a_speaker() {
    let dir = saved_session();
    let mut viewer = Viewer::open(dir.path());
    let screen = viewer.screen();
    assert!(screen.contains("[00:00:05] S2: hi"), "{screen}");
    assert!(
        screen.contains(" S1 4s ") && screen.contains(" S2 1s "),
        "{screen}"
    );
    // The first click highlights S2's lines and nobody else's.
    viewer.click(" S2 1s ");
    let plain = viewer.background_of("[00:00:00] S1: hello");
    assert_ne!(viewer.background_of("[00:00:05] S2: hi"), plain);
    assert!(!viewer.screen().contains("Name for S2"));
    // A second click on it opens the name prompt.
    viewer.click(" S2 1s ");
    assert!(
        viewer.screen().contains("Name for S2"),
        "{}",
        viewer.screen()
    );
    viewer.type_text("Andy");
    let screen = viewer.screen();
    assert!(screen.contains("[00:00:05] Andy: hi"), "{screen}");
    assert!(screen.contains(" Andy 1s "), "{screen}");
    // The renamed lines stay highlighted; a click in the transcript clears it.
    assert_ne!(viewer.background_of("[00:00:05] Andy: hi"), plain);
    viewer.click("[00:00:00] S1: hello");
    assert_eq!(viewer.background_of("[00:00:05] Andy: hi"), plain);
    let labeled = std::fs::read_to_string(dir.path().join("transcript_speakers.txt")).unwrap();
    assert_eq!(labeled, "[00:00:00] S1: hello\n[00:00:05] Andy: hi\n");
    let json = std::fs::read_to_string(dir.path().join("speakers.json")).unwrap();
    assert!(
        json.contains("\"2\": \"Andy\"") && json.contains("\"turns\""),
        "{json}"
    );
}

#[test]
fn a_session_without_speakers_shows_the_plain_transcript_and_no_chips() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("transcript.txt"), "[00:00:00] hello\n").unwrap();
    let mut viewer = Viewer::open(dir.path());
    let screen = viewer.screen();
    assert!(screen.contains("[00:00:00] hello"), "{screen}");
    assert!(!screen.contains(" S1 "), "{screen}");
}

#[test]
fn the_keyboard_chooses_and_renames_a_speaker() {
    let dir = saved_session();
    let mut viewer = Viewer::open(dir.path());
    let plain = viewer.background_of("[00:00:05] S2: hi");
    // Tab reaches the bar and selects the first chip.
    viewer.press(KeyCode::Tab);
    assert_ne!(viewer.background_of("[00:00:00] S1: hello"), plain);
    assert_eq!(viewer.background_of("[00:00:05] S2: hi"), plain);
    // Right moves to S2 and stops there; Left goes back.
    viewer.press(KeyCode::Right);
    viewer.press(KeyCode::Right);
    assert_ne!(viewer.background_of("[00:00:05] S2: hi"), plain);
    assert_eq!(viewer.background_of("[00:00:00] S1: hello"), plain);
    viewer.press(KeyCode::Left);
    viewer.press(KeyCode::Enter);
    assert!(
        viewer.screen().contains("Name for S1"),
        "{}",
        viewer.screen()
    );
    viewer.type_text("Bea");
    assert!(
        viewer.screen().contains("[00:00:00] Bea: hello"),
        "{}",
        viewer.screen()
    );
    // Escape drops the selection; Enter then opens nothing.
    viewer.press(KeyCode::Esc);
    assert_eq!(viewer.background_of("[00:00:00] Bea: hello"), plain);
    viewer.press(KeyCode::Enter);
    assert!(!viewer.screen().contains("Name for"), "{}", viewer.screen());
}

#[test]
fn the_selected_chip_is_underlined_only_while_the_bar_has_the_focus() {
    let dir = saved_session();
    let mut viewer = Viewer::open(dir.path());
    viewer.press(KeyCode::Tab);
    assert!(viewer.underlined(" S1 4s "));
    // The digest takes the focus; the chip stays selected but loses the underline.
    viewer.press(KeyCode::Tab);
    assert!(!viewer.underlined(" S1 4s "));
    // Shift+Tab comes back to the bar.
    viewer.press(KeyCode::BackTab);
    assert!(viewer.underlined(" S1 4s "));
}
