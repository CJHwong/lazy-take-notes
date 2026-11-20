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
