//! The template builder against a fake model, through a real host.

use std::collections::VecDeque;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use crossterm::event::{KeyCode, KeyModifiers, MouseButton, MouseEvent, MouseEventKind};

use super::*;
use crate::adapters::paths::Paths;
use crate::ports::llm::ChatResponse;
use crate::tui::host::Host;

/// Replies in order; an `Err` reply fails the call, and one of kind `panic!` panics.
struct Scripted(Mutex<VecDeque<Result<String, LlmError>>>);

#[async_trait]
impl LlmClient for Scripted {
    async fn chat(
        &self,
        _model: &str,
        _messages: &[ChatMessage],
    ) -> Result<ChatResponse, LlmError> {
        let next = self
            .0
            .lock()
            .unwrap()
            .pop_front()
            .expect("a scripted reply");
        if let Err(e) = &next
            && e.kind == "panic!"
        {
            panic!("{}", e.message);
        }
        next.map(|content| ChatResponse {
            content,
            prompt_tokens: 1,
        })
    }

    async fn check_connectivity(&self) -> Result<(), String> {
        Ok(())
    }

    async fn check_models(&self, _models: &[String]) -> Vec<String> {
        Vec::new()
    }
}

const TEMPLATE_JSON: &str = r#"{"metadata": {"name": "One on One", "description": "Weekly", "locale": "en"},
"system_prompt": "You take notes.", "digest_user_template": "{new_lines}",
"final_user_template": "{full_transcript}", "recognition_hints": ["1:1"],
"quick_actions": [{"label": "Actions", "description": "List them", "prompt_template": "{digest_markdown}"}]}"#;

fn fenced(text: &str, json: &str) -> String {
    format!("{text}\n```json\n{json}\n```")
}

struct Run {
    host: Host,
    dir: tempfile::TempDir,
    _runtime: tokio::runtime::Runtime,
}

fn start(replies: Vec<Result<String, LlmError>>, base: Option<(SessionTemplate, String)>) -> Run {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(1)
        .enable_all()
        .build()
        .unwrap();
    let dir = tempfile::tempdir().unwrap();
    let paths = Paths::under(dir.path());
    let templates = YamlTemplates::new(&paths);
    let example = templates.load_template("default_en").unwrap();
    let llm: Arc<dyn LlmClient> = Arc::new(Scripted(Mutex::new(replies.into())));
    let setup: Setup = Box::new(move || {
        Ok(BuilderSetup {
            llm: llm.clone(),
            model: "m1".into(),
            provider: "ollama".into(),
            example: example.clone(),
        })
    });
    let mut host = Host::new("textual-dark", runtime.handle().clone());
    host.push(Box::new(TemplateBuilderScreen::new(setup, templates, base)));
    Run {
        host,
        dir,
        _runtime: runtime,
    }
}

impl Run {
    fn press(&mut self, name: &str) {
        let code = match name {
            "enter" => KeyCode::Enter,
            "escape" => KeyCode::Esc,
            "backspace" => KeyCode::Backspace,
            "tab" => KeyCode::Tab,
            other => KeyCode::Char(other.chars().next().unwrap()),
        };
        let modifiers = match name {
            "ctrl+s" => KeyModifiers::CONTROL,
            _ => KeyModifiers::NONE,
        };
        let code = if name == "ctrl+s" {
            KeyCode::Char('s')
        } else {
            code
        };
        self.host.key(&KeyEvent::new(code, modifiers));
        self.settle();
    }

    fn send(&mut self, text: &str) {
        self.host.paste(text);
        self.press("enter");
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

    fn screen(&mut self) -> String {
        let backend = ratatui::backend::TestBackend::new(110, 32);
        let mut terminal = ratatui::Terminal::new(backend).unwrap();
        terminal.draw(|frame| self.host.draw(frame)).unwrap();
        let buffer = terminal.backend().buffer().clone();
        (0..buffer.area.height)
            .map(|y| {
                (0..buffer.area.width)
                    .map(|x| buffer[(x, y)].symbol().to_string())
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// A mouse event at a cell of the last drawn 110x32 screen.
    fn mouse(&mut self, kind: MouseEventKind, column: u16, row: u16) {
        self.screen();
        self.host.mouse(&MouseEvent {
            kind,
            column,
            row,
            modifiers: KeyModifiers::NONE,
        });
        self.settle();
    }

    fn click(&mut self, column: u16, row: u16) {
        self.mouse(MouseEventKind::Down(MouseButton::Left), column, row);
    }

    fn saved(&self, name: &str) -> std::path::PathBuf {
        Paths::under(self.dir.path())
            .user_templates_dir()
            .join(format!("{name}.yaml"))
    }
}

#[test]
fn the_markdown_marks_become_styles_and_brackets_stay_text() {
    let escaped = "**bold** *it* `code` [b]x".replace('[', &OPEN.to_string());
    let lines = tagged_lines(&markdown_marks(&escaped), Style::default());
    let spans: Vec<(String, Modifier)> = lines[0]
        .spans
        .iter()
        .map(|span| (span.content.to_string(), span.style.add_modifier))
        .collect();
    assert_eq!(
        spans,
        vec![
            ("bold".into(), Modifier::BOLD),
            (" ".into(), Modifier::empty()),
            ("it".into(), Modifier::ITALIC),
            (" ".into(), Modifier::empty()),
            ("code".into(), Modifier::BOLD),
            (" [b]x".into(), Modifier::empty()),
        ]
    );
    // A heading is bold; a lone star is text.
    let lines = tagged_lines(&markdown_marks("## Head\n2 * 3"), Style::default());
    assert_eq!(lines[0].spans[0].style.add_modifier, Modifier::BOLD);
    assert_eq!(lines[1].spans[0].content, "2 * 3");
}

#[test]
fn slugify_keeps_word_characters_as_python_does() {
    assert_eq!(slugify("One on One"), "one_on_one");
    assert_eq!(slugify("  __Hi!!  there__ "), "hi_there");
    assert_eq!(slugify("會議 記錄"), "會議_記錄");
    assert_eq!(slugify("!!!"), "custom_template");
}

#[test]
fn the_preview_lists_the_fields_like_python() {
    let template: SessionTemplate = serde_json::from_str(TEMPLATE_JSON).unwrap();
    assert_eq!(
        preview_markdown(&template),
        "## One on One\n\n> Weekly\n\n**Locale:** `en`\n\n### Quick Actions\n- **`1`** Actions — List them\n\n**Recognition hints:** 1:1\n\n---\n\n### System Prompt\n\nYou take notes.\n\n---\n\n### Digest Template\n\n{new_lines}"
    );
    let mut bare = template;
    bare.metadata = Default::default();
    bare.quick_actions.clear();
    bare.recognition_hints.clear();
    assert!(preview_markdown(&bare).starts_with("## Untitled\n\n\n\n**Locale:** `not set`\n\n---"));
}

#[test]
fn a_missing_model_names_the_fix_for_its_provider() {
    let ollama = LlmError::new("ResponseError", "model 'm1' not found (status code: 404)");
    assert_eq!(
        error_text(&ollama, "m1", "ollama"),
        "The AI model m1 is not available. Run: ollama pull m1"
    );
    let openai = LlmError::new("NotFoundError", "nope");
    assert_eq!(
        error_text(&openai, "m1", "openai"),
        "The AI model m1 is not available. Pick an available model in Settings > Models."
    );
    let other = LlmError::new("APIConnectionError", "Connection error.");
    assert_eq!(
        error_text(&other, "m1", "openai"),
        "Error: APIConnectionError: Connection error."
    );
}

#[test]
fn a_template_after_one_auto_fix_shows_and_saves_under_its_name() {
    let broken = TEMPLATE_JSON.replace("{new_lines}", "{bogus}");
    let mut run = start(
        vec![
            Ok(fenced("Here.", &broken)),
            Ok(fenced("Fixed.", TEMPLATE_JSON)),
        ],
        None,
    );
    run.send("a 1on1 template");
    let screen = run.screen();
    assert!(screen.contains("You: a 1on1 template"), "{screen}");
    assert!(screen.contains("AI: Fixed."), "{screen}");
    assert!(screen.contains("One on One"), "{screen}");
    // Ctrl+S offers the slug; Enter saves it.
    run.press("ctrl+s");
    assert!(run.screen().contains("Save Template"));
    run.press("enter");
    let saved = std::fs::read_to_string(run.saved("one_on_one")).unwrap();
    assert!(saved.contains("name: One on One"), "{saved}");
    let toasts = run.host.ctx().toast_texts();
    assert!(
        toasts[0].starts_with("Saved as one_on_one → "),
        "{toasts:?}"
    );
    // Saved: Escape leaves at once.
    run.press("escape");
    assert!(run.host.finished());
}

#[test]
fn typing_over_the_selected_name_replaces_it_and_the_display_name_renames() {
    let mut run = start(vec![Ok(fenced("Done.", TEMPLATE_JSON))], None);
    run.send("go");
    run.press("ctrl+s");
    run.press("x");
    run.press("tab");
    run.host.paste("Renamed");
    run.press("enter");
    let saved = std::fs::read_to_string(run.saved("x")).unwrap();
    assert!(saved.contains("name: Renamed"), "{saved}");
    assert!(run.screen().contains("## Renamed") || run.screen().contains("Renamed"));
}

#[test]
fn an_unsaved_template_asks_before_quitting() {
    let mut run = start(vec![Ok(fenced("Done.", TEMPLATE_JSON))], None);
    run.send("go");
    run.press("escape");
    assert!(
        run.screen()
            .contains("You have an unsaved template. Quit anyway?")
    );
    run.press("n");
    assert!(!run.host.finished());
    run.press("escape");
    run.press("y");
    assert!(run.host.finished());
}

#[test]
fn nothing_to_save_warns_and_a_failed_call_leaves_the_builder_usable() {
    let mut run = start(
        vec![
            Err(LlmError::new(
                "ResponseError",
                "model 'm1' not found (status code: 404)",
            )),
            Ok("Which language?".into()),
        ],
        None,
    );
    run.press("ctrl+s");
    assert_eq!(
        run.host.ctx().toast_texts(),
        vec!["No template to save — generate one first.".to_string()]
    );
    run.send("first");
    let screen = run.screen();
    assert!(
        screen.contains("AI: The AI model m1 is not available."),
        "{screen}"
    );
    assert!(
        screen.contains("Run: ollama") && screen.contains("pull m1"),
        "{screen}"
    );
    run.send("second");
    assert!(run.screen().contains("AI: Which language?"));
    // Nothing was generated, so Escape needs no confirm.
    run.press("escape");
    assert!(run.host.finished());
}

#[test]
fn edit_mode_reports_the_changed_fields_and_saves_back_under_its_key() {
    let base: SessionTemplate = serde_json::from_str(TEMPLATE_JSON).unwrap();
    let edited = TEMPLATE_JSON.replace("You take notes.", "You take short notes.");
    let mut run = start(
        vec![Ok(fenced("Shorter.", &edited))],
        Some((base, "mine".into())),
    );
    let screen = run.screen();
    assert!(
        screen.contains("Template Builder: Editing: One on One"),
        "{screen}"
    );
    assert!(screen.contains("Describe the change..."));
    // Nothing changed yet: Escape leaves without a confirm.
    run.send("shorter");
    assert!(run.screen().contains("AI: Changed: system prompt"));
    run.press("ctrl+s");
    run.press("enter");
    assert!(run.saved("mine").exists());
}

#[test]
fn a_setup_error_shows_and_the_next_send_tries_again() {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(1)
        .enable_all()
        .build()
        .unwrap();
    let dir = tempfile::tempdir().unwrap();
    let tries = Arc::new(Mutex::new(0));
    let seen = tries.clone();
    let setup: Setup = Box::new(move || {
        *seen.lock().unwrap() += 1;
        Err("Unknown LLM provider 'x'".into())
    });
    let mut host = Host::new("textual-dark", runtime.handle().clone());
    host.push(Box::new(TemplateBuilderScreen::new(
        setup,
        YamlTemplates::new(&Paths::under(dir.path())),
        None,
    )));
    let mut run = Run {
        host,
        dir,
        _runtime: runtime,
    };
    run.send("one");
    run.send("two");
    assert_eq!(*tries.lock().unwrap(), 2);
    assert!(run.screen().contains("AI: Error: Unknown LLM provider 'x'"));
}

#[test]
fn a_panicking_call_shows_an_error_and_the_builder_still_works() {
    let mut run = start(
        vec![
            Err(LlmError::new("panic!", "client bug")),
            Ok("Which language?".into()),
        ],
        None,
    );
    run.send("first");
    let screen = run.screen();
    assert!(screen.contains("client bug"), "{screen}");
    run.send("second");
    assert!(run.screen().contains("AI: Which language?"));
}

#[test]
fn a_click_moves_the_focus_between_the_chat_and_the_input() {
    let mut run = start(Vec::new(), None);
    // The chat fills the left column above the 3-row input (rows 28 to 30).
    run.click(10, 5);
    run.host.paste("typed-in-chat");
    assert!(!run.screen().contains("typed-in-chat"));
    run.click(10, 29);
    run.host.paste("typed-in-input");
    assert!(run.screen().contains("typed-in-input"));
}

#[test]
fn the_wheel_scrolls_the_chat_under_the_pointer() {
    let lines: Vec<String> = (0..60).map(|n| format!("reply line {n}")).collect();
    let mut run = start(vec![Ok(lines.join("\n"))], None);
    run.send("make it");
    let chat = |screen: &str| -> Vec<String> {
        screen
            .lines()
            .skip(1)
            .take(27)
            .map(|line| line.chars().take(55).collect())
            .collect()
    };
    let at_end = chat(&run.screen());
    assert!(at_end.iter().any(|line| line.contains("reply line 59")));
    run.mouse(MouseEventKind::ScrollUp, 10, 5);
    let scrolled = chat(&run.screen());
    assert_ne!(scrolled, at_end);
    // The wheel over the preview leaves the chat alone.
    run.mouse(MouseEventKind::ScrollDown, 80, 5);
    assert_eq!(chat(&run.screen()), scrolled);
    run.mouse(MouseEventKind::ScrollDown, 10, 5);
    assert_eq!(chat(&run.screen()), at_end);
}
