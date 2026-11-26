//! The template builder runtime against a fake model, with no screen.

use std::collections::VecDeque;
use std::sync::Mutex;
use std::sync::mpsc::{Receiver, channel};

use async_trait::async_trait;

use super::*;
use crate::ports::llm::ChatResponse;

/// Replies in order.
struct Scripted(Mutex<VecDeque<String>>);

#[async_trait]
impl LlmClient for Scripted {
    async fn chat(
        &self,
        _model: &str,
        _messages: &[ChatMessage],
    ) -> Result<ChatResponse, LlmError> {
        let content = self
            .0
            .lock()
            .unwrap()
            .pop_front()
            .expect("a scripted reply");
        Ok(ChatResponse {
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

/// A fake front end: task results land in `results`.
struct Front {
    tasks: Tasks,
    results: Receiver<Payload>,
    dir: tempfile::TempDir,
    _runtime: tokio::runtime::Runtime,
}

impl Front {
    fn new() -> Self {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .enable_all()
            .build()
            .unwrap();
        let (sender, results) = channel();
        let sender = Mutex::new(sender);
        let tasks = Tasks::new(
            runtime.handle().clone(),
            Arc::new(move |payload| {
                let _ = sender.lock().unwrap().send(payload);
            }),
            Arc::new(|_| true),
        );
        Self {
            tasks,
            results,
            dir: tempfile::tempdir().unwrap(),
            _runtime: runtime,
        }
    }

    fn templates(&self) -> YamlTemplates {
        YamlTemplates::new(&Paths::under(self.dir.path()))
    }

    fn builder(
        &self,
        replies: Vec<String>,
        base: Option<(SessionTemplate, String)>,
    ) -> TemplateBuilderRuntime {
        let example = self.templates().load_template("default_en").unwrap();
        let llm: Arc<dyn LlmClient> = Arc::new(Scripted(Mutex::new(replies.into())));
        let setup: Setup = Box::new(move || {
            Ok(BuilderSetup {
                llm: llm.clone(),
                model: "m1".into(),
                provider: "ollama".into(),
                example: example.clone(),
            })
        });
        TemplateBuilderRuntime::new(setup, self.templates(), base)
    }

    /// Send `text` and hand the reply back to the runtime.
    fn turn(&self, builder: &mut TemplateBuilderRuntime, text: &str) {
        assert!(builder.send(text.into(), &self.tasks));
        assert!(builder.busy());
        let payload = self
            .results
            .recv_timeout(std::time::Duration::from_secs(5))
            .expect("a reply");
        assert!(builder.task_result(payload).is_ok());
        assert!(!builder.busy());
    }
}

fn last_ai(builder: &TemplateBuilderRuntime) -> &str {
    let (role, text) = builder.chat().last().expect("a message");
    assert_eq!(*role, Role::Ai);
    text
}

#[test]
fn a_new_template_shows_and_saves_under_its_slug() {
    let front = Front::new();
    let mut builder = front.builder(vec![fenced("Here.", TEMPLATE_JSON)], None);
    assert_eq!(builder.title(), "Template Builder");
    assert_eq!(builder.preview(), NO_TEMPLATE);
    assert!(builder.save_suggestion().is_none());
    assert!(!builder.unsaved());
    let revision = builder.preview_revision();
    front.turn(&mut builder, "a 1on1 template");
    assert_eq!(builder.chat()[1], (Role::User, "a 1on1 template".into()));
    assert_eq!(last_ai(&builder), "Here.");
    assert_ne!(builder.preview_revision(), revision);
    assert!(builder.preview().starts_with("## One on One"));
    assert!(builder.unsaved());
    let suggestion = builder.save_suggestion().unwrap();
    assert_eq!(suggestion.file_name, "one_on_one");
    assert_eq!(suggestion.display_name, "One on One");
    let (name, path) = builder.save("My File", "Renamed").unwrap().unwrap();
    assert_eq!(name, "my_file");
    assert!(
        std::fs::read_to_string(path)
            .unwrap()
            .contains("name: Renamed")
    );
    assert!(builder.preview().starts_with("## Renamed"));
    assert!(!builder.unsaved());
}

#[test]
fn edit_mode_lists_the_changed_fields_and_saves_under_its_key() {
    let front = Front::new();
    let base: SessionTemplate = serde_json::from_str(TEMPLATE_JSON).unwrap();
    let edited = TEMPLATE_JSON.replace("You take notes.", "You take short notes.");
    let mut builder = front.builder(
        vec![fenced("Shorter.", &edited)],
        Some((base, "mine".into())),
    );
    assert_eq!(builder.title(), "Template Builder: Editing: One on One");
    assert_eq!(builder.placeholder(), "Describe the change...");
    assert!(!builder.unsaved(), "nothing changed yet");
    front.turn(&mut builder, "shorter");
    assert_eq!(last_ai(&builder), "Changed: system prompt");
    assert_eq!(builder.save_suggestion().unwrap().file_name, "mine");
}

#[test]
fn a_setup_error_answers_at_once_and_the_next_send_tries_again() {
    let front = Front::new();
    let tries = Arc::new(Mutex::new(0));
    let seen = tries.clone();
    let setup: Setup = Box::new(move || {
        *seen.lock().unwrap() += 1;
        Err("Unknown LLM provider 'x'".into())
    });
    let mut builder = TemplateBuilderRuntime::new(setup, front.templates(), None);
    assert!(builder.send("one".into(), &front.tasks));
    assert!(!builder.busy());
    assert!(builder.send("two".into(), &front.tasks));
    assert_eq!(*tries.lock().unwrap(), 2);
    assert_eq!(last_ai(&builder), "Error: Unknown LLM provider 'x'");
}

#[test]
fn a_second_send_waits_and_other_results_are_given_back() {
    let front = Front::new();
    let mut builder = front.builder(vec!["Which language?".into()], None);
    assert!(builder.send("first".into(), &front.tasks));
    assert!(!builder.send("again".into(), &front.tasks));
    assert!(builder.task_result(Box::new(7_u8)).is_err());
    builder.leave();
}
