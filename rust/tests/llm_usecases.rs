mod common;

use common::{FakeLlm, Reply, built_in_template, segment};
use lazy_take_notes::domain::chat::{ChatMessage, Role};
use lazy_take_notes::domain::speaker_turn::SpeakerTurn;
use lazy_take_notes::domain::template::SessionTemplate;
use lazy_take_notes::ports::llm::LlmError;
use lazy_take_notes::usecases::label::generate_label;
use lazy_take_notes::usecases::quick_action::{QuickActionInput, run_quick_action};
use lazy_take_notes::usecases::speaker_attribution::{
    SPAN_TOLERANCE, speaker_at, speaker_for_span,
};
use lazy_take_notes::usecases::template_builder::TemplateBuilder;
use serde_json::{Value, json};

fn valid() -> Value {
    json!({
        "metadata": {"name": "Test Template", "description": "A test template", "locale": "en"},
        "system_prompt": "You are a meeting assistant.",
        "digest_user_template": "New lines ({line_count}):\n{new_lines}\n{user_context}",
        "final_user_template": "Final ({line_count}):\n{new_lines}\n{user_context}\n{full_transcript}",
        "quick_actions": [{"label": "Summary", "description": "Quick summary", "prompt_template": "Summary:\n{digest_markdown}\n{recent_transcript}"}]
    })
}

fn with(field: &str, value: Value) -> Value {
    let mut data = valid();
    data[field] = value;
    data
}

fn fenced(value: &Value, preamble: &str) -> String {
    format!(
        "{preamble}\n\n```json\n{}\n```",
        serde_json::to_string_pretty(value).unwrap()
    )
}

fn example() -> SessionTemplate {
    serde_json::from_value(valid()).unwrap()
}

mod builder {
    use super::*;

    #[tokio::test]
    async fn a_valid_template_comes_back_with_the_chat_text() {
        let llm = FakeLlm::text(&fenced(&valid(), "Here is your template:"));
        let result = TemplateBuilder::new(&example(), None)
            .generate(&llm, "go", "m", &mut Vec::new())
            .await
            .unwrap();
        assert_eq!(result.template.unwrap().metadata.name, "Test Template");
        assert!(result.validation_errors.is_empty() && result.error.is_empty());
        assert_eq!(result.assistant_message, "Here is your template:");
    }

    #[tokio::test]
    async fn a_question_is_returned_as_chat() {
        let llm = FakeLlm::text("What language should the template be in?");
        let result = TemplateBuilder::new(&example(), None)
            .generate(&llm, "hi", "m", &mut Vec::new())
            .await
            .unwrap();
        assert!(result.template.is_none() && result.error.is_empty());
        assert!(result.assistant_message.contains("language"));
    }

    #[tokio::test]
    async fn bad_json_bad_variables_and_too_many_actions_are_reported() {
        let mut builder = TemplateBuilder::new(&example(), None);
        let llm = FakeLlm::text("```json\n{not valid json\n```");
        let result = builder
            .generate(&llm, "go", "m", &mut Vec::new())
            .await
            .unwrap();
        assert!(result.template.is_none() && result.error.starts_with("Invalid JSON"));

        llm.set(Reply::Text(
            fenced(&with("digest_user_template", json!("Uses {bogus_var}")), ""),
            0,
        ));
        let result = builder
            .generate(&llm, "go", "m", &mut Vec::new())
            .await
            .unwrap();
        assert!(result.template.is_some() && result.validation_errors.contains("bogus_var"));

        let six: Vec<Value> = (0..6)
            .map(|i| json!({"label": format!("A{i}"), "prompt_template": "{digest_markdown}"}))
            .collect();
        llm.set(Reply::Text(
            fenced(&with("quick_actions", json!(six)), ""),
            0,
        ));
        let result = builder
            .generate(&llm, "go", "m", &mut Vec::new())
            .await
            .unwrap();
        assert!(result.template.is_none() && result.error.starts_with("Schema validation failed"));
    }

    #[tokio::test]
    async fn a_failed_call_leaves_the_history_unchanged() {
        let llm = FakeLlm::new(Reply::Fail(LlmError::new(
            "ConnectError",
            "Connection refused",
        )));
        let mut history = vec![ChatMessage::user("hi"), ChatMessage::assistant("hello")];
        assert!(
            TemplateBuilder::new(&example(), None)
                .generate(&llm, "again", "m", &mut history)
                .await
                .is_err()
        );
        assert_eq!(history.len(), 2);
    }

    #[tokio::test]
    async fn the_system_prompt_then_the_history_then_the_new_message_are_sent() {
        let llm = FakeLlm::text("ok");
        let mut history = vec![ChatMessage::user("hi"), ChatMessage::assistant("hello")];
        TemplateBuilder::new(&example(), None)
            .generate(&llm, "next", "m", &mut history)
            .await
            .unwrap();
        let sent = llm.last_messages();
        assert_eq!(sent[0].role, Role::System);
        assert!(sent[0].content.to_lowercase().contains("template builder"));
        let rest: Vec<&str> = sent[1..].iter().map(|m| m.content.as_str()).collect();
        assert_eq!(rest, ["hi", "hello", "next"]);
        assert_eq!(history.len(), 4);
    }

    #[tokio::test]
    async fn auto_fix_sends_the_errors() {
        let llm = FakeLlm::text(&fenced(&valid(), "Fixed."));
        let mut history = Vec::new();
        let result = TemplateBuilder::new(&example(), None)
            .auto_fix(&llm, "uses {bogus}", "m", &mut history)
            .await
            .unwrap();
        assert!(result.template.is_some());
        assert!(history[0].content.contains("validation errors"));
    }

    #[tokio::test]
    async fn edit_mode_carries_the_template_and_builds_on_each_accepted_version() {
        let edited = with("system_prompt", json!("You are a terse assistant."));
        let llm = FakeLlm::text(&fenced(&edited, "Done."));
        let mut builder = TemplateBuilder::new(&example(), Some(example()));
        let mut history = Vec::new();
        let first_prompt = builder.prompt();
        assert!(first_prompt.contains("\"system_prompt\": \"You are a meeting assistant.\""));
        assert!(
            first_prompt.contains("Apply this change?")
                && !first_prompt.contains("ALWAYS ask 1-3 focused")
        );
        let current = first_prompt.split("## Current template").nth(1).unwrap();
        assert!(!current.contains("\"key\""));

        builder
            .generate(&llm, "yes", "m", &mut history)
            .await
            .unwrap();
        assert_eq!(history.last().unwrap().content, "Done.");
        llm.set(Reply::Text("Apply this change?".into(), 0));
        builder
            .generate(&llm, "also shorter", "m", &mut history)
            .await
            .unwrap();
        let prompt = &llm.last_messages()[0].content;
        assert!(
            prompt.contains("You are a terse assistant.")
                && !prompt.contains("You are a meeting assistant.")
        );
    }

    #[tokio::test]
    async fn an_invalid_edit_stays_in_the_history_for_the_fix() {
        let broken = with("digest_user_template", json!("Notes {no_such_field}"));
        let llm = FakeLlm::text(&fenced(&broken, ""));
        let mut builder = TemplateBuilder::new(&example(), Some(example()));
        let mut history = Vec::new();
        let result = builder
            .generate(&llm, "yes", "m", &mut history)
            .await
            .unwrap();
        assert!(!result.validation_errors.is_empty());
        assert!(history.last().unwrap().content.contains("```json"));
        assert_eq!(builder.current_template(), Some(&example()));
    }

    #[tokio::test]
    async fn a_json_only_reply_shows_no_chat_and_records_a_short_note() {
        let llm = FakeLlm::text(&format!(
            "```json\n{}\n```",
            serde_json::to_string_pretty(&valid()).unwrap()
        ));
        let result = TemplateBuilder::new(&example(), None)
            .generate(&llm, "go", "m", &mut Vec::new())
            .await
            .unwrap();
        assert!(result.template.is_some() && result.assistant_message.is_empty());
        let mut history = Vec::new();
        TemplateBuilder::new(&example(), Some(example()))
            .generate(&llm, "yes", "m", &mut history)
            .await
            .unwrap();
        assert_eq!(history.last().unwrap().content, "Template updated.");
    }

    #[test]
    fn the_create_prompt_has_no_current_template() {
        assert!(
            !TemplateBuilder::new(&example(), None)
                .prompt()
                .contains("## Current template")
        );
    }
}

mod quick_actions {
    use super::*;

    fn input<'a>(
        segments: &'a [lazy_take_notes::domain::transcript::TranscriptSegment],
        context: &'a str,
    ) -> QuickActionInput<'a> {
        QuickActionInput {
            latest_digest: Some("digest"),
            segments,
            user_context: context,
        }
    }

    #[tokio::test]
    async fn key_one_runs_the_first_action() {
        let template = built_in_template("default_zh_tw");
        let llm = FakeLlm::text("Action result");
        let (text, label) = run_quick_action(&llm, "1", &template, "m", input(&[], ""))
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            (text.as_str(), label.as_str()),
            ("Action result", template.quick_actions[0].label.as_str())
        );
    }

    #[tokio::test]
    async fn a_template_error_reaches_the_screen_as_python_prints_it() {
        // Python's KeyError reads `'nope'`; the modal shows `Error: 'nope'`.
        let mut template = built_in_template("default_zh_tw");
        template.quick_actions[0].prompt_template = "{nope} x".into();
        let llm = FakeLlm::text("unused");
        let result = run_quick_action(&llm, "1", &template, "m", input(&[], "")).await;
        assert_eq!(result, Some(Err("'nope'".to_string())));
        assert!(llm.single_calls.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn an_unknown_key_runs_nothing() {
        let template = built_in_template("default_zh_tw");
        let llm = FakeLlm::text("x");
        for key in ["nonexistent", "99", "0"] {
            assert!(
                run_quick_action(&llm, key, &template, "m", input(&[], ""))
                    .await
                    .is_none(),
                "{key}"
            );
        }
        assert!(llm.single_calls.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn the_prompt_holds_the_last_fifty_lines_and_the_context() {
        let template = built_in_template("default_zh_tw");
        let llm = FakeLlm::text("OK");
        let segments: Vec<_> = (0..60)
            .map(|i| segment(&format!("Seg {i}"), i as f64, i as f64 + 1.0))
            .collect();
        run_quick_action(
            &llm,
            "1",
            &template,
            "m",
            input(&segments, "Speaker A = Alice"),
        )
        .await
        .unwrap()
        .unwrap();
        let prompt = llm.single_calls.lock().unwrap()[0].1.clone();
        assert!(
            prompt.contains("Seg 59") && prompt.contains("Seg 10\n") && !prompt.contains("Seg 9\n")
        );
        assert!(prompt.contains("Speaker A = Alice"));
    }
}

#[tokio::test]
async fn a_label_is_sanitized() {
    let llm = FakeLlm::text("\"Sprint Planning Review\"");
    assert_eq!(
        generate_label(&llm, "Default", "", "digest", "m")
            .await
            .unwrap(),
        "sprint_planning_review"
    );
}

mod speakers {
    use super::*;

    fn turn(start: f64, end: f64, speaker: u32) -> SpeakerTurn {
        SpeakerTurn {
            start,
            end,
            speaker,
        }
    }

    #[test]
    fn speaker_at_a_moment() {
        let turns = [turn(0.0, 2.0, 1), turn(2.0, 5.0, 2)];
        assert_eq!(
            (
                speaker_at(&turns, 1.5),
                speaker_at(&turns, 2.0),
                speaker_at(&turns, 9.0)
            ),
            (Some(1), Some(2), None)
        );
        assert_eq!(speaker_at(&[], 1.0), None);
        let overlap = [turn(0.0, 10.0, 1), turn(4.0, 4.5, 2)];
        assert_eq!(
            (speaker_at(&overlap, 4.2), speaker_at(&overlap, 6.0)),
            (Some(2), Some(1))
        );
    }

    #[test]
    fn speaker_for_a_span() {
        assert_eq!(
            speaker_for_span(
                &[turn(0.0, 2.0, 1), turn(2.0, 6.0, 2)],
                1.95,
                4.0,
                SPAN_TOLERANCE
            ),
            Some(2)
        );
        assert_eq!(
            speaker_for_span(&[turn(0.0, 2.0, 1)], 2.4, 3.0, SPAN_TOLERANCE),
            Some(1)
        );
        assert_eq!(
            speaker_for_span(&[turn(0.0, 2.0, 1)], 30.0, 31.0, SPAN_TOLERANCE),
            None
        );
        assert_eq!(
            speaker_for_span(&[turn(1.0, 2.0, 3)], 1.5, 1.5, SPAN_TOLERANCE),
            Some(3)
        );
    }
}
