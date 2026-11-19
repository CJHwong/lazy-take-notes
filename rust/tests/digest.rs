mod common;

use std::time::{Duration, Instant};

use common::{FakeLlm, Reply, built_in_template};
use lazy_take_notes::domain::chat::ChatMessage;
use lazy_take_notes::domain::digest_state::DigestState;
use lazy_take_notes::domain::template::SessionTemplate;
use lazy_take_notes::ports::llm::{ChatResponse, LlmError};
use lazy_take_notes::usecases::compact::compact_messages;
use lazy_take_notes::usecases::digest::{self, DigestInput, should_trigger_digest};

const REPLY: &str =
    "## 目前議題\n討論 REST vs GraphQL API 設計方案。\n\n## 待釐清問題\n- 要選哪個框架？\n";

fn setup() -> (DigestState, SessionTemplate) {
    let template = built_in_template("default_zh_tw");
    let mut state = DigestState::new(&template.system_prompt);
    state.buffer = (0..20).map(|i| format!("Line {i}")).collect();
    state.all_lines = state.buffer.clone();
    (state, template)
}

#[tokio::test]
async fn success_updates_the_state() {
    let (mut state, template) = setup();
    let llm = FakeLlm::new(Reply::Text(REPLY.into(), 500));
    let result = digest::run(&llm, &mut state, "m", &template, DigestInput::default()).await;
    assert!(result.unwrap().contains("目前議題"));
    assert_eq!(
        (
            state.digest_count,
            state.consecutive_failures,
            state.prompt_tokens
        ),
        (1, 0, 500)
    );
    assert!(state.buffer.is_empty());
    assert_eq!(state.messages.len(), 3);
}

#[tokio::test]
async fn an_empty_reply_fails_and_keeps_the_buffer() {
    let (mut state, template) = setup();
    let before = state.messages.clone();
    let llm = FakeLlm::text("   ");
    let error = digest::run(&llm, &mut state, "m", &template, DigestInput::default())
        .await
        .unwrap_err();
    assert!(error.contains("Empty response"));
    assert_eq!((state.consecutive_failures, state.buffer.len()), (1, 20));
    assert_eq!(state.messages, before);
}

#[tokio::test]
async fn an_llm_error_fails_the_digest() {
    let (mut state, template) = setup();
    let llm = FakeLlm::new(Reply::Fail(LlmError::new("ConnectionError", "LLM down")));
    let error = digest::run(&llm, &mut state, "m", &template, DigestInput::default())
        .await
        .unwrap_err();
    assert_eq!(error, "LLM error: ConnectionError: LLM down");
    assert_eq!(state.consecutive_failures, 1);
}

#[tokio::test]
async fn a_final_digest_sends_the_full_transcript() {
    let (mut state, template) = setup();
    let llm = FakeLlm::text(REPLY);
    let input = DigestInput {
        is_final: true,
        full_transcript: "Full transcript text",
        ..Default::default()
    };
    digest::run(&llm, &mut state, "m", &template, input)
        .await
        .unwrap();
    assert!(
        llm.last_messages()
            .last()
            .unwrap()
            .content
            .contains("Full transcript text")
    );
}

#[tokio::test]
async fn a_malformed_template_fails_without_calling_the_llm() {
    for bad in ["{unknown_var}", "stray { brace", "{line_count:%%}"] {
        let (mut state, template) = setup();
        let broken = SessionTemplate {
            digest_user_template: bad.into(),
            ..template
        };
        let before = state.messages.clone();
        let llm = FakeLlm::text(REPLY);
        let error = digest::run(&llm, &mut state, "m", &broken, DigestInput::default())
            .await
            .unwrap_err();
        assert!(error.starts_with("Template error"), "{bad}: {error}");
        assert_eq!(state.messages, before);
        assert!(llm.chat_calls.lock().unwrap().is_empty());
    }
}

#[tokio::test]
async fn a_cancelled_digest_removes_its_prompt() {
    let (mut state, template) = setup();
    let before = state.messages.clone();
    let llm = FakeLlm::new(Reply::Hang);
    let run = digest::run(&llm, &mut state, "m", &template, DigestInput::default());
    assert!(
        tokio::time::timeout(Duration::from_millis(10), run)
            .await
            .is_err()
    );
    assert_eq!(state.messages, before);
}

#[test]
fn an_abandoned_digest_keeps_a_newer_prompt() {
    let (mut state, template) = setup();
    let before = state.messages.clone();
    let request = digest::begin(&mut state, &template, DigestInput::default()).unwrap();
    let newer = ChatMessage::user("final prompt");
    state.messages.push(newer.clone());
    digest::abandon(&mut state, &request);
    assert_eq!(state.messages, [before, vec![newer]].concat());
}

#[test]
fn a_failed_digest_keeps_a_newer_prompt() {
    let failures = [
        Err(LlmError::new("ConnectionError", "dropped")),
        Ok(ChatResponse {
            content: "   ".into(),
            prompt_tokens: 0,
        }),
    ];
    for reply in failures {
        let (mut state, template) = setup();
        let before = state.messages.clone();
        let request = digest::begin(&mut state, &template, DigestInput::default()).unwrap();
        let newer = ChatMessage::user("final prompt");
        state.messages.push(newer.clone());
        assert!(digest::finish(&mut state, &request, reply).is_err());
        assert_eq!(state.messages, [before, vec![newer]].concat());
    }
}

#[test]
fn speaker_labeled_lines_replace_the_buffer_in_the_prompt() {
    let (mut state, template) = setup();
    let labeled = vec!["S1: hello".to_string()];
    let input = DigestInput {
        buffer_lines: Some(&labeled),
        ..Default::default()
    };
    let request = digest::begin(&mut state, &template, input).unwrap();
    let prompt = &request.messages.last().unwrap().content;
    assert!(prompt.contains("S1: hello"));
    assert!(!prompt.contains("Line 0"));
}

#[test]
fn compaction_leaves_three_messages() {
    let (mut state, _) = setup();
    state.prompt_tokens = 9000;
    compact_messages(&mut state, "## Topic", "SYS");
    let contents: Vec<&str> = state.messages.iter().map(|m| m.content.as_str()).collect();
    assert_eq!(contents[0], "SYS");
    assert!(contents[1].contains("## Topic") && contents[1].contains("compacted"));
    assert_eq!(contents[2], "## Topic");
    assert_eq!(state.prompt_tokens, 0);
}

fn state_with(lines: usize, last_digest: Instant) -> DigestState {
    DigestState {
        buffer: vec!["line".into(); lines],
        last_digest_time: last_digest,
        ..Default::default()
    }
}

#[test]
fn trigger_rules() {
    let now = Instant::now();
    let long_ago = now - Duration::from_secs(10_000);
    assert!(!should_trigger_digest(
        &state_with(2, long_ago),
        5,
        0.0,
        None,
        now
    ));
    assert!(!should_trigger_digest(
        &state_with(6, now),
        5,
        9999.0,
        Some(100),
        now
    ));
    assert!(should_trigger_digest(
        &state_with(20, long_ago),
        5,
        0.0,
        None,
        now
    ));
    assert!(should_trigger_digest(
        &state_with(10, now),
        5,
        9999.0,
        Some(10),
        now
    ));
    assert!(should_trigger_digest(
        &state_with(10, now),
        5,
        9999.0,
        None,
        now
    ));
    assert!(!should_trigger_digest(
        &state_with(6, now),
        5,
        9999.0,
        Some(10),
        now
    ));
}
