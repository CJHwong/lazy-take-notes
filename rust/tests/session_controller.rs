//! Ported from tests/l3_interface_adapters/controllers/test_session_controller.py.

mod common;

use std::sync::{Arc, Mutex};

use common::{FakeLlm, FakePersistence, Reply, Saved, built_in_template};
use lazy_take_notes::adapters::session_controller::SessionController;
use lazy_take_notes::config::build_app_config;
use lazy_take_notes::domain::chat::Role;
use lazy_take_notes::domain::speaker_turn::SpeakerTurn;
use lazy_take_notes::domain::transcript::TranscriptSegment;
use lazy_take_notes::ports::llm::LlmError;

const VALID_DIGEST: &str = "## Topic\nSome content";

struct Setup {
    ctrl: SessionController,
    llm: Arc<FakeLlm>,
    saved: Arc<Mutex<Saved>>,
}

fn setup(config_yaml: &str, template: &str, failing: &[&'static str], tokens: u64) -> Setup {
    let raw = yaml_serde::from_str(config_yaml).unwrap();
    let config = build_app_config(&raw).unwrap();
    let llm = Arc::new(FakeLlm::new(Reply::Text(VALID_DIGEST.into(), tokens)));
    let (persistence, saved) = FakePersistence::failing(failing);
    let ctrl = SessionController::new(
        config,
        built_in_template(template),
        llm.clone(),
        Box::new(persistence),
    );
    Setup { ctrl, llm, saved }
}

fn controller() -> Setup {
    setup("{}", "default_zh_tw", &[], 50)
}

fn line(text: &str, start: f64, end: f64) -> TranscriptSegment {
    TranscriptSegment::new(text, start, end)
}

fn buffer(ctrl: &mut SessionController, buffered: &[&str], all: &[&str]) {
    ctrl.digest_state.buffer = buffered.iter().map(|s| s.to_string()).collect();
    ctrl.digest_state.all_lines = all.iter().map(|s| s.to_string()).collect();
}

#[test]
fn buffers_and_persists_segments() {
    let Setup {
        mut ctrl, saved, ..
    } = controller();
    ctrl.on_transcript_segments(&[line("Hello", 0.0, 1.0), line("World", 1.0, 2.0)]);
    assert_eq!(ctrl.all_segments.len(), 2);
    assert_eq!(ctrl.digest_state.buffer, ["Hello", "World"]);
    assert_eq!(saved.lock().unwrap().transcript.len(), 1);
    assert!(saved.lock().unwrap().transcript[0].1, "appends");
}

#[test]
fn saves_the_template_name_once_with_the_first_lines() {
    let Setup {
        mut ctrl, saved, ..
    } = controller();
    assert!(saved.lock().unwrap().template.is_empty());
    ctrl.on_transcript_segments(&[line("One", 0.0, 1.0)]);
    ctrl.on_transcript_segments(&[line("Two", 1.0, 2.0)]);
    let name = ctrl.template().metadata.name.clone();
    assert_eq!(saved.lock().unwrap().template, [name]);
}

#[test]
fn one_line_does_not_trigger_a_digest() {
    let Setup { mut ctrl, .. } = controller();
    assert!(!ctrl.on_transcript_segments(&[line("Line", 0.0, 1.0)]));
}

#[tokio::test]
async fn a_digest_saves_notes_and_history() {
    let Setup {
        mut ctrl, saved, ..
    } = controller();
    let lines: Vec<String> = (0..20).map(|i| format!("Line {i}")).collect();
    ctrl.digest_state.buffer = lines.clone();
    ctrl.digest_state.all_lines = lines;
    assert_eq!(ctrl.run_digest(false).await.unwrap(), VALID_DIGEST);
    assert_eq!(ctrl.latest_digest.as_deref(), Some(VALID_DIGEST));
    let saved = saved.lock().unwrap();
    assert_eq!(saved.digest, [(VALID_DIGEST.to_string(), 1)]);
    assert_eq!(saved.history, [(VALID_DIGEST.to_string(), 1, false)]);
}

#[tokio::test]
async fn a_final_digest_marks_the_history_final() {
    let Setup {
        mut ctrl, saved, ..
    } = controller();
    buffer(&mut ctrl, &["Line 1"], &["Line 1", "Line 2"]);
    ctrl.run_digest(true).await.unwrap();
    assert!(saved.lock().unwrap().history[0].2);
}

#[tokio::test]
async fn only_a_final_digest_saves_a_non_empty_context() {
    for (is_final, context, expected) in [
        (true, "Speaker A = Alice", vec!["Speaker A = Alice"]),
        (true, "", vec![]),
        (false, "some context", vec![]),
    ] {
        let Setup {
            mut ctrl, saved, ..
        } = controller();
        buffer(&mut ctrl, &["Line 1"], &["Line 1"]);
        ctrl.user_context = context.into();
        ctrl.run_digest(is_final).await.unwrap();
        assert_eq!(saved.lock().unwrap().context, expected, "final={is_final}");
    }
}

#[tokio::test]
async fn a_long_history_is_compacted() {
    let Setup { mut ctrl, .. } = setup(
        "digest: {compact_token_threshold: 50}",
        "default_zh_tw",
        &[],
        200,
    );
    buffer(&mut ctrl, &["Line 1", "Line 2"], &["Line 1", "Line 2"]);
    ctrl.run_digest(false).await.unwrap();
    assert_eq!(ctrl.digest_state.messages.len(), 3);
    assert_eq!(ctrl.digest_state.messages[0].role, Role::System);
}

#[tokio::test]
async fn a_label_comes_from_the_fast_model() {
    let Setup { mut ctrl, llm, .. } = controller();
    ctrl.latest_digest = Some("## Sprint Review\nGood progress.".into());
    llm.set(Reply::Text("sprint_review_notes".into(), 0));
    assert_eq!(
        ctrl.generate_label().await.as_deref(),
        Some("sprint_review_notes")
    );
    assert_eq!(llm.single_calls.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn no_label_without_notes_on_error_or_when_empty() {
    let Setup { ctrl, llm, .. } = controller();
    assert_eq!(ctrl.generate_label().await, None);
    assert!(llm.single_calls.lock().unwrap().is_empty());

    let Setup { mut ctrl, llm, .. } = controller();
    ctrl.latest_digest = Some("some digest".into());
    llm.set(Reply::Fail(LlmError::new("RuntimeError", "LLM is down")));
    assert_eq!(ctrl.generate_label().await, None);
    llm.set(Reply::Text("!!!".into(), 0));
    assert_eq!(ctrl.generate_label().await, None);
}

#[tokio::test]
async fn a_quick_action_runs_only_for_a_known_key() {
    let Setup { ctrl, .. } = controller();
    assert!(ctrl.run_quick_action("1").await.is_some());
    assert!(ctrl.run_quick_action("nonexistent").await.is_none());
}

fn with_turns() -> Setup {
    let mut setup = controller();
    setup
        .ctrl
        .on_transcript_segments(&[line("Hello there", 1.0, 2.0), line("Hi back", 6.0, 7.0)]);
    setup.ctrl.set_speaker_turns(vec![
        SpeakerTurn {
            start: 0.0,
            end: 5.0,
            speaker: 1,
        },
        SpeakerTurn {
            start: 5.5,
            end: 8.0,
            speaker: 2,
        },
    ]);
    setup
}

fn last_prompt(llm: &FakeLlm) -> String {
    llm.last_messages().last().unwrap().content.clone()
}

#[test]
fn a_label_uses_the_mapped_name() {
    let Setup { mut ctrl, .. } = with_turns();
    ctrl.set_speaker_name(1, "Alice");
    assert_eq!(
        ctrl.speaker_label(&ctrl.all_segments[0]).as_deref(),
        Some("Alice")
    );
    assert_eq!(
        ctrl.speaker_label(&ctrl.all_segments[1]).as_deref(),
        Some("S2")
    );
}

#[tokio::test]
async fn the_digest_prompt_carries_speaker_labels() {
    let Setup { mut ctrl, llm, .. } = with_turns();
    ctrl.set_speaker_name(1, "Alice");
    ctrl.run_digest(false).await.unwrap();
    let prompt = last_prompt(&llm);
    assert!(prompt.contains("Alice: Hello there"), "{prompt}");
    assert!(prompt.contains("S2: Hi back"), "{prompt}");
}

#[tokio::test]
async fn the_prompt_is_plain_without_speaker_turns() {
    let Setup { mut ctrl, llm, .. } = controller();
    ctrl.on_transcript_segments(&[line("Hello there", 1.0, 2.0)]);
    ctrl.run_digest(false).await.unwrap();
    let prompt = last_prompt(&llm);
    assert!(prompt.contains("Hello there"));
    assert!(!prompt.contains("S1:"));
}

#[tokio::test]
async fn the_final_digest_labels_the_full_transcript() {
    let Setup { mut ctrl, llm, .. } = with_turns();
    ctrl.set_speaker_name(2, "Bob");
    ctrl.run_digest(true).await.unwrap();
    let prompt = last_prompt(&llm);
    assert!(prompt.contains("S1: Hello there"), "{prompt}");
    assert!(prompt.contains("Bob: Hi back"), "{prompt}");
}

const FAULT_CONFIG: &str = "output: {save_context: true, save_notes_history: true}";

#[test]
fn a_failed_transcript_write_keeps_the_segments_and_reports_once() {
    let Setup { mut ctrl, .. } = setup(FAULT_CONFIG, "default_en", &["save_transcript_lines"], 50);
    ctrl.on_transcript_segments(&[line("Hello", 0.0, 1.0)]);
    assert_eq!(ctrl.all_segments.len(), 1);
    assert_eq!(ctrl.digest_state.buffer, ["Hello"]);
    assert_eq!(
        ctrl.take_persist_error().as_deref(),
        Some("Could not save transcript: [Errno 28] No space left on device")
    );
    assert_eq!(ctrl.take_persist_error(), None);
}

#[tokio::test]
async fn a_failed_digest_write_still_returns_the_digest() {
    for method in ["save_digest_md", "save_history", "save_session_context"] {
        for is_final in [false, true] {
            let Setup { mut ctrl, .. } = setup(FAULT_CONFIG, "default_en", &[method], 50);
            ctrl.user_context = "context".into();
            ctrl.on_transcript_segments(&[line("Hello", 0.0, 1.0)]);
            assert_eq!(ctrl.run_digest(is_final).await.unwrap(), VALID_DIGEST);
            assert_eq!(ctrl.latest_digest.as_deref(), Some(VALID_DIGEST));
            let reaches = method != "save_session_context" || is_final;
            assert_eq!(
                ctrl.take_persist_error().is_some(),
                reaches,
                "{method} final={is_final}"
            );
        }
    }
}

#[test]
fn save_context_now_writes_only_a_non_empty_context() {
    let Setup {
        mut ctrl, saved, ..
    } = setup("output: {save_context: true}", "default_en", &[], 0);
    ctrl.user_context = "   ".into();
    ctrl.save_context_now();
    assert!(saved.lock().unwrap().context.is_empty());
    ctrl.user_context = "agenda".into();
    ctrl.save_context_now();
    assert_eq!(saved.lock().unwrap().context, ["agenda"]);
}

#[test]
fn save_context_now_reports_a_failed_write() {
    let Setup { mut ctrl, .. } = setup(FAULT_CONFIG, "default_en", &["save_session_context"], 0);
    ctrl.user_context = "agenda".into();
    ctrl.save_context_now();
    assert!(
        ctrl.take_persist_error()
            .unwrap()
            .contains("session context")
    );
}

#[tokio::test]
async fn an_abandoned_digest_leaves_no_prompt_behind() {
    let Setup { mut ctrl, .. } = controller();
    buffer(&mut ctrl, &["Line 1"], &["Line 1"]);
    let before = ctrl.digest_state.messages.len();
    let request = ctrl.begin_digest(false).unwrap();
    assert_eq!(ctrl.digest_state.messages.len(), before + 1);
    ctrl.abandon_digest(&request);
    assert_eq!(ctrl.digest_state.messages.len(), before);
}
