//! One digest cycle: build the prompt, ask the LLM, update the state.
//!
//! The cycle has two halves around the LLM call, so the caller never holds the
//! state across an await: `begin` adds the prompt to the history, `finish` applies
//! the reply. A digest that a newer one replaced calls `abandon` instead, which
//! removes its prompt so the history keeps strict user and assistant pairs.

use std::time::Instant;

use tracing::{debug, error, info, warn};

use crate::domain::chat::ChatMessage;
use crate::domain::digest_state::DigestState;
use crate::domain::template::SessionTemplate;
use crate::ports::llm::{ChatResponse, LlmClient, LlmError};
use crate::usecases::prompt_builder::build_digest_prompt;

/// The markdown of a successful digest, or the reason it failed.
pub type DigestResult = Result<String, String>;

/// A digest prompt on its way to the LLM.
#[derive(Debug, Clone)]
pub struct DigestRequest {
    /// The whole history to send, ending with this digest's prompt.
    pub messages: Vec<ChatMessage>,
    prompt: ChatMessage,
}

/// What the digest prompt is built from.
#[derive(Debug, Clone, Copy, Default)]
pub struct DigestInput<'a> {
    pub is_final: bool,
    pub full_transcript: &'a str,
    pub user_context: &'a str,
    /// Lines to send instead of `state.buffer`, e.g. with speaker labels.
    pub buffer_lines: Option<&'a [String]>,
}

/// Add this digest's prompt to the history. A template that fails to format fails the digest.
pub fn begin(
    state: &mut DigestState,
    template: &SessionTemplate,
    input: DigestInput<'_>,
) -> Result<DigestRequest, String> {
    info!(
        target: "ltn.llm",
        "Digest request: {} buffer lines, final={}, msgs={}, prompt_tokens~{}",
        state.buffer.len(),
        input.is_final,
        state.messages.len(),
        state.prompt_tokens
    );
    let lines = input.buffer_lines.unwrap_or(&state.buffer);
    // Only the template builder validates a template, so a hand-edited one can still
    // hold a bad placeholder. That must fail this digest, not the session.
    let prompt = build_digest_prompt(
        template,
        lines,
        input.is_final,
        input.full_transcript,
        input.user_context,
    )
    .map_err(|e| {
        state.consecutive_failures += 1;
        let message = format!("Template error: {}: {e}", e.kind());
        error!(target: "ltn.llm", "{message}");
        message
    })?;
    let prompt = ChatMessage::user(prompt);
    state.messages.push(prompt.clone());
    Ok(DigestRequest {
        messages: state.messages.clone(),
        prompt,
    })
}

/// Apply the LLM's reply, or its failure, to the state.
pub fn finish(
    state: &mut DigestState,
    request: &DigestRequest,
    reply: Result<ChatResponse, LlmError>,
) -> DigestResult {
    let response = match reply {
        Ok(response) => response,
        Err(e) => return fail(state, request, format!("LLM error: {e}")),
    };
    let raw = response.content;
    debug!(target: "ltn.llm", "LLM raw response ({} chars): {}", raw.chars().count(), preview(&raw));
    if raw.trim().is_empty() {
        return fail(state, request, "Empty response from LLM".into());
    }
    state.messages.push(ChatMessage::assistant(raw.clone()));
    state.consecutive_failures = 0;
    state.digest_count += 1;
    state.prompt_tokens = response.prompt_tokens;
    // Lines that arrived during the call are dropped too. The Python version does the same.
    state.buffer.clear();
    state.last_digest_time = Instant::now();
    info!(target: "ltn.llm", "Digest #{} succeeded (prompt_tokens={})", state.digest_count, state.prompt_tokens);
    Ok(raw.trim().to_string())
}

/// A newer digest replaced this one. Drop its unanswered prompt.
pub fn abandon(state: &mut DigestState, request: &DigestRequest) {
    remove_prompt(state, &request.prompt);
}

/// Run a whole cycle in place: for callers that own the state, such as headless mode.
pub async fn run(
    llm: &dyn LlmClient,
    state: &mut DigestState,
    model: &str,
    template: &SessionTemplate,
    input: DigestInput<'_>,
) -> DigestResult {
    let request = begin(state, template, input)?;
    let mut guard = AbandonOnDrop {
        state,
        request: Some(request),
    };
    let messages = guard
        .request
        .as_ref()
        .map(|r| r.messages.clone())
        .unwrap_or_default();
    let reply = llm.chat(model, &messages).await;
    let request = guard.request.take().expect("request is set until here");
    finish(guard.state, &request, reply)
}

/// Removes the prompt when `run` is cancelled during the LLM call.
struct AbandonOnDrop<'a> {
    state: &'a mut DigestState,
    request: Option<DigestRequest>,
}

impl Drop for AbandonOnDrop<'_> {
    fn drop(&mut self) {
        if let Some(request) = self.request.take() {
            abandon(self.state, &request);
        }
    }
}

fn fail(state: &mut DigestState, request: &DigestRequest, message: String) -> DigestResult {
    state.consecutive_failures += 1;
    remove_prompt(state, &request.prompt);
    warn!(target: "ltn.llm", "{message}");
    Err(message)
}

/// Remove this digest's prompt, searching from the end.
///
/// A replacing digest may have added its prompt after this one, so the last
/// message is not always ours. Two pending prompts with the same text are
/// interchangeable, so matching by value gives the same history as by identity.
fn remove_prompt(state: &mut DigestState, prompt: &ChatMessage) {
    if let Some(index) = state.messages.iter().rposition(|message| message == prompt) {
        state.messages.remove(index);
    }
}

fn preview(raw: &str) -> String {
    raw.chars()
        .take(500)
        .collect::<String>()
        .replace('\n', "\\n")
}

/// Trigger when the buffer holds `min_lines` and `min_interval` seconds have passed,
/// or at once when it holds `max_lines` (twice `min_lines` when unset).
pub fn should_trigger_digest(
    state: &DigestState,
    min_lines: usize,
    min_interval: f64,
    max_lines: Option<usize>,
    now: Instant,
) -> bool {
    let size = state.buffer.len();
    if size < min_lines {
        return false;
    }
    if size >= max_lines.unwrap_or(min_lines * 2) {
        return true;
    }
    now.saturating_duration_since(state.last_digest_time)
        .as_secs_f64()
        >= min_interval
}
