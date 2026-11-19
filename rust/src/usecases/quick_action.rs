use crate::domain::template::{QuickAction, SessionTemplate};
use crate::domain::transcript::TranscriptSegment;
use crate::ports::llm::LlmClient;
use crate::usecases::prompt_builder::build_quick_action_prompt;

/// How many of the newest transcript lines a quick action sees.
pub const RECENT_LINES: usize = 50;

/// The quick action on key `"1"` to `"5"`, by position.
pub fn find_action<'t>(key: &str, template: &'t SessionTemplate) -> Option<&'t QuickAction> {
    let position: usize = key.trim().parse().ok()?;
    template.quick_actions.get(position.checked_sub(1)?)
}

/// What a quick action needs from the session.
#[derive(Debug, Clone, Copy)]
pub struct QuickActionInput<'a> {
    pub latest_digest: Option<&'a str>,
    pub segments: &'a [TranscriptSegment],
    pub user_context: &'a str,
}

/// Run a quick action. Returns (reply, label), or None for a key with no action.
/// Fails with a message on a template error or an LLM error.
pub async fn run_quick_action(
    llm: &dyn LlmClient,
    key: &str,
    template: &SessionTemplate,
    model: &str,
    input: QuickActionInput<'_>,
) -> Option<Result<(String, String), String>> {
    let action = find_action(key, template)?;
    let start = input.segments.len().saturating_sub(RECENT_LINES);
    let recent: Vec<&str> = input.segments[start..]
        .iter()
        .map(|segment| segment.text.as_str())
        .collect();
    let digest = input.latest_digest.unwrap_or("(no digest yet)");
    let prompt = match build_quick_action_prompt(
        &action.prompt_template,
        digest,
        &recent.join("\n"),
        input.user_context,
    ) {
        Ok(prompt) => prompt,
        // Python lets the formatting error reach the screen, which shows `Error: {e}`.
        Err(e) => return Some(Err(e.to_string())),
    };
    let reply = llm
        .chat_single(model, &prompt)
        .await
        .map_err(|e| e.to_string());
    Some(reply.map(|text| (text, action.label.clone())))
}
