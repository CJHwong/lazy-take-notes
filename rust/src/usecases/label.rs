use std::sync::LazyLock;

use regex::Regex;

use crate::ports::llm::{LlmClient, LlmError};
use crate::usecases::prompt_builder::build_label_prompt;

const MAX_WORDS: usize = 5;
static NOT_SNAKE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new("[^a-z0-9]+").expect("valid regex"));

/// Turn an LLM reply into a snake_case label of at most five words. May be empty.
pub fn sanitize_label(raw: &str) -> String {
    let lowered = raw.trim().trim_matches(['"', '\'']).to_lowercase();
    let snake = NOT_SNAKE.replace_all(&lowered, "_");
    let words: Vec<&str> = snake
        .trim_matches('_')
        .split('_')
        .take(MAX_WORDS)
        .filter(|w| !w.is_empty())
        .collect();
    words.join("_")
}

/// Ask the fast model for a short session label.
pub async fn generate_label(
    llm: &dyn LlmClient,
    template_name: &str,
    template_description: &str,
    latest_digest: &str,
    model: &str,
) -> Result<String, LlmError> {
    let prompt = build_label_prompt(template_name, template_description, latest_digest);
    let raw = llm.chat_single(model, &prompt).await?;
    Ok(sanitize_label(&raw))
}
