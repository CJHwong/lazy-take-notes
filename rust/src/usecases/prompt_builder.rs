//! Pure functions that build LLM prompts from templates.

use super::pyformat::{self, FormatError};
use crate::domain::template::SessionTemplate;

/// The user prompt of one digest cycle.
pub fn build_digest_prompt(
    template: &SessionTemplate,
    buffer: &[String],
    is_final: bool,
    full_transcript: &str,
    user_context: &str,
) -> Result<String, FormatError> {
    let new_lines = buffer.join("\n");
    let line_count = buffer.len().to_string();
    let context = user_context.trim();
    let context_section = if context.is_empty() {
        String::new()
    } else {
        format!("User corrections and additions:\n{context}")
    };
    let mut values = vec![
        ("line_count", line_count.as_str()),
        ("new_lines", new_lines.as_str()),
        ("user_context", context_section.as_str()),
    ];
    if !is_final {
        return pyformat::format(&template.digest_user_template, &values);
    }
    let full = if full_transcript.is_empty() {
        "(no full transcript)"
    } else {
        full_transcript
    };
    values.push(("full_transcript", full));
    pyformat::format(&template.final_user_template, &values)
}

/// The user prompt of one quick action.
pub fn build_quick_action_prompt(
    prompt_template: &str,
    digest_markdown: &str,
    recent_transcript: &str,
    user_context: &str,
) -> Result<String, FormatError> {
    let digest = if digest_markdown.is_empty() {
        "(no digest yet)"
    } else {
        digest_markdown
    };
    let recent = if recent_transcript.is_empty() {
        "(no transcript yet)"
    } else {
        recent_transcript
    };
    let mut result = pyformat::format(
        prompt_template,
        &[("digest_markdown", digest), ("recent_transcript", recent)],
    )?;
    let context = user_context.trim();
    if !context.is_empty() {
        result.push_str(&format!("\n\nUser corrections and context:\n{context}"));
    }
    Ok(result)
}

/// A prompt that asks for a short session label.
pub fn build_label_prompt(
    template_name: &str,
    template_description: &str,
    latest_digest: &str,
) -> String {
    let mut header = format!("Template: {template_name}");
    if !template_description.is_empty() {
        header.push_str(&format!(" — {template_description}"));
    }
    format!(
        "Generate a short label (2-5 words, snake_case, lowercase) that summarises this session.\n\n\
         {header}\n\nLatest digest:\n{latest_digest}\n\nOutput ONLY the label, nothing else."
    )
}

/// The synthetic user message that replaces a compacted history.
pub fn build_compact_user_message(latest_markdown: &str) -> String {
    format!(
        "(Prior conversation compacted) Current session state:\n\n{latest_markdown}\n\n\
         Continue analyzing subsequent transcript segments based on this state."
    )
}
