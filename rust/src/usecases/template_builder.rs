//! LLM-powered template builder: a chat that ends in a validated template.

use std::sync::LazyLock;

use regex::Regex;
use serde_json::Value;

use super::template_validator::{
    DIGEST_VARIABLES, FINAL_VARIABLES, QUICK_ACTION_VARIABLES, format_variables, validate_template,
};
use crate::domain::chat::ChatMessage;
use crate::domain::template::SessionTemplate;
use crate::ports::llm::{LlmClient, LlmError};

/// The JSON schema the Python version gave the LLM, kept byte for byte. The file
/// ends with a newline for the repo's end-of-file hook; the prompt has none.
const SCHEMA_FILE: &str = include_str!("session_template_schema.json");

static JSON_FENCE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?s)```json\s*\n(.*?)\n\s*```").expect("valid regex"));

fn schema_doc() -> String {
    format!(
        "## Template Schema (JSON)\n\n{schema}\n\n## Field descriptions\n\n\
- **metadata**: name, description, locale (e.g. \"en\", \"zh-TW\"), key (leave empty).\n\
- **system_prompt**: Instructions for the AI summarizer. Defines tone, structure, and rules.\n\
- **digest_user_template**: The prompt sent each time new transcript lines arrive. \
Available variables: {digest}. The AI uses this to produce an updated summary.\n\
- **final_user_template**: The prompt sent when the session ends. \
Available variables: {final_vars}. Should instruct the AI to produce a final, \
comprehensive summary.\n\
- **recognition_hints**: List of domain-specific words/names to help speech recognition.\n\
- **quick_actions**: Up to 5 shortcut prompts (label, description, prompt_template). \
Available variables: {quick}.\n\n\
## User-facing abstraction\n\n\
The user describes \"what the summary should look like\" — you decide how to fill \
digest_user_template and final_user_template appropriately. Don't ask the user \
about these fields directly.\n\n",
        digest = format_variables(DIGEST_VARIABLES),
        final_vars = format_variables(FINAL_VARIABLES),
        quick = format_variables(QUICK_ACTION_VARIABLES),
        schema = SCHEMA_FILE.trim_end(),
    )
}

fn create_prompt(example: &SessionTemplate) -> String {
    let example = pretty_json(&serde_json::to_value(example).expect("a template serializes"));
    format!(
        "You are a template builder for a live transcription and AI digest app.\n\n\
Your job: help the user create a session template. The template controls how the \
AI summarizes a live transcript during a meeting, lecture, or other audio session.\n\n\
{schema}\
## Behavior\n\n\
- Be thorough before generating. On the first message, ALWAYS ask 1-3 focused \
follow-up questions to clarify what the user wants. Key things to probe for:\n\
\x20 - Language/locale (e.g. English, Chinese, etc.)\n\
\x20 - What sections the summary should have (action items? decisions? open questions?)\n\
\x20 - Tone and audience (formal minutes vs casual notes)\n\
\x20 - Any quick actions they'd find useful\n\
\x20 - Domain-specific vocabulary (recognition hints)\n\
- Once you have enough context, ALWAYS confirm before generating. Summarize your \
understanding and ask \"Should I generate the template now?\" or similar. Only \
generate after the user confirms (e.g. \"yes\", \"go ahead\", \"looks good\").\n\
- When generating, output the complete JSON wrapped in ```json fences.\n\
- Always produce valid JSON that conforms to the schema above.\n\n\
## Example template (for reference)\n\n\
```json\n{example}\n```\n",
        schema = schema_doc(),
    )
}

fn edit_prompt(current: &SessionTemplate) -> String {
    format!(
        "You are a template editor for a live transcription and AI digest app.\n\n\
The user wants to change an existing session template. The template controls how the \
AI summarizes a live transcript during a meeting, lecture, or other audio session.\n\n\
{schema}\
## Behavior\n\n\
- Before you change anything, restate the requested change in plain words, name the \
fields it touches, and ask \"Apply this change?\". If the request is unclear, ask one \
focused question instead.\n\
- Generate only after the user confirms (e.g. \"yes\", \"go ahead\", \"looks good\").\n\
- Change only what the user agreed to. Copy every other field exactly as it is in the \
current template, including wording, order, and language.\n\
- When generating, output the complete template as JSON wrapped in ```json fences, \
not only the changed fields.\n\
- Always produce valid JSON that conforms to the schema above.\n\n\
## Current template\n\n\
This is the latest version. It already includes every change applied so far.\n\n\
```json\n{current}\n```\n",
        schema = schema_doc(),
        current = template_json(current),
    )
}

/// The template as the LLM sees it. The loader sets the key, so the LLM must not invent one.
fn template_json(template: &SessionTemplate) -> String {
    let mut value = serde_json::to_value(template).expect("a template serializes");
    if let Some(metadata) = value.get_mut("metadata").and_then(Value::as_object_mut) {
        metadata.shift_remove("key");
    }
    pretty_json(&value)
}

/// `json.dumps(value, indent=2, ensure_ascii=False)`.
fn pretty_json(value: &Value) -> String {
    serde_json::to_string_pretty(value).expect("a JSON value serializes")
}

/// The outcome of one builder turn.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct BuilderResult {
    pub template: Option<SessionTemplate>,
    pub validation_errors: String,
    pub assistant_message: String,
    pub error: String,
}

/// Creates a template, or edits `base_template` when it is set. Each valid
/// version the LLM returns becomes the template the next edit turn starts from.
pub struct TemplateBuilder {
    system_prompt: String,
    base_template: Option<SessionTemplate>,
}

impl TemplateBuilder {
    pub fn new(example: &SessionTemplate, base_template: Option<SessionTemplate>) -> Self {
        Self {
            system_prompt: create_prompt(example),
            base_template,
        }
    }

    /// The template the next edit turn starts from. None while creating.
    pub fn current_template(&self) -> Option<&SessionTemplate> {
        self.base_template.as_ref()
    }

    pub fn prompt(&self) -> String {
        match &self.base_template {
            None => self.system_prompt.clone(),
            Some(current) => edit_prompt(current),
        }
    }

    /// Send one user turn and parse the reply for a template.
    pub async fn generate(
        &mut self,
        llm: &dyn LlmClient,
        user_message: &str,
        model: &str,
        history: &mut Vec<ChatMessage>,
    ) -> Result<BuilderResult, LlmError> {
        let user_turn = ChatMessage::user(user_message);
        let mut messages = Vec::with_capacity(history.len() + 2);
        messages.push(ChatMessage::system(self.prompt()));
        messages.extend(history.iter().cloned());
        messages.push(user_turn.clone());
        let reply = llm.chat(model, &messages).await?.content;
        let result = parse_response(&reply);

        let mut recorded = reply;
        if self.base_template.is_some()
            && result.validation_errors.is_empty()
            && let Some(template) = &result.template
        {
            // The edit prompt now carries this version. An older copy in the history would contradict it.
            self.base_template = Some(template.clone());
            recorded = if result.assistant_message.is_empty() {
                "Template updated.".into()
            } else {
                result.assistant_message.clone()
            };
        }
        // Record the turn only once it has a reply: a failed call must not leave a
        // lone user turn, or the retry sends two user turns in a row.
        history.push(user_turn);
        history.push(ChatMessage::assistant(recorded));
        Ok(result)
    }

    /// Ask the LLM to fix the validation errors of its last template.
    pub async fn auto_fix(
        &mut self,
        llm: &dyn LlmClient,
        validation_errors: &str,
        model: &str,
        history: &mut Vec<ChatMessage>,
    ) -> Result<BuilderResult, LlmError> {
        let fix = format!(
            "The generated template has validation errors:\n{validation_errors}\n\n\
             Please fix these errors and output the corrected JSON in ```json fences."
        );
        self.generate(llm, &fix, model, history).await
    }
}

/// Find the fenced JSON in a reply and validate it as a template.
pub fn parse_response(reply: &str) -> BuilderResult {
    let Some(captures) = JSON_FENCE.captures(reply) else {
        // No JSON: the LLM is asking questions.
        return BuilderResult {
            assistant_message: reply.to_string(),
            ..Default::default()
        };
    };
    let json = captures.get(1).map_or("", |m| m.as_str());
    let conversational = JSON_FENCE.replace_all(reply, "").trim().to_string();
    let shown = if conversational.is_empty() {
        reply.to_string()
    } else {
        conversational.clone()
    };

    let raw: Value = match serde_json::from_str(json) {
        Ok(raw) => raw,
        Err(e) => {
            return BuilderResult {
                assistant_message: shown,
                error: format!("Invalid JSON: {e}"),
                ..Default::default()
            };
        }
    };
    let template = match serde_json::from_value::<SessionTemplate>(raw)
        .map_err(|e| e.to_string())
        .and_then(|t| t.check().map_err(|e| e.to_string()))
    {
        Ok(template) => template,
        Err(e) => {
            return BuilderResult {
                assistant_message: shown,
                error: format!("Schema validation failed: {e}"),
                ..Default::default()
            };
        }
    };
    // The template parsed, so its JSON goes to the preview, not the chat.
    let validation = validate_template(&template);
    let validation_errors = if validation.valid() {
        String::new()
    } else {
        validation.to_string()
    };
    BuilderResult {
        template: Some(template),
        validation_errors,
        assistant_message: conversational,
        error: String::new(),
    }
}
