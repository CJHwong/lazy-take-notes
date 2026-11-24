//! Checks the format variables of a template before a live session formats it.
//!
//! A template can pass the schema and still fail at `.format()` time in the middle
//! of a recording. This finds those errors up front.

use std::fmt;

use super::pyformat;
use crate::domain::template::SessionTemplate;

pub const DIGEST_VARIABLES: &[&str] = &["line_count", "new_lines", "user_context"];
pub const FINAL_VARIABLES: &[&str] =
    &["full_transcript", "line_count", "new_lines", "user_context"];
pub const QUICK_ACTION_VARIABLES: &[&str] = &["digest_markdown", "recent_transcript"];

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TemplateValidationResult {
    pub errors: Vec<String>,
}

impl TemplateValidationResult {
    pub fn valid(&self) -> bool {
        self.errors.is_empty()
    }
}

impl fmt::Display for TemplateValidationResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.valid() {
            return f.write_str("Template is valid.");
        }
        let lines: Vec<String> = self
            .errors
            .iter()
            .map(|error| format!("- {error}"))
            .collect();
        f.write_str(&lines.join("\n"))
    }
}

/// `{a}, {b}` for a sorted variable list.
pub fn format_variables(variables: &[&str]) -> String {
    let mut sorted = variables.to_vec();
    sorted.sort_unstable();
    sorted
        .iter()
        .map(|name| format!("{{{name}}}"))
        .collect::<Vec<_>>()
        .join(", ")
}

pub fn validate_template(template: &SessionTemplate) -> TemplateValidationResult {
    let mut result = TemplateValidationResult::default();
    check_field(
        &mut result,
        "digest_user_template",
        &template.digest_user_template,
        DIGEST_VARIABLES,
    );
    check_field(
        &mut result,
        "final_user_template",
        &template.final_user_template,
        FINAL_VARIABLES,
    );
    for (index, action) in template.quick_actions.iter().enumerate() {
        let name = format!("quick_actions[{index}].prompt_template");
        check_field(
            &mut result,
            &name,
            &action.prompt_template,
            QUICK_ACTION_VARIABLES,
        );
    }
    if template.system_prompt.trim().is_empty() {
        result.errors.push("system_prompt is empty.".into());
    }
    result
}

fn check_field(result: &mut TemplateValidationResult, field: &str, text: &str, allowed: &[&str]) {
    if text.trim().is_empty() {
        return;
    }
    let mut unknown: Vec<String> = pyformat::field_names(text)
        .into_iter()
        .filter(|name| !allowed.contains(&name.as_str()))
        .collect();
    unknown.sort();
    let allowed_list = format_variables(allowed);
    for name in unknown {
        result.errors.push(format!(
            "{field} uses unknown variable {{{name}}}. Allowed: {allowed_list}"
        ));
    }
    let dummies: Vec<(&str, &str)> = allowed.iter().map(|name| (*name, "")).collect();
    if let Err(error) = pyformat::format(text, &dummies) {
        result.errors.push(format!("{field} format error: {error}"));
    }
}
