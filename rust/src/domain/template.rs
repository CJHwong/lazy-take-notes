use serde::{Deserialize, Serialize};
use thiserror::Error;

pub const MAX_QUICK_ACTIONS: usize = 5;

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct TemplateMetadata {
    pub name: String,
    pub description: String,
    pub locale: String,
    /// The file key. The loader sets it; the YAML file does not store it.
    pub key: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QuickAction {
    pub label: String,
    #[serde(default)]
    pub description: String,
    pub prompt_template: String,
}

/// A session template. The serde field order is the order of the JSON the LLM sees.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct SessionTemplate {
    pub metadata: TemplateMetadata,
    pub system_prompt: String,
    pub digest_user_template: String,
    pub final_user_template: String,
    pub recognition_hints: Vec<String>,
    pub quick_actions: Vec<QuickAction>,
}

#[derive(Debug, Error, PartialEq, Eq)]
#[error("At most 5 quick_actions allowed, got {0}")]
pub struct TooManyQuickActions(pub usize);

impl SessionTemplate {
    /// The schema rule that serde cannot express: at most five quick actions (keys 1 to 5).
    pub fn check(self) -> Result<Self, TooManyQuickActions> {
        if self.quick_actions.len() > MAX_QUICK_ACTIONS {
            return Err(TooManyQuickActions(self.quick_actions.len()));
        }
        Ok(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_fields_take_their_defaults() {
        let template: SessionTemplate = serde_json::from_str(r#"{"system_prompt": "S"}"#).unwrap();
        assert_eq!(template.system_prompt, "S");
        assert!(template.quick_actions.is_empty());
        assert_eq!(template.metadata, TemplateMetadata::default());
    }

    #[test]
    fn six_quick_actions_are_rejected() {
        let action = QuickAction {
            label: "a".into(),
            description: String::new(),
            prompt_template: "p".into(),
        };
        let template = SessionTemplate {
            quick_actions: vec![action; 6],
            ..Default::default()
        };
        assert_eq!(
            template.check().unwrap_err().to_string(),
            "At most 5 quick_actions allowed, got 6"
        );
    }
}
