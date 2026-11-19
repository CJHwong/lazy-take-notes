use std::time::Duration;

use async_trait::async_trait;
use thiserror::Error;

use crate::domain::chat::ChatMessage;

/// Upper bound for one LLM request. Without it a stalled server never answers, the
/// digest never finishes, and the session cannot quit. A final digest of a long
/// session on a local model can take minutes.
pub const LLM_REQUEST_TIMEOUT: Duration = Duration::from_secs(600);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatResponse {
    pub content: String,
    pub prompt_tokens: u64,
}

/// A failed LLM call. `kind` names the failure class for the user, like "ConnectError".
#[derive(Debug, Clone, Error, PartialEq, Eq)]
#[error("{kind}: {message}")]
pub struct LlmError {
    pub kind: String,
    pub message: String,
}

impl LlmError {
    pub fn new(kind: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            kind: kind.into(),
            message: message.into(),
        }
    }
}

#[async_trait]
pub trait LlmClient: Send + Sync {
    /// Multi-turn chat.
    async fn chat(&self, model: &str, messages: &[ChatMessage]) -> Result<ChatResponse, LlmError>;

    /// One user turn, no history. Returns the raw text.
    async fn chat_single(&self, model: &str, prompt: &str) -> Result<String, LlmError> {
        let response = self.chat(model, &[ChatMessage::user(prompt)]).await?;
        Ok(response.content)
    }

    /// Preflight: can the server be reached and does it accept the credentials.
    async fn check_connectivity(&self) -> Result<(), String>;

    /// The names from `models` that the server does not offer.
    async fn check_models(&self, models: &[String]) -> Vec<String>;
}
