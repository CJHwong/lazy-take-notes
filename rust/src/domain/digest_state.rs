use std::time::Instant;

use super::chat::ChatMessage;

/// Mutable state of the rolling digest pipeline.
#[derive(Debug, Clone)]
pub struct DigestState {
    pub messages: Vec<ChatMessage>,
    pub buffer: Vec<String>,
    pub all_lines: Vec<String>,
    pub digest_count: u32,
    pub consecutive_failures: u32,
    pub last_digest_time: Instant,
    pub start_time: Instant,
    pub prompt_tokens: u64,
}

impl Default for DigestState {
    fn default() -> Self {
        let now = Instant::now();
        Self {
            messages: Vec::new(),
            buffer: Vec::new(),
            all_lines: Vec::new(),
            digest_count: 0,
            consecutive_failures: 0,
            last_digest_time: now,
            start_time: now,
            prompt_tokens: 0,
        }
    }
}

impl DigestState {
    pub fn new(system_prompt: &str) -> Self {
        let mut state = Self::default();
        state.init_messages(system_prompt);
        state
    }

    /// Start the conversation with the system prompt.
    pub fn init_messages(&mut self, system_prompt: &str) {
        self.messages = vec![ChatMessage::system(system_prompt)];
    }
}
