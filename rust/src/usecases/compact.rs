use crate::domain::chat::ChatMessage;
use crate::domain::digest_state::DigestState;
use crate::usecases::prompt_builder::build_compact_user_message;

/// Replace the history with three messages: system, compacted state, last reply.
pub fn compact_messages(state: &mut DigestState, latest_markdown: &str, system_prompt: &str) {
    state.messages = vec![
        ChatMessage::system(system_prompt),
        ChatMessage::user(build_compact_user_message(latest_markdown)),
        ChatMessage::assistant(latest_markdown),
    ];
    state.prompt_tokens = 0;
}
