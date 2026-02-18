"""Use case: compact conversation history to stay within token budget."""

from __future__ import annotations

from lazy_take_notes.l1_entities.chat_message import ChatMessage
from lazy_take_notes.l1_entities.digest_state import DigestState
from lazy_take_notes.l2_use_cases.utils.prompt_builder import build_compact_user_message


class CompactMessagesUseCase:
    """Compacts a DigestState's message history to 3 messages."""

    def execute(
        self,
        state: DigestState,
        latest_markdown: str,
        system_prompt: str,
    ) -> None:
        """Replace message history with system + compacted user + assistant."""
        state.messages = [
            ChatMessage(role='system', content=system_prompt),
            ChatMessage(role='user', content=build_compact_user_message(latest_markdown)),
            ChatMessage(role='assistant', content=latest_markdown),
        ]
        state.prompt_tokens = 0
