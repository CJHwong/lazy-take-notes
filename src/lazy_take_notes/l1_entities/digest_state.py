"""Digest pipeline state entity."""

from __future__ import annotations

import time

from pydantic import BaseModel, Field

from lazy_take_notes.l1_entities.chat_message import ChatMessage


class DigestState(BaseModel):
    """Mutable state for the rolling digest pipeline."""

    messages: list[ChatMessage] = Field(default_factory=list)
    buffer: list[str] = Field(default_factory=list)
    all_lines: list[str] = Field(default_factory=list)
    digest_count: int = 0
    consecutive_failures: int = 0
    last_digest_time: float = Field(default_factory=time.monotonic)
    start_time: float = Field(default_factory=time.monotonic)
    prompt_tokens: int = 0

    model_config = {'arbitrary_types_allowed': True}

    def init_messages(self, system_prompt: str) -> None:
        """Initialize conversation with system prompt."""
        self.messages = [ChatMessage(role='system', content=system_prompt)]
