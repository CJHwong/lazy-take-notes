"""Chat message entity — typed replacement for dict[str, Any]."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class ChatMessage(BaseModel):
    """A single message in an LLM conversation."""

    role: Literal['system', 'user', 'assistant']
    content: str
