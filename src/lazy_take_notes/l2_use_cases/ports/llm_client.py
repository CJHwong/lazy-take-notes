"""Port: LLM chat client."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from lazy_take_notes.l1_entities.chat_message import ChatMessage

# Upper bound for one LLM request. Without it a stalled server never answers,
# the digest flag never clears, and the session cannot quit. A final digest of a
# long session on a local model can legitimately take minutes.
LLM_REQUEST_TIMEOUT_S = 600.0


@dataclass(frozen=True)
class ChatResponse:
    """Response from an LLM chat call."""

    content: str
    prompt_tokens: int = 0


class LLMClient(Protocol):
    """Abstract LLM client. Zero framework types leak through."""

    async def chat(self, model: str, messages: list[ChatMessage]) -> ChatResponse:
        """Multi-turn chat. Returns structured response."""
        ...

    async def chat_single(self, model: str, prompt: str) -> str:
        """Single-turn chat convenience. Returns raw text."""
        ...

    def check_connectivity(self) -> tuple[bool, str]:
        """Pre-flight connectivity check. Returns (ok, error_message)."""
        ...

    def check_models(self, models: list[str]) -> list[str]:
        """Return model names from the list that are not available locally."""
        ...
