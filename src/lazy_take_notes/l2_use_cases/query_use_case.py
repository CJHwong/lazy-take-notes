"""Use case: run a quick-action query via the LLM client."""

from __future__ import annotations

from lazy_take_notes.l2_use_cases.ports.llm_client import LLMClient


class RunQueryUseCase:
    """Runs a stateless single-turn query against the LLM."""

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm = llm_client

    async def execute(self, prompt: str, model: str) -> str:
        """Execute a single-turn query. Returns the response text. Raises on LLM failure."""
        return await self._llm.chat_single(model=model, prompt=prompt)
