"""Use case: generate a short session label from the latest digest."""

from __future__ import annotations

import re

from lazy_take_notes.l2_use_cases.ports.llm_client import LLMClient
from lazy_take_notes.l2_use_cases.query_use_case import RunQueryUseCase
from lazy_take_notes.l2_use_cases.utils.prompt_builder import build_label_prompt

_MAX_WORDS = 5
_SNAKE_RE = re.compile(r'[^a-z0-9]+')


def sanitize_label(raw: str) -> str:
    """Normalise an LLM response into a clean snake_case label."""
    lowered = raw.strip().strip('"\'').lower()
    snake = _SNAKE_RE.sub('_', lowered).strip('_')
    words = snake.split('_')[:_MAX_WORDS]
    return '_'.join(w for w in words if w)


class GenerateLabelUseCase:
    """Asks the fast model for a short descriptive session label."""

    def __init__(self, llm_client: LLMClient) -> None:
        self._query = RunQueryUseCase(llm_client)

    async def execute(self, template_name: str, template_description: str, latest_digest: str, model: str) -> str:
        """Generate a label. Returns sanitised snake_case string (may be empty)."""
        prompt = build_label_prompt(template_name, template_description, latest_digest)
        raw = await self._query.execute(prompt, model)
        return sanitize_label(raw)
