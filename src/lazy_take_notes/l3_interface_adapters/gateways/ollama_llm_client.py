"""Gateway: Ollama LLM client — implements LLMClient port."""

from __future__ import annotations

import ollama as ollama_sync

from lazy_take_notes.l1_entities.chat_message import ChatMessage
from lazy_take_notes.l2_use_cases.ports.llm_client import ChatResponse


class OllamaLLMClient:
    """Wraps ollama.AsyncClient to implement the LLMClient protocol."""

    def __init__(self, host: str = 'http://localhost:11434') -> None:
        self._host = host

    async def chat(self, model: str, messages: list[ChatMessage]) -> ChatResponse:
        client = ollama_sync.AsyncClient(host=self._host)
        resp = await client.chat(model=model, messages=[m.model_dump() for m in messages])
        return ChatResponse(
            content=resp.message.content or '',
            prompt_tokens=getattr(resp, 'prompt_eval_count', 0) or 0,
        )

    async def chat_single(self, model: str, prompt: str) -> str:
        client = ollama_sync.AsyncClient(host=self._host)
        resp = await client.chat(
            model=model,
            messages=[{'role': 'user', 'content': prompt}],
        )
        return resp.message.content or ''

    def check_connectivity(self) -> tuple[bool, str]:
        try:
            client = ollama_sync.Client(host=self._host)
            client.list()
            return True, ''
        except Exception as e:
            return False, f'Cannot connect to Ollama: {e}'
