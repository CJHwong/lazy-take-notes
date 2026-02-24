"""Gateway: OpenAI-compatible LLM client — implements LLMClient port.

Works with any OpenAI-compatible API: OpenAI, Gemini, Groq, Together, vLLM, etc.
"""

from __future__ import annotations

import openai

from lazy_take_notes.l1_entities.chat_message import ChatMessage
from lazy_take_notes.l2_use_cases.ports.llm_client import ChatResponse


class OpenAICompatLLMClient:
    """Wraps openai.AsyncOpenAI to implement the LLMClient protocol."""

    def __init__(self, api_key: str | None = None, base_url: str = 'https://api.openai.com/v1') -> None:
        self._api_key = api_key
        self._base_url = base_url

    async def chat(self, model: str, messages: list[ChatMessage]) -> ChatResponse:
        client = openai.AsyncOpenAI(api_key=self._api_key, base_url=self._base_url)
        resp = await client.chat.completions.create(
            model=model,
            messages=[{'role': m.role, 'content': m.content} for m in messages],  # ty: ignore[invalid-argument-type] -- dict satisfies ChatCompletionMessageParam at runtime
        )
        content = resp.choices[0].message.content or ''
        prompt_tokens = resp.usage.prompt_tokens if resp.usage else 0
        return ChatResponse(content=content, prompt_tokens=prompt_tokens)

    async def chat_single(self, model: str, prompt: str) -> str:
        client = openai.AsyncOpenAI(api_key=self._api_key, base_url=self._base_url)
        resp = await client.chat.completions.create(
            model=model,
            messages=[{'role': 'user', 'content': prompt}],
        )
        return resp.choices[0].message.content or ''

    def check_connectivity(self) -> tuple[bool, str]:
        try:
            client = openai.OpenAI(api_key=self._api_key, base_url=self._base_url)
            client.models.list()
            return True, ''
        except openai.AuthenticationError as e:
            return False, f'Authentication failed: {e}'
        except Exception as e:
            return False, f'Cannot connect to OpenAI-compatible API: {e}'

    def check_models(self, models: list[str]) -> list[str]:
        """Return model names that don't exist on the remote API.

        Falls back to empty list if the models endpoint is unsupported
        (common with non-OpenAI compatible providers).
        """
        try:
            client = openai.OpenAI(api_key=self._api_key, base_url=self._base_url)
            missing = []
            for model in models:
                try:
                    client.models.retrieve(model)
                except openai.NotFoundError:
                    missing.append(model)
            return missing
        except Exception:
            return []
