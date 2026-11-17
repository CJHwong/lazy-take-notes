"""Use case: run a digest cycle via the LLM client."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass

from lazy_take_notes.l1_entities.chat_message import ChatMessage
from lazy_take_notes.l1_entities.digest_state import DigestState
from lazy_take_notes.l1_entities.template import SessionTemplate
from lazy_take_notes.l2_use_cases.ports.llm_client import LLMClient
from lazy_take_notes.l2_use_cases.utils.prompt_builder import build_digest_prompt

log = logging.getLogger('ltn.llm')


@dataclass(frozen=True)
class DigestResult:
    """Result of a digest cycle — either success with markdown or failure with reason."""

    data: str | None = None
    error: str = ''

    @property
    def ok(self) -> bool:
        return self.data is not None


class RunDigestUseCase:
    """Runs one digest cycle: builds prompt, calls LLM, updates state."""

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm = llm_client

    async def execute(
        self,
        state: DigestState,
        model: str,
        template: SessionTemplate,
        *,
        is_final: bool = False,
        full_transcript: str = '',
        user_context: str = '',
        buffer_lines: list[str] | None = None,
    ) -> DigestResult:
        """Execute a digest cycle. Mutates state on success."""
        log.info(
            'Digest request: %d buffer lines, final=%s, msgs=%d, prompt_tokens~%d',
            len(state.buffer),
            is_final,
            len(state.messages),
            state.prompt_tokens,
        )

        # A hand-edited user template is only validated by the template builder,
        # so a bad placeholder must fail this digest, not the whole session.
        try:
            prompt = build_digest_prompt(
                template,
                buffer_lines if buffer_lines is not None else state.buffer,
                is_final=is_final,
                full_transcript=full_transcript,
                user_context=user_context,
            )
        except (KeyError, IndexError, ValueError) as e:
            state.consecutive_failures += 1
            err = f'Template error: {type(e).__name__}: {e}'
            log.error(err, exc_info=True)
            return DigestResult(error=err)
        prompt_message = ChatMessage(role='user', content=prompt)
        state.messages.append(prompt_message)

        try:
            resp = await self._llm.chat(model=model, messages=state.messages)
            raw = resp.content
            log.debug('LLM raw response (%d chars): %s', len(raw), raw[:500].replace('\n', '\\n'))

            if not raw.strip():
                state.consecutive_failures += 1
                _remove_prompt(state, prompt_message)
                err = 'Empty response from LLM'
                log.warning(err)
                return DigestResult(error=err)

        except asyncio.CancelledError:
            # A newer digest replaced this one. Drop the unanswered prompt so the
            # history keeps strict user/assistant pairs.
            _remove_prompt(state, prompt_message)
            raise
        except Exception as e:
            state.consecutive_failures += 1
            _remove_prompt(state, prompt_message)
            err = f'LLM error: {type(e).__name__}: {e}'
            log.error(err, exc_info=True)
            return DigestResult(error=err)

        # Success
        state.messages.append(ChatMessage(role='assistant', content=raw))
        state.consecutive_failures = 0
        state.digest_count += 1
        state.prompt_tokens = resp.prompt_tokens
        state.buffer.clear()
        state.last_digest_time = time.monotonic()
        log.info(
            'Digest #%d succeeded (prompt_tokens=%d)',
            state.digest_count,
            state.prompt_tokens,
        )

        return DigestResult(data=raw.strip())


def _remove_prompt(state: DigestState, prompt_message: ChatMessage) -> None:
    """Remove this digest's own prompt by identity.

    A replacing digest may have appended its prompt after this one, so the
    last message is not necessarily ours.
    """
    for index in range(len(state.messages) - 1, -1, -1):
        if state.messages[index] is prompt_message:
            del state.messages[index]
            return


def should_trigger_digest(
    state: DigestState,
    min_lines: int,
    min_interval: float,
    max_lines: int | None = None,
) -> bool:
    """Check if a digest cycle should trigger based on buffer size and elapsed time.

    Triggers when buffer >= min_lines AND elapsed >= min_interval.
    Force-triggers when buffer >= max_lines regardless of interval.
    max_lines defaults to 2×min_lines when None.
    """
    buf_size = len(state.buffer)
    if buf_size < min_lines:
        return False
    cap = max_lines if max_lines is not None else min_lines * 2
    if buf_size >= cap:
        return True
    elapsed = time.monotonic() - state.last_digest_time
    return elapsed >= min_interval
