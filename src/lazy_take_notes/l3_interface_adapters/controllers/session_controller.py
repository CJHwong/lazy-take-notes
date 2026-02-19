"""SessionController — orchestrates use cases, manages state, posts Textual messages."""

from __future__ import annotations

import logging

from lazy_take_notes.l1_entities.config import AppConfig
from lazy_take_notes.l1_entities.digest_state import DigestState
from lazy_take_notes.l1_entities.template import SessionTemplate
from lazy_take_notes.l1_entities.transcript import TranscriptSegment
from lazy_take_notes.l2_use_cases.compact_messages_use_case import CompactMessagesUseCase
from lazy_take_notes.l2_use_cases.digest_use_case import DigestResult, RunDigestUseCase, should_trigger_digest
from lazy_take_notes.l2_use_cases.ports.llm_client import LLMClient
from lazy_take_notes.l2_use_cases.ports.persistence import PersistenceGateway
from lazy_take_notes.l2_use_cases.quick_action_use_case import RunQuickActionUseCase

log = logging.getLogger('ltn.controller')


class SessionController:
    """Central orchestrator bridging use cases to the TUI.

    Owns DigestState, all_segments, latest_digest. The App (L4) delegates
    all business decisions to this controller.
    """

    def __init__(
        self,
        config: AppConfig,
        template: SessionTemplate,
        llm_client: LLMClient,
        persistence: PersistenceGateway,
    ) -> None:
        self._config = config
        self._template = template
        self._persistence = persistence

        self._digest_uc = RunDigestUseCase(llm_client)
        self._compact_uc = CompactMessagesUseCase()
        self._quick_action_uc = RunQuickActionUseCase(llm_client)

        self.digest_state = DigestState()
        self.digest_state.init_messages(template.system_prompt)

        self.all_segments: list[TranscriptSegment] = []
        self.latest_digest: str | None = None
        self.user_context: str = ''

    def on_transcript_segments(self, segments: list[TranscriptSegment]) -> bool:
        """Process new transcript segments. Returns True if digest should trigger."""
        self.all_segments.extend(segments)
        for seg in segments:
            self.digest_state.buffer.append(seg.text)
            self.digest_state.all_lines.append(seg.text)

        self._persistence.save_transcript_lines(segments, append=True)

        dc = self._config.digest
        return should_trigger_digest(self.digest_state, dc.min_lines, dc.min_interval, dc.max_lines)

    async def run_digest(self, *, is_final: bool = False) -> DigestResult:
        """Run a digest cycle."""
        full_transcript = ''
        if is_final:
            full_transcript = '\n'.join(self.digest_state.all_lines)

        result = await self._digest_uc.execute(
            state=self.digest_state,
            model=self._config.digest.model,
            template=self._template,
            is_final=is_final,
            full_transcript=full_transcript,
            user_context=self.user_context,
        )

        if result.data is not None:
            self.latest_digest = result.data
            self._persistence.save_digest_md(result.data, self.digest_state.digest_count)
            self._persistence.save_history(
                result.data,
                self.digest_state.digest_count,
                is_final=is_final,
            )
            if is_final and self.user_context.strip():
                self._persistence.save_session_context(self.user_context)

            # Compact if needed
            if self.digest_state.prompt_tokens > self._config.digest.compact_token_threshold:
                self._compact_uc.execute(
                    self.digest_state,
                    result.data,
                    self._template.system_prompt,
                )

        return result

    async def run_quick_action(self, key: str) -> tuple[str, str] | None:
        """Execute a quick action by key. Returns (result, label) or None."""
        return await self._quick_action_uc.execute(
            key=key,
            template=self._template,
            model=self._config.interactive.model,
            latest_digest=self.latest_digest,
            all_segments=self.all_segments,
            user_context=self.user_context,
        )
