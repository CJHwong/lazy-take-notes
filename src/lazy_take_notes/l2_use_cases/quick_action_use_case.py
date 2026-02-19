"""Use case: execute a quick action by key."""

from __future__ import annotations

from lazy_take_notes.l1_entities.template import QuickAction, SessionTemplate
from lazy_take_notes.l1_entities.transcript import TranscriptSegment
from lazy_take_notes.l2_use_cases.ports.llm_client import LLMClient
from lazy_take_notes.l2_use_cases.query_use_case import RunQueryUseCase
from lazy_take_notes.l2_use_cases.utils.prompt_builder import build_quick_action_prompt


class RunQuickActionUseCase:
    """Finds a quick action by key, builds prompt, runs query."""

    def __init__(self, llm_client: LLMClient) -> None:
        self._query = RunQueryUseCase(llm_client)

    async def execute(
        self,
        key: str,
        template: SessionTemplate,
        model: str,
        latest_digest: str | None,
        all_segments: list[TranscriptSegment],
        *,
        user_context: str = '',
    ) -> tuple[str, str] | None:
        """Execute a quick action. Returns (result_text, label) or None if key not found."""
        qa = self._find_action(key, template)
        if qa is None:
            return None

        recent = all_segments[-50:] if all_segments else []
        recent_transcript = '\n'.join(seg.text for seg in recent)

        prompt = build_quick_action_prompt(
            qa.prompt_template,
            latest_digest or '(no digest yet)',
            recent_transcript,
            user_context=user_context,
        )

        result = await self._query.execute(prompt, model)
        return result, qa.label

    @staticmethod
    def _find_action(key: str, template: SessionTemplate) -> QuickAction | None:
        try:
            idx = int(key) - 1
        except ValueError:
            return None
        if 0 <= idx < len(template.quick_actions):
            return template.quick_actions[idx]
        return None
