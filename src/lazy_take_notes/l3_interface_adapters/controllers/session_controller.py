"""SessionController — orchestrates use cases, manages state, posts Textual messages."""

from __future__ import annotations

import logging
from collections.abc import Callable

from lazy_take_notes.l1_entities.config import AppConfig
from lazy_take_notes.l1_entities.digest_state import DigestState
from lazy_take_notes.l1_entities.speaker_turn import SpeakerTurn
from lazy_take_notes.l1_entities.template import SessionTemplate
from lazy_take_notes.l1_entities.transcript import TranscriptSegment
from lazy_take_notes.l2_use_cases.compact_messages_use_case import CompactMessagesUseCase
from lazy_take_notes.l2_use_cases.digest_use_case import DigestResult, RunDigestUseCase, should_trigger_digest
from lazy_take_notes.l2_use_cases.generate_label_use_case import GenerateLabelUseCase
from lazy_take_notes.l2_use_cases.ports.llm_client import LLMClient
from lazy_take_notes.l2_use_cases.ports.persistence import PersistenceGateway
from lazy_take_notes.l2_use_cases.quick_action_use_case import RunQuickActionUseCase
from lazy_take_notes.l2_use_cases.speaker_attribution import speaker_for_span

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
        self._label_uc = GenerateLabelUseCase(llm_client)

        self.digest_state = DigestState()
        self.digest_state.init_messages(template.system_prompt)

        self.all_segments: list[TranscriptSegment] = []
        self.latest_digest: str | None = None
        self.user_context: str = ''
        self.speaker_turns: list[SpeakerTurn] = []
        self.speaker_names: dict[int, str] = {}
        self._persist_error: str | None = None
        self._template_saved = False

    def _persist(self, what: str, write: Callable[[], object]) -> None:
        """Run one disk write. A full or vanished disk must not end the recording."""
        try:
            write()
        except (OSError, UnicodeError) as e:
            log.error('Could not save %s: %s', what, e, exc_info=True)
            self._persist_error = f'Could not save {what}: {e}'

    def take_persist_error(self) -> str | None:
        """Return the newest unreported write failure, then clear it."""
        error, self._persist_error = self._persist_error, None
        return error

    def save_context_now(self) -> None:
        """Write the session context outside a final digest, e.g. on a shutdown signal."""
        if self.user_context.strip() and self._config.output.save_context:
            self._persist('session context', lambda: self._persistence.save_session_context(self.user_context))

    def set_speaker_turns(self, turns: list[SpeakerTurn]) -> None:
        """Replace the acoustic speaker turns used for labels and prompts."""
        self.speaker_turns = turns

    def set_speaker_name(self, speaker: int, name: str) -> None:
        """Attach a human name to an acoustic speaker track."""
        self.speaker_names[speaker] = name

    def speaker_label(self, segment: TranscriptSegment) -> str | None:
        """Label for one segment: the mapped name, else S<n>, else None."""
        speaker = speaker_for_span(self.speaker_turns, segment.wall_start, segment.wall_end)
        if speaker is None:
            return None
        return self.speaker_names.get(speaker) or f'S{speaker}'

    def _labeled_lines(self, segments: list[TranscriptSegment]) -> list[str]:
        lines: list[str] = []
        for segment in segments:
            label = self.speaker_label(segment)
            lines.append(f'{label}: {segment.text}' if label else segment.text)
        return lines

    def on_transcript_segments(self, segments: list[TranscriptSegment]) -> bool:
        """Process new transcript segments. Returns True if digest should trigger."""
        self.all_segments.extend(segments)
        for seg in segments:
            self.digest_state.buffer.append(seg.text)
            self.digest_state.all_lines.append(seg.text)

        self._persist('transcript', lambda: self._persistence.save_transcript_lines(segments, append=True))
        # With the first lines, not at start: a session cancelled before any
        # speech must not leave a folder that looks like a session.
        if not self._template_saved:
            self._template_saved = True
            name = self._template.metadata.name
            self._persist('template name', lambda: self._persistence.save_template_name(name))

        dc = self._config.digest
        return should_trigger_digest(self.digest_state, dc.min_lines, dc.min_interval, dc.max_lines)

    async def run_digest(self, *, is_final: bool = False) -> DigestResult:
        """Run a digest cycle."""
        # Speak-labeled text for the prompt: the same labels the TUI shows.
        buffer_lines: list[str] | None = None
        if self.speaker_turns:
            recent = self.all_segments[-len(self.digest_state.buffer) :] if self.digest_state.buffer else []
            buffer_lines = self._labeled_lines(recent)

        full_transcript = ''
        if is_final:
            if self.speaker_turns:
                full_transcript = '\n'.join(self._labeled_lines(self.all_segments))
            else:
                full_transcript = '\n'.join(self.digest_state.all_lines)

        result = await self._digest_uc.execute(
            state=self.digest_state,
            model=self._config.digest.model,
            template=self._template,
            is_final=is_final,
            full_transcript=full_transcript,
            user_context=self.user_context,
            buffer_lines=buffer_lines,
        )

        if result.data is not None:
            self.latest_digest = result.data
            markdown, number = result.data, self.digest_state.digest_count
            self._persist('notes', lambda: self._persistence.save_digest_md(markdown, number))
            if self._config.output.save_notes_history:
                self._persist(
                    'notes history',
                    lambda: self._persistence.save_history(markdown, number, is_final=is_final),
                )
            if is_final:
                self.save_context_now()

            # Compact if needed
            if self.digest_state.prompt_tokens > self._config.digest.compact_token_threshold:
                self._compact_uc.execute(
                    self.digest_state,
                    result.data,
                    self._template.system_prompt,
                )

        return result

    async def generate_label(self) -> str | None:
        """Generate a short session label from the latest digest. Returns None on failure."""
        if self.latest_digest is None:
            return None
        try:
            label = await self._label_uc.execute(
                self._template.metadata.name,
                self._template.metadata.description or '',
                self.latest_digest,
                self._config.interactive.model,
            )
            return label or None
        except Exception:
            log.warning('Auto-label generation failed', exc_info=True)
            return None

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
