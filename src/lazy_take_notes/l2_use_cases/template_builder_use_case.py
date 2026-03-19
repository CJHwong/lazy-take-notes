"""Use case: LLM-powered template builder.

Takes a user description, runs it through an LLM with schema guidance,
extracts JSON from the response, validates it as a SessionTemplate.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from lazy_take_notes.l1_entities.chat_message import ChatMessage
from lazy_take_notes.l1_entities.template import SessionTemplate
from lazy_take_notes.l2_use_cases.ports.llm_client import LLMClient
from lazy_take_notes.l2_use_cases.utils.template_validator import (
    DIGEST_VARIABLES,
    FINAL_VARIABLES,
    QUICK_ACTION_VARIABLES,
    validate_template,
)

_JSON_FENCE_RE = re.compile(r'```json\s*\n(.*?)\n\s*```', re.DOTALL)

_SYSTEM_PROMPT = """\
You are a template builder for a live transcription and AI digest app.

Your job: help the user create a session template. The template controls how the \
AI summarizes a live transcript during a meeting, lecture, or other audio session.

## Template Schema (JSON)

{schema}

## Field descriptions

- **metadata**: name, description, locale (e.g. "en", "zh-TW"), key (leave empty).
- **system_prompt**: Instructions for the AI summarizer. Defines tone, structure, and rules.
- **digest_user_template**: The prompt sent each time new transcript lines arrive. \
Available variables: {digest_vars}. The AI uses this to produce an updated summary.
- **final_user_template**: The prompt sent when the session ends. \
Available variables: {final_vars}. Should instruct the AI to produce a final, \
comprehensive summary.
- **recognition_hints**: List of domain-specific words/names to help speech recognition.
- **quick_actions**: Up to 5 shortcut prompts (label, description, prompt_template). \
Available variables: {qa_vars}.

## User-facing abstraction

The user describes "what the summary should look like" — you decide how to fill \
digest_user_template and final_user_template appropriately. Don't ask the user \
about these fields directly.

## Behavior

- Be thorough before generating. On the first message, ALWAYS ask 1-3 focused \
follow-up questions to clarify what the user wants. Key things to probe for:
  - Language/locale (e.g. English, Chinese, etc.)
  - What sections the summary should have (action items? decisions? open questions?)
  - Tone and audience (formal minutes vs casual notes)
  - Any quick actions they'd find useful
  - Domain-specific vocabulary (recognition hints)
- Once you have enough context, ALWAYS confirm before generating. Summarize your \
understanding and ask "Should I generate the template now?" or similar. Only \
generate after the user confirms (e.g. "yes", "go ahead", "looks good").
- When generating, output the complete JSON wrapped in ```json fences.
- Always produce valid JSON that conforms to the schema above.

## Example template (for reference)

```json
{example}
```
"""


def _format_vars(varset: frozenset[str]) -> str:
    return ', '.join(f'{{{v}}}' for v in sorted(varset))


def _build_system_prompt(example_template: SessionTemplate) -> str:
    schema = json.dumps(SessionTemplate.model_json_schema(), indent=2)
    example = json.dumps(example_template.model_dump(), indent=2, ensure_ascii=False)
    return _SYSTEM_PROMPT.format(
        schema=schema,
        digest_vars=_format_vars(DIGEST_VARIABLES),
        final_vars=_format_vars(FINAL_VARIABLES),
        qa_vars=_format_vars(QUICK_ACTION_VARIABLES),
        example=example,
    )


@dataclass
class BuilderResult:
    """Result of a template generation attempt."""

    template: SessionTemplate | None = None
    validation_errors: str = ''
    assistant_message: str = ''
    error: str = ''


@dataclass
class TemplateBuildUseCase:
    """Orchestrates LLM-powered template generation."""

    llm_client: LLMClient
    example_template: SessionTemplate
    _system_prompt: str = field(init=False)

    def __post_init__(self) -> None:
        self._system_prompt = _build_system_prompt(self.example_template)

    async def generate(
        self,
        user_message: str,
        model: str,
        history: list[ChatMessage],
    ) -> BuilderResult:
        """Send user message, parse response for JSON template."""
        history.append(ChatMessage(role='user', content=user_message))

        messages = [ChatMessage(role='system', content=self._system_prompt), *history]
        response = await self.llm_client.chat(model, messages)
        assistant_text = response.content

        history.append(ChatMessage(role='assistant', content=assistant_text))

        return self._parse_response(assistant_text)

    async def auto_fix(
        self,
        validation_errors: str,
        model: str,
        history: list[ChatMessage],
    ) -> BuilderResult:
        """Ask the LLM to fix validation errors from its previous output."""
        fix_prompt = (
            f'The generated template has validation errors:\n{validation_errors}\n\n'
            'Please fix these errors and output the corrected JSON in ```json fences.'
        )
        return await self.generate(fix_prompt, model, history)

    def _parse_response(self, assistant_text: str) -> BuilderResult:
        """Extract JSON from response, validate as SessionTemplate."""
        match = _JSON_FENCE_RE.search(assistant_text)
        if not match:
            # No JSON — LLM is asking questions, which is fine
            return BuilderResult(assistant_message=assistant_text)

        json_str = match.group(1)

        # Strip conversational text around the JSON fence for display
        conversational = _JSON_FENCE_RE.sub('', assistant_text).strip()

        try:
            raw = json.loads(json_str)
        except json.JSONDecodeError as exc:
            return BuilderResult(
                assistant_message=conversational or assistant_text,
                error=f'Invalid JSON: {exc}',
            )

        try:
            template = SessionTemplate.model_validate(raw)
        except Exception as exc:  # noqa: BLE001 -- Pydantic can raise various errors
            return BuilderResult(
                assistant_message=conversational or assistant_text,
                error=f'Schema validation failed: {exc}',
            )

        validation = validate_template(template)
        if not validation.valid:
            return BuilderResult(
                template=template,
                validation_errors=str(validation),
                assistant_message=conversational or assistant_text,
            )

        return BuilderResult(
            template=template,
            assistant_message=conversational or assistant_text,
        )
