"""Pure functions for building LLM prompts from templates."""

from __future__ import annotations

from lazy_take_notes.l1_entities.template import SessionTemplate


def build_digest_prompt(
    template: SessionTemplate,
    buffer: list[str],
    *,
    is_final: bool = False,
    full_transcript: str = '',
) -> str:
    """Build the user prompt for a digest cycle."""
    new_lines = '\n'.join(buffer)

    if is_final:
        return template.final_user_template.format(
            line_count=len(buffer),
            new_lines=new_lines,
            user_context='',
            full_transcript=full_transcript or '(no full transcript)',
        )
    return template.digest_user_template.format(
        line_count=len(buffer),
        new_lines=new_lines,
        user_context='',
    )


def build_quick_action_prompt(
    prompt_template: str,
    digest_markdown: str,
    recent_transcript: str,
) -> str:
    """Build the user prompt for a quick action."""
    return prompt_template.format(
        digest_markdown=digest_markdown or '(no digest yet)',
        recent_transcript=recent_transcript or '(no transcript yet)',
    )


def build_compact_user_message(latest_markdown: str) -> str:
    """Build the synthetic user message for conversation compaction."""
    return (
        '(Prior conversation compacted) Current session state:\n\n'
        f'{latest_markdown}\n\n'
        'Continue analyzing subsequent transcript segments based on this state.'
    )
