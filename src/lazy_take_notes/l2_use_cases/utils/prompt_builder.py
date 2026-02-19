"""Pure functions for building LLM prompts from templates."""

from __future__ import annotations

from lazy_take_notes.l1_entities.template import SessionTemplate


def build_digest_prompt(
    template: SessionTemplate,
    buffer: list[str],
    *,
    is_final: bool = False,
    full_transcript: str = '',
    user_context: str = '',
) -> str:
    """Build the user prompt for a digest cycle."""
    new_lines = '\n'.join(buffer)
    context_section = f'User corrections and additions:\n{user_context.strip()}' if user_context.strip() else ''

    if is_final:
        return template.final_user_template.format(
            line_count=len(buffer),
            new_lines=new_lines,
            user_context=context_section,
            full_transcript=full_transcript or '(no full transcript)',
        )
    return template.digest_user_template.format(
        line_count=len(buffer),
        new_lines=new_lines,
        user_context=context_section,
    )


def build_quick_action_prompt(
    prompt_template: str,
    digest_markdown: str,
    recent_transcript: str,
    *,
    user_context: str = '',
) -> str:
    """Build the user prompt for a quick action."""
    result = prompt_template.format(
        digest_markdown=digest_markdown or '(no digest yet)',
        recent_transcript=recent_transcript or '(no transcript yet)',
    )
    if user_context.strip():
        result += f'\n\nUser corrections and context:\n{user_context.strip()}'
    return result


def build_compact_user_message(latest_markdown: str) -> str:
    """Build the synthetic user message for conversation compaction."""
    return (
        '(Prior conversation compacted) Current session state:\n\n'
        f'{latest_markdown}\n\n'
        'Continue analyzing subsequent transcript segments based on this state.'
    )
