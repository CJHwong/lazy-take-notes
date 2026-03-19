"""Validate SessionTemplate format variables and structure.

Business rule: each template field may only use specific format variables.
This catches errors that pass YAML/Pydantic validation but would crash at
runtime when .format() is called during a live recording.
"""

from __future__ import annotations

import string
from dataclasses import dataclass, field

from lazy_take_notes.l1_entities.template import SessionTemplate

# ── Allowed variables per field ──────────────────────────────────────────────

DIGEST_VARIABLES = frozenset({'line_count', 'new_lines', 'user_context'})
FINAL_VARIABLES = frozenset({'line_count', 'new_lines', 'user_context', 'full_transcript'})
QUICK_ACTION_VARIABLES = frozenset({'digest_markdown', 'recent_transcript'})

# Dummy values for dry-run formatting
_DIGEST_DUMMY = {k: '' for k in DIGEST_VARIABLES}
_FINAL_DUMMY = {k: '' for k in FINAL_VARIABLES}
_QUICK_ACTION_DUMMY = {k: '' for k in QUICK_ACTION_VARIABLES}

_FORMATTER = string.Formatter()


@dataclass
class TemplateValidationResult:
    """Collects all validation errors for a template."""

    errors: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return len(self.errors) == 0

    def __str__(self) -> str:
        if self.valid:
            return 'Template is valid.'
        return '\n'.join(f'- {e}' for e in self.errors)


def validate_template(template: SessionTemplate) -> TemplateValidationResult:
    """Run all validation checks on a SessionTemplate.

    Returns a result with a list of human-readable error strings.
    """
    result = TemplateValidationResult()

    _check_field_variables(
        result,
        'digest_user_template',
        template.digest_user_template,
        DIGEST_VARIABLES,
        _DIGEST_DUMMY,
    )
    _check_field_variables(
        result,
        'final_user_template',
        template.final_user_template,
        FINAL_VARIABLES,
        _FINAL_DUMMY,
    )
    for idx, action in enumerate(template.quick_actions):
        _check_field_variables(
            result,
            f'quick_actions[{idx}].prompt_template',
            action.prompt_template,
            QUICK_ACTION_VARIABLES,
            _QUICK_ACTION_DUMMY,
        )

    if not template.system_prompt.strip():
        result.errors.append('system_prompt is empty.')

    return result


def _check_field_variables(
    result: TemplateValidationResult,
    field_name: str,
    template_str: str,
    allowed: frozenset[str],
    dummy_values: dict[str, str],
) -> None:
    """Check that a template string only uses allowed format variables."""
    if not template_str.strip():
        return  # empty fields are allowed (Pydantic defaults)

    # Extract variable names from the format string
    used = _extract_field_names(template_str)
    unknown = used - allowed
    if unknown:
        allowed_list = ', '.join(f'{{{v}}}' for v in sorted(allowed))
        for var in sorted(unknown):
            result.errors.append(f'{field_name} uses unknown variable {{{var}}}. Allowed: {allowed_list}')

    # Dry-run: catch format spec errors, unclosed braces, etc.
    try:
        template_str.format(**dummy_values)
    except (KeyError, ValueError, IndexError) as exc:
        result.errors.append(f'{field_name} format error: {exc}')


def _extract_field_names(template_str: str) -> set[str]:
    """Extract all field names from a Python format string."""
    names: set[str] = set()
    try:
        for _, field_name, _, _ in _FORMATTER.parse(template_str):
            if field_name is not None:
                # Handle nested attributes like {foo.bar} — take root name
                root = field_name.split('.')[0].split('[')[0]
                if root:
                    names.add(root)
    except (ValueError, IndexError):
        pass  # malformed format string — caught by dry-run
    return names
