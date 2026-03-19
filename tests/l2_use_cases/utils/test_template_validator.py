"""Tests for template_validator — format variable validation and dry-run checks."""

from __future__ import annotations

from lazy_take_notes.l1_entities.template import QuickAction, SessionTemplate, TemplateMetadata
from lazy_take_notes.l2_use_cases.utils.template_validator import (
    DIGEST_VARIABLES,
    FINAL_VARIABLES,
    QUICK_ACTION_VARIABLES,
    TemplateValidationResult,
    validate_template,
)


def _make_template(**overrides: object) -> SessionTemplate:
    """Build a valid template with optional field overrides."""
    defaults: dict[str, object] = {
        'metadata': TemplateMetadata(name='Test', locale='en'),
        'system_prompt': 'You are a meeting assistant.',
        'digest_user_template': 'Lines ({line_count}):\n{new_lines}\n{user_context}',
        'final_user_template': 'Final ({line_count}):\n{new_lines}\n{user_context}\n{full_transcript}',
        'quick_actions': [
            QuickAction(label='Summary', prompt_template='{digest_markdown}\n{recent_transcript}'),
        ],
    }
    defaults.update(overrides)
    return SessionTemplate.model_validate(defaults)


class TestValidateTemplateValid:
    def test_valid_template_passes(self):
        result = validate_template(_make_template())
        assert result.valid
        assert result.errors == []

    def test_empty_optional_fields_pass(self):
        result = validate_template(
            _make_template(
                digest_user_template='',
                final_user_template='',
                quick_actions=[],
            )
        )
        assert result.valid

    def test_no_quick_actions_passes(self):
        result = validate_template(_make_template(quick_actions=[]))
        assert result.valid


class TestValidateTemplateDigestVars:
    def test_unknown_variable_in_digest(self):
        result = validate_template(
            _make_template(
                digest_user_template='Hello {bogus_var}',
            )
        )
        assert not result.valid
        assert any('bogus_var' in e for e in result.errors)

    def test_all_allowed_digest_variables(self):
        template_str = ' '.join(f'{{{v}}}' for v in DIGEST_VARIABLES)
        result = validate_template(_make_template(digest_user_template=template_str))
        assert result.valid

    def test_final_only_var_in_digest_rejected(self):
        result = validate_template(
            _make_template(
                digest_user_template='{full_transcript}',
            )
        )
        assert not result.valid
        assert any('full_transcript' in e for e in result.errors)


class TestValidateTemplateFinalVars:
    def test_unknown_variable_in_final(self):
        result = validate_template(
            _make_template(
                final_user_template='Hello {nonexistent}',
            )
        )
        assert not result.valid
        assert any('nonexistent' in e for e in result.errors)

    def test_all_allowed_final_variables(self):
        template_str = ' '.join(f'{{{v}}}' for v in FINAL_VARIABLES)
        result = validate_template(_make_template(final_user_template=template_str))
        assert result.valid


class TestValidateTemplateQuickActionVars:
    def test_unknown_variable_in_quick_action(self):
        qa = QuickAction(label='Bad', prompt_template='{unknown_var}')
        result = validate_template(_make_template(quick_actions=[qa]))
        assert not result.valid
        assert any('unknown_var' in e for e in result.errors)

    def test_all_allowed_quick_action_variables(self):
        template_str = ' '.join(f'{{{v}}}' for v in QUICK_ACTION_VARIABLES)
        qa = QuickAction(label='Good', prompt_template=template_str)
        result = validate_template(_make_template(quick_actions=[qa]))
        assert result.valid

    def test_multiple_quick_actions_validated(self):
        qa1 = QuickAction(label='OK', prompt_template='{digest_markdown}')
        qa2 = QuickAction(label='Bad', prompt_template='{nope}')
        result = validate_template(_make_template(quick_actions=[qa1, qa2]))
        assert not result.valid
        assert any('quick_actions[1]' in e for e in result.errors)


class TestValidateTemplateSystemPrompt:
    def test_empty_system_prompt_rejected(self):
        result = validate_template(_make_template(system_prompt=''))
        assert not result.valid
        assert any('system_prompt' in e for e in result.errors)

    def test_whitespace_only_system_prompt_rejected(self):
        result = validate_template(_make_template(system_prompt='   \n  '))
        assert not result.valid
        assert any('system_prompt' in e for e in result.errors)


class TestValidateTemplateFormatErrors:
    def test_malformed_format_string(self):
        result = validate_template(
            _make_template(
                digest_user_template='{line_count} {unclosed',
            )
        )
        assert not result.valid
        assert any('format error' in e for e in result.errors)

    def test_unclosed_brace_caught_by_extract(self):
        # Malformed format string where parse() raises ValueError
        # _extract_field_names catches it silently, but dry-run catches the error
        result = validate_template(
            _make_template(
                digest_user_template='{line_count} {',
            )
        )
        assert not result.valid


class TestTemplateValidationResult:
    def test_str_valid(self):
        result = TemplateValidationResult()
        assert str(result) == 'Template is valid.'

    def test_str_with_errors(self):
        result = TemplateValidationResult(errors=['error one', 'error two'])
        text = str(result)
        assert '- error one' in text
        assert '- error two' in text

    def test_valid_property(self):
        assert TemplateValidationResult().valid is True
        assert TemplateValidationResult(errors=['x']).valid is False
