"""Tests for template Pydantic models."""

import pytest

from lazy_take_notes.l1_entities.template import QuickAction, SessionTemplate, TemplateMetadata


class TestTemplateModels:
    def test_template_metadata_defaults(self):
        meta = TemplateMetadata()
        assert not meta.name
        assert not meta.locale

    def test_quick_action_creation(self):
        qa = QuickAction(label='Test', prompt_template='{digest_markdown}')
        assert qa.label == 'Test'

    def test_session_template_rejects_more_than_5_quick_actions(self):
        from pydantic import ValidationError

        actions = [QuickAction(label=f'Action {i}', prompt_template='x') for i in range(6)]
        with pytest.raises(ValidationError, match='At most 5 quick_actions'):
            SessionTemplate(quick_actions=actions)

    def test_session_template_defaults(self):
        tmpl = SessionTemplate()
        assert not tmpl.system_prompt
        assert tmpl.quick_actions == []
