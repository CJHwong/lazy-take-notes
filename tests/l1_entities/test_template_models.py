"""Tests for template Pydantic models."""

from lazy_take_notes.l1_entities.template import QuickAction, SessionTemplate, TemplateMetadata


class TestTemplateModels:
    def test_template_metadata_defaults(self):
        meta = TemplateMetadata()
        assert meta.name == ''
        assert meta.locale == ''

    def test_quick_action_creation(self):
        qa = QuickAction(key='1', label='Test', prompt_template='{digest_markdown}')
        assert qa.key == '1'
        assert qa.label == 'Test'

    def test_session_template_defaults(self):
        tmpl = SessionTemplate()
        assert tmpl.system_prompt == ''
        assert tmpl.quick_actions == []
