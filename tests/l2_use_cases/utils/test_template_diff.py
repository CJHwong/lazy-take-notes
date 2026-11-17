"""diff_templates: the exact list of what an LLM edit changed, so unasked edits are visible."""

from __future__ import annotations

from lazy_take_notes.l1_entities.template import QuickAction, SessionTemplate, TemplateMetadata
from lazy_take_notes.l2_use_cases.utils.template_diff import diff_templates


def _template(**overrides) -> SessionTemplate:
    base = SessionTemplate(
        metadata=TemplateMetadata(name='Standup', description='Daily', locale='en', key='standup'),
        system_prompt='Be brief.',
        digest_user_template='{new_lines}',
        final_user_template='{full_transcript}',
        recognition_hints=['Jira'],
        quick_actions=[
            QuickAction(label='Blockers', prompt_template='{digest_markdown}'),
            QuickAction(label='Status', prompt_template='{recent_transcript}'),
        ],
    )
    return base.model_copy(update=overrides)


def _actions(*labels: str) -> list[QuickAction]:
    return [QuickAction(label=label, prompt_template='{digest_markdown}') for label in labels]


def test_an_identical_template_has_no_changes():
    assert diff_templates(_template(), _template()) == []


def test_the_file_key_is_not_a_change():
    # The loader sets the key; the LLM never returns it.
    edited = _template(metadata=TemplateMetadata(name='Standup', description='Daily', locale='en'))
    assert diff_templates(_template(), edited) == []


def test_text_fields_are_named():
    edited = _template(
        metadata=TemplateMetadata(name='Standup', description='Daily', locale='zh-TW'),
        system_prompt='Be thorough.',
        final_user_template='{full_transcript} done',
    )
    assert diff_templates(_template(), edited) == ['locale', 'system prompt', 'final summary prompt']


def test_recognition_hints_list_what_was_added_and_removed():
    edited = _template(recognition_hints=['Confluence'])
    assert diff_templates(_template(), edited) == ["recognition hints (+'Confluence', -'Jira')"]


def test_quick_actions_are_matched_by_label():
    old = _template()
    changed_status = QuickAction(label='Status', prompt_template='{digest_markdown} now')
    edited = _template(quick_actions=[changed_status, *_actions('Risks')])
    assert diff_templates(old, edited) == [
        "quick action 'Blockers' removed",
        "quick action 'Status' changed",
        "quick action 'Risks' added",
    ]


def test_a_reorder_is_reported_because_keys_are_positional():
    old = _template(quick_actions=_actions('Blockers', 'Status'))
    edited = _template(quick_actions=_actions('Status', 'Blockers'))
    assert diff_templates(old, edited) == ['quick action order (keys 1-5 follow it)']
