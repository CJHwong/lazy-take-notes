"""Tests for prompt builder pure functions."""

from lazy_take_notes.l2_use_cases.utils.prompt_builder import (
    build_compact_user_message,
    build_digest_prompt,
    build_quick_action_prompt,
)
from lazy_take_notes.l3_interface_adapters.gateways.yaml_template_loader import YamlTemplateLoader


class TestBuildDigestPrompt:
    def test_regular_prompt_has_placeholders_filled(self):
        tmpl = YamlTemplateLoader().load('default_zh_tw')
        buffer = ['Line 1', 'Line 2', 'Line 3']
        result = build_digest_prompt(tmpl, buffer)
        assert '3' in result  # line_count
        assert 'Line 1' in result
        assert 'Line 2' in result

    def test_final_prompt_includes_full_transcript(self):
        tmpl = YamlTemplateLoader().load('default_zh_tw')
        buffer = ['Line 1']
        result = build_digest_prompt(
            tmpl,
            buffer,
            is_final=True,
            full_transcript='Full session text',
        )
        assert 'Full session text' in result

    def test_final_prompt_fallback_when_no_transcript(self):
        tmpl = YamlTemplateLoader().load('default_zh_tw')
        result = build_digest_prompt(tmpl, ['x'], is_final=True, full_transcript='')
        assert '(no full transcript)' in result

    def test_user_context_appears_with_header(self):
        tmpl = YamlTemplateLoader().load('default_zh_tw')
        result = build_digest_prompt(tmpl, ['x'], user_context='John = speaker A')
        assert 'User corrections and additions:' in result
        assert 'John = speaker A' in result

    def test_empty_user_context_leaves_no_header(self):
        tmpl = YamlTemplateLoader().load('default_zh_tw')
        result = build_digest_prompt(tmpl, ['x'], user_context='')
        assert 'User corrections and additions:' not in result

    def test_whitespace_only_user_context_leaves_no_header(self):
        tmpl = YamlTemplateLoader().load('default_zh_tw')
        result = build_digest_prompt(tmpl, ['x'], user_context='   \n  ')
        assert 'User corrections and additions:' not in result


class TestBuildQuickActionPrompt:
    def test_fills_placeholders(self):
        result = build_quick_action_prompt(
            'Digest: {digest_markdown}\nRecent: {recent_transcript}',
            'Some digest',
            'Some transcript',
        )
        assert 'Some digest' in result
        assert 'Some transcript' in result

    def test_empty_digest_uses_fallback(self):
        result = build_quick_action_prompt('{digest_markdown}', '', 'text')
        assert '(no digest yet)' in result

    def test_user_context_appended_when_set(self):
        result = build_quick_action_prompt(
            '{digest_markdown}',
            'digest text',
            'transcript',
            user_context='Fix spelling: Huss = Hoss',
        )
        assert 'User corrections and context:' in result
        assert 'Fix spelling: Huss = Hoss' in result

    def test_empty_user_context_not_appended(self):
        result = build_quick_action_prompt('{digest_markdown}', 'digest', 'text', user_context='')
        assert 'User corrections' not in result


class TestBuildCompactUserMessage:
    def test_contains_markdown(self):
        result = build_compact_user_message('## Topic\nStuff')
        assert '## Topic' in result
        assert 'compacted' in result.lower()
