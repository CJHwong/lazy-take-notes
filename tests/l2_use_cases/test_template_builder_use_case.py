"""Tests for TemplateBuildUseCase — uses FakeLLMClient, no library mocking."""

from __future__ import annotations

import json

import pytest

from lazy_take_notes.l1_entities.chat_message import ChatMessage
from lazy_take_notes.l1_entities.template import SessionTemplate
from lazy_take_notes.l2_use_cases.template_builder_use_case import (
    BuilderResult,
    TemplateBuildUseCase,
)
from tests.conftest import FakeLLMClient

_VALID_TEMPLATE_DICT = {
    'metadata': {
        'name': 'Test Template',
        'description': 'A test template',
        'locale': 'en',
    },
    'system_prompt': 'You are a meeting assistant.',
    'digest_user_template': 'New lines ({line_count}):\n{new_lines}\n{user_context}',
    'final_user_template': ('Final ({line_count}):\n{new_lines}\n{user_context}\n{full_transcript}'),
    'quick_actions': [
        {
            'label': 'Summary',
            'description': 'Quick summary',
            'prompt_template': 'Summary:\n{digest_markdown}\n{recent_transcript}',
        },
    ],
}

_VALID_JSON = json.dumps(_VALID_TEMPLATE_DICT, indent=2)


def _make_use_case(fake_llm: FakeLLMClient) -> TemplateBuildUseCase:
    """Build a TemplateBuildUseCase with a minimal example template."""
    example = SessionTemplate.model_validate(_VALID_TEMPLATE_DICT)
    return TemplateBuildUseCase(llm_client=fake_llm, example_template=example)


def _wrap_json(json_str: str, preamble: str = 'Here is your template:') -> str:
    return f'{preamble}\n\n```json\n{json_str}\n```'


class TestGenerate:
    @pytest.mark.asyncio
    async def test_valid_template_returned(self):
        fake_llm = FakeLLMClient(response=_wrap_json(_VALID_JSON))
        use_case = _make_use_case(fake_llm)
        history: list[ChatMessage] = []

        result = await use_case.generate('Make me a meeting template', 'test-model', history)

        assert result.template is not None
        assert result.template.metadata.name == 'Test Template'
        assert not result.validation_errors
        assert not result.error
        assert result.assistant_message  # has conversational text

    @pytest.mark.asyncio
    async def test_no_json_returns_conversational_response(self):
        fake_llm = FakeLLMClient(response='What language should the template be in?')
        use_case = _make_use_case(fake_llm)
        history: list[ChatMessage] = []

        result = await use_case.generate('I want a template', 'test-model', history)

        assert result.template is None
        assert not result.error
        assert not result.validation_errors
        assert 'language' in result.assistant_message.lower()

    @pytest.mark.asyncio
    async def test_invalid_json_returns_error(self):
        fake_llm = FakeLLMClient(response=_wrap_json('{not valid json'))
        use_case = _make_use_case(fake_llm)
        history: list[ChatMessage] = []

        result = await use_case.generate('Make template', 'test-model', history)

        assert result.template is None
        assert 'Invalid JSON' in result.error

    @pytest.mark.asyncio
    async def test_bad_format_variables_returns_validation_errors(self):
        bad_template = dict(_VALID_TEMPLATE_DICT)
        bad_template['digest_user_template'] = 'Uses {bogus_var}'
        bad_json = json.dumps(bad_template)
        fake_llm = FakeLLMClient(response=_wrap_json(bad_json))
        use_case = _make_use_case(fake_llm)
        history: list[ChatMessage] = []

        result = await use_case.generate('Make template', 'test-model', history)

        assert result.template is not None  # parsed OK, but invalid vars
        assert 'bogus_var' in result.validation_errors

    @pytest.mark.asyncio
    async def test_schema_validation_failure(self):
        # quick_actions > 5 triggers Pydantic validator
        bad = dict(_VALID_TEMPLATE_DICT)
        bad['quick_actions'] = [{'label': f'Action {i}', 'prompt_template': '{digest_markdown}'} for i in range(6)]
        fake_llm = FakeLLMClient(response=_wrap_json(json.dumps(bad)))
        use_case = _make_use_case(fake_llm)
        history: list[ChatMessage] = []

        result = await use_case.generate('Make template', 'test-model', history)

        assert result.template is None
        assert 'Schema validation failed' in result.error

    @pytest.mark.asyncio
    async def test_history_maintained_across_calls(self):
        fake_llm = FakeLLMClient(response='What kind of meeting?')
        use_case = _make_use_case(fake_llm)
        history: list[ChatMessage] = []

        await use_case.generate('I want a template', 'test-model', history)
        assert len(history) == 2  # user + assistant

        fake_llm.set_response(_wrap_json(_VALID_JSON))
        await use_case.generate('English, for standups', 'test-model', history)
        assert len(history) == 4  # 2 more

    @pytest.mark.asyncio
    async def test_system_prompt_sent_to_llm(self):
        fake_llm = FakeLLMClient(response='ok')
        use_case = _make_use_case(fake_llm)
        history: list[ChatMessage] = []

        await use_case.generate('hello', 'test-model', history)

        model, messages = fake_llm.chat_calls[0]
        assert model == 'test-model'
        assert messages[0].role == 'system'
        assert 'template builder' in messages[0].content.lower()


class TestAutoFix:
    @pytest.mark.asyncio
    async def test_auto_fix_sends_error_feedback(self):
        fake_llm = FakeLLMClient(response=_wrap_json(_VALID_JSON))
        use_case = _make_use_case(fake_llm)
        history: list[ChatMessage] = []

        result = await use_case.auto_fix(
            'digest_user_template uses unknown variable {bogus}',
            'test-model',
            history,
        )

        assert result.template is not None
        # The fix prompt should be in history
        assert any('validation errors' in msg.content.lower() for msg in history if msg.role == 'user')

    @pytest.mark.asyncio
    async def test_auto_fix_with_still_invalid_response(self):
        fake_llm = FakeLLMClient(response=_wrap_json('{bad json'))
        use_case = _make_use_case(fake_llm)
        history: list[ChatMessage] = []

        result = await use_case.auto_fix('some error', 'test-model', history)

        assert result.template is None
        assert result.error


class TestBuilderResult:
    def test_defaults(self):
        result = BuilderResult()
        assert result.template is None
        assert not result.validation_errors
        assert not result.assistant_message
        assert not result.error
