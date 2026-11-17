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
    async def test_a_failed_call_leaves_the_history_unchanged(self):
        # A user turn with no reply would make the next request send two user turns in a row.
        class DeadLLM(FakeLLMClient):
            async def chat(self, model, messages):
                raise ConnectionError('Connection refused')

        history = [ChatMessage(role='user', content='hi'), ChatMessage(role='assistant', content='hello')]
        with pytest.raises(ConnectionError):
            await _make_use_case(DeadLLM()).generate('I want a template', 'test-model', history)
        assert [message.content for message in history] == ['hi', 'hello']

    @pytest.mark.asyncio
    async def test_the_new_message_is_sent_after_the_history(self):
        fake_llm = FakeLLMClient()
        history = [ChatMessage(role='user', content='hi'), ChatMessage(role='assistant', content='hello')]
        await _make_use_case(fake_llm).generate('next', 'test-model', history)
        sent = fake_llm.chat_calls[-1][1]
        assert [message.content for message in sent[1:]] == ['hi', 'hello', 'next']

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


def _make_edit_use_case(fake_llm: FakeLLMClient, base: SessionTemplate) -> TemplateBuildUseCase:
    example = SessionTemplate.model_validate(_VALID_TEMPLATE_DICT)
    return TemplateBuildUseCase(llm_client=fake_llm, example_template=example, base_template=base)


def _system_prompt(fake_llm: FakeLLMClient) -> str:
    return fake_llm.chat_calls[-1][1][0].content


class TestEditMode:
    """Editing feeds the current template in, and asks before it changes anything."""

    @pytest.mark.asyncio
    async def test_the_edit_prompt_carries_the_template_and_asks_first(self):
        fake_llm = FakeLLMClient(response='Apply this change?')
        base = SessionTemplate.model_validate(_VALID_TEMPLATE_DICT)
        await _make_edit_use_case(fake_llm, base).generate('add a risks action', 'm', [])
        prompt = _system_prompt(fake_llm)
        assert '"system_prompt": "You are a meeting assistant."' in prompt
        assert 'Apply this change?' in prompt
        assert 'ALWAYS ask 1-3 focused' not in prompt  # the create interview does not apply

    @pytest.mark.asyncio
    async def test_the_create_prompt_has_no_current_template(self):
        fake_llm = FakeLLMClient()
        await _make_use_case(fake_llm).generate('hi', 'm', [])
        assert '## Current template' not in _system_prompt(fake_llm)

    @pytest.mark.asyncio
    async def test_a_second_edit_builds_on_the_first(self):
        base = SessionTemplate.model_validate(_VALID_TEMPLATE_DICT)
        edited = {**_VALID_TEMPLATE_DICT, 'system_prompt': 'You are a terse assistant.'}
        fake_llm = FakeLLMClient(response=_wrap_json(json.dumps(edited)))
        use_case = _make_edit_use_case(fake_llm, base)
        history: list[ChatMessage] = []
        await use_case.generate('yes', 'm', history)

        fake_llm.set_response('Apply this change?')
        await use_case.generate('also shorter', 'm', history)
        prompt = _system_prompt(fake_llm)
        assert 'You are a terse assistant.' in prompt
        assert 'You are a meeting assistant.' not in prompt

    @pytest.mark.asyncio
    async def test_an_accepted_version_is_not_repeated_in_the_history(self):
        # The prompt already holds the newest template; old copies would contradict it.
        base = SessionTemplate.model_validate(_VALID_TEMPLATE_DICT)
        fake_llm = FakeLLMClient(response=_wrap_json(_VALID_JSON, preamble='Done.'))
        history: list[ChatMessage] = []
        await _make_edit_use_case(fake_llm, base).generate('yes', 'm', history)
        assert '```json' not in history[-1].content
        assert history[-1].content.startswith('Done.')

    @pytest.mark.asyncio
    async def test_an_invalid_version_stays_in_the_history_for_the_fix(self):
        base = SessionTemplate.model_validate(_VALID_TEMPLATE_DICT)
        broken = {**_VALID_TEMPLATE_DICT, 'digest_user_template': 'Notes {no_such_field}'}
        fake_llm = FakeLLMClient(response=_wrap_json(json.dumps(broken)))
        use_case = _make_edit_use_case(fake_llm, base)
        history: list[ChatMessage] = []
        result = await use_case.generate('yes', 'm', history)
        assert result.validation_errors
        assert '```json' in history[-1].content
        assert use_case.current_template is base


class TestJsonOnlyReply:
    """A reply that is only the JSON fence must not show raw JSON as the chat message."""

    @pytest.mark.asyncio
    async def test_the_message_is_empty_for_a_valid_template(self):
        fake_llm = FakeLLMClient(response=f'```json\n{_VALID_JSON}\n```')
        result = await _make_use_case(fake_llm).generate('go', 'm', [])
        assert result.template is not None
        assert not result.assistant_message

    @pytest.mark.asyncio
    async def test_the_edit_history_records_a_short_note(self):
        base = SessionTemplate.model_validate(_VALID_TEMPLATE_DICT)
        fake_llm = FakeLLMClient(response=f'```json\n{_VALID_JSON}\n```')
        history: list[ChatMessage] = []
        await _make_edit_use_case(fake_llm, base).generate('yes', 'm', history)
        assert history[-1].content == 'Template updated.'


class TestBuilderResult:
    def test_defaults(self):
        result = BuilderResult()
        assert result.template is None
        assert not result.validation_errors
        assert not result.assistant_message
        assert not result.error
