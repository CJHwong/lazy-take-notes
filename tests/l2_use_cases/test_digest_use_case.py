"""Tests for RunDigestUseCase — uses FakeLLMClient, NOT @patch("ollama...")."""

from __future__ import annotations

import pytest

from lazy_take_notes.l1_entities.digest_state import DigestState
from lazy_take_notes.l2_use_cases.digest_use_case import RunDigestUseCase
from lazy_take_notes.l3_interface_adapters.gateways.yaml_template_loader import YamlTemplateLoader
from tests.conftest import FakeLLMClient

VALID_DIGEST_RESPONSE = """\
## 目前議題
討論 REST vs GraphQL API 設計方案。

## 已結案議題
### 開場介紹
**結論：** 完成
- 大家自我介紹

## 待釐清問題
- 要選哪個框架？

## 建議行動
- **進行效能測試**：需要數據來做決定

## 停車場
（無）
"""


@pytest.fixture
def digest_state() -> DigestState:
    tmpl = YamlTemplateLoader().load('default_zh_tw')
    state = DigestState()
    state.init_messages(tmpl.system_prompt)
    state.buffer = [f'Line {i}' for i in range(20)]
    state.all_lines = state.buffer.copy()
    return state


@pytest.fixture
def template():
    return YamlTemplateLoader().load('default_zh_tw')


class TestRunDigest:
    @pytest.mark.asyncio
    async def test_success(self, digest_state, template):
        fake_llm = FakeLLMClient(response=VALID_DIGEST_RESPONSE, prompt_tokens=500)
        uc = RunDigestUseCase(fake_llm)

        result = await uc.execute(state=digest_state, model='test-model', template=template)

        assert result.ok
        assert result.data is not None
        assert '目前議題' in result.data
        assert digest_state.digest_count == 1
        assert digest_state.consecutive_failures == 0
        assert len(digest_state.buffer) == 0

    @pytest.mark.asyncio
    async def test_empty_response(self, digest_state, template):
        fake_llm = FakeLLMClient(response='   ')
        uc = RunDigestUseCase(fake_llm)

        result = await uc.execute(state=digest_state, model='test-model', template=template)

        assert not result.ok
        assert 'Empty response' in result.error
        assert digest_state.consecutive_failures == 1
        assert len(digest_state.buffer) == 20  # Buffer preserved

    @pytest.mark.asyncio
    async def test_llm_exception(self, digest_state, template):
        fake_llm = FakeLLMClient()

        # Make the chat method raise
        async def _raise(*a, **kw):
            raise ConnectionError('LLM down')

        fake_llm.chat = _raise  # type: ignore[invalid-assignment]
        uc = RunDigestUseCase(fake_llm)

        result = await uc.execute(state=digest_state, model='test-model', template=template)

        assert not result.ok
        assert 'LLM error' in result.error
        assert digest_state.consecutive_failures == 1

    @pytest.mark.asyncio
    async def test_final_digest(self, digest_state, template):
        fake_llm = FakeLLMClient(response=VALID_DIGEST_RESPONSE)
        uc = RunDigestUseCase(fake_llm)

        result = await uc.execute(
            state=digest_state,
            model='test-model',
            template=template,
            is_final=True,
            full_transcript='Full transcript text',
        )

        assert result.ok
        # Verify the prompt contained the full transcript
        _, messages = fake_llm.chat_calls[0]
        user_msg = messages[-1].content
        assert 'Full transcript text' in user_msg


class TestDigestFaults:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'bad_template',
        ['{unknown_var}', 'stray { brace', '{line_count:%%}'],
        ids=['unknown-variable', 'unclosed-brace', 'bad-format-spec'],
    )
    async def test_a_malformed_template_returns_an_error_instead_of_raising(self, digest_state, template, bad_template):
        broken = template.model_copy(update={'digest_user_template': bad_template})
        fake_llm = FakeLLMClient(response=VALID_DIGEST_RESPONSE)
        messages_before = list(digest_state.messages)

        result = await RunDigestUseCase(fake_llm).execute(state=digest_state, model='m', template=broken)

        assert not result.ok
        assert 'Template error' in result.error
        assert digest_state.messages == messages_before
        assert fake_llm.chat_calls == []

    @pytest.mark.asyncio
    async def test_a_cancelled_digest_removes_its_prompt_from_the_history(self, digest_state, template):
        import asyncio

        fake_llm = FakeLLMClient()

        async def _hang(*a, **kw):
            await asyncio.Event().wait()

        fake_llm.chat = _hang  # type: ignore[invalid-assignment]
        messages_before = list(digest_state.messages)
        task = asyncio.create_task(RunDigestUseCase(fake_llm).execute(state=digest_state, model='m', template=template))
        await asyncio.sleep(0)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task
        assert digest_state.messages == messages_before


class TestDigestPromptOwnership:
    """Each digest removes its own prompt, even after a newer digest appended one."""

    @pytest.mark.asyncio
    async def test_a_cancelled_digest_keeps_a_newer_prompt(self, digest_state, template):
        import asyncio

        from lazy_take_notes.l1_entities.chat_message import ChatMessage

        fake_llm = FakeLLMClient()

        async def _hang(*a, **kw):
            await asyncio.Event().wait()

        fake_llm.chat = _hang  # type: ignore[invalid-assignment]
        before = list(digest_state.messages)
        task = asyncio.create_task(RunDigestUseCase(fake_llm).execute(state=digest_state, model='m', template=template))
        await asyncio.sleep(0)
        newer = ChatMessage(role='user', content='final prompt')
        digest_state.messages.append(newer)  # the replacing final digest got here first
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task
        assert digest_state.messages == [*before, newer]
        assert digest_state.messages[-1] is newer

    @pytest.mark.asyncio
    @pytest.mark.parametrize('outcome', ['raises', 'empty'])
    async def test_a_failed_digest_keeps_a_newer_prompt(self, digest_state, template, outcome):
        import asyncio

        from lazy_take_notes.l1_entities.chat_message import ChatMessage
        from lazy_take_notes.l2_use_cases.ports.llm_client import ChatResponse

        newer = ChatMessage(role='user', content='final prompt')
        release = asyncio.Event()
        fake_llm = FakeLLMClient()

        async def _late_failure(*a, **kw):
            await release.wait()
            if outcome == 'raises':
                raise ConnectionError('dropped')
            return ChatResponse(content='   ')

        fake_llm.chat = _late_failure  # type: ignore[invalid-assignment]
        before = list(digest_state.messages)
        task = asyncio.create_task(RunDigestUseCase(fake_llm).execute(state=digest_state, model='m', template=template))
        await asyncio.sleep(0)
        digest_state.messages.append(newer)
        release.set()

        result = await task
        assert not result.ok
        assert digest_state.messages == [*before, newer]
