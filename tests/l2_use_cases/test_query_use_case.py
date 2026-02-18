"""Tests for RunQueryUseCase — uses FakeLLMClient."""

import pytest

from lazy_take_notes.l2_use_cases.query_use_case import RunQueryUseCase
from tests.conftest import FakeLLMClient


class TestRunQuery:
    @pytest.mark.asyncio
    async def test_success(self):
        fake_llm = FakeLLMClient(response="Here's your summary...")
        uc = RunQueryUseCase(fake_llm)

        result = await uc.execute('Summarize', model='test-model')
        assert result == "Here's your summary..."

    @pytest.mark.asyncio
    async def test_error(self):
        fake_llm = FakeLLMClient()

        async def _raise(*a, **kw):
            raise ConnectionError('down')

        fake_llm.chat_single = _raise  # type: ignore[invalid-assignment]
        uc = RunQueryUseCase(fake_llm)

        result = await uc.execute('Summarize', model='test-model')
        assert 'Error' in result
