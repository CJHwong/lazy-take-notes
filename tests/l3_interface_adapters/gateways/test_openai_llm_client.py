"""Tests for OpenAI-compatible LLM client gateway — mocks openai here (L3 boundary)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lazy_take_notes.l1_entities.chat_message import ChatMessage
from lazy_take_notes.l3_interface_adapters.gateways.openai_llm_client import OpenAICompatLLMClient


def _make_chat_response(content='Test response', prompt_tokens=42):
    """Build a mock ChatCompletion response."""
    choice = MagicMock()
    choice.message.content = content
    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = usage
    return resp


class TestOpenAICompatLLMClient:
    @pytest.mark.asyncio
    @patch('lazy_take_notes.l3_interface_adapters.gateways.openai_llm_client.openai.AsyncOpenAI')
    async def test_chat_success(self, mock_cls):
        mock_client = AsyncMock()
        mock_cls.return_value = mock_client
        mock_client.chat.completions.create = AsyncMock(return_value=_make_chat_response())

        client = OpenAICompatLLMClient()
        result = await client.chat('gpt-4o', [ChatMessage(role='user', content='hi')])

        assert result.content == 'Test response'
        assert result.prompt_tokens == 42

    @pytest.mark.asyncio
    @patch('lazy_take_notes.l3_interface_adapters.gateways.openai_llm_client.openai.AsyncOpenAI')
    async def test_chat_single_success(self, mock_cls):
        mock_client = AsyncMock()
        mock_cls.return_value = mock_client
        mock_client.chat.completions.create = AsyncMock(
            return_value=_make_chat_response(content='Summary result'),
        )

        client = OpenAICompatLLMClient()
        result = await client.chat_single('gpt-4o-mini', 'Summarize')

        assert result == 'Summary result'

    @pytest.mark.asyncio
    @patch('lazy_take_notes.l3_interface_adapters.gateways.openai_llm_client.openai.AsyncOpenAI')
    async def test_chat_handles_none_content(self, mock_cls):
        mock_client = AsyncMock()
        mock_cls.return_value = mock_client
        mock_client.chat.completions.create = AsyncMock(return_value=_make_chat_response(content=None))

        client = OpenAICompatLLMClient()
        result = await client.chat('gpt-4o', [ChatMessage(role='user', content='hi')])

        assert not result.content

    @pytest.mark.asyncio
    @patch('lazy_take_notes.l3_interface_adapters.gateways.openai_llm_client.openai.AsyncOpenAI')
    async def test_chat_handles_no_usage(self, mock_cls):
        mock_client = AsyncMock()
        mock_cls.return_value = mock_client
        resp = _make_chat_response()
        resp.usage = None
        mock_client.chat.completions.create = AsyncMock(return_value=resp)

        client = OpenAICompatLLMClient()
        result = await client.chat('gpt-4o', [ChatMessage(role='user', content='hi')])

        assert result.prompt_tokens == 0

    @patch('lazy_take_notes.l3_interface_adapters.gateways.openai_llm_client.openai.OpenAI')
    def test_check_connectivity_success(self, mock_cls):
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.models.list.return_value = MagicMock()

        client = OpenAICompatLLMClient()
        ok, err = client.check_connectivity()
        assert ok is True
        assert not err

    @patch('lazy_take_notes.l3_interface_adapters.gateways.openai_llm_client.openai.OpenAI')
    def test_check_connectivity_auth_failure(self, mock_cls):
        import openai as openai_lib

        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.models.list.side_effect = openai_lib.AuthenticationError(
            message='Invalid API key',
            response=MagicMock(status_code=401),
            body=None,
        )

        client = OpenAICompatLLMClient()
        ok, err = client.check_connectivity()
        assert ok is False
        assert 'Authentication failed' in err

    @patch('lazy_take_notes.l3_interface_adapters.gateways.openai_llm_client.openai.OpenAI')
    def test_check_connectivity_network_failure(self, mock_cls):
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.models.list.side_effect = ConnectionError('Connection refused')

        client = OpenAICompatLLMClient()
        ok, err = client.check_connectivity()
        assert ok is False
        assert 'Cannot connect' in err

    @patch('lazy_take_notes.l3_interface_adapters.gateways.openai_llm_client.openai.OpenAI')
    def test_check_models_all_present(self, mock_cls):
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.models.retrieve.return_value = MagicMock()

        client = OpenAICompatLLMClient()
        assert client.check_models(['gpt-4o', 'gpt-4o-mini']) == []

    @patch('lazy_take_notes.l3_interface_adapters.gateways.openai_llm_client.openai.OpenAI')
    def test_check_models_some_missing(self, mock_cls):
        import openai as openai_lib

        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        def _retrieve(model):
            if model == 'gpt-4oo-typo':
                raise openai_lib.NotFoundError(
                    message='model not found',
                    response=MagicMock(status_code=404),
                    body=None,
                )
            return MagicMock()

        mock_client.models.retrieve.side_effect = _retrieve

        client = OpenAICompatLLMClient()
        assert client.check_models(['gpt-4o', 'gpt-4oo-typo']) == ['gpt-4oo-typo']

    @patch('lazy_take_notes.l3_interface_adapters.gateways.openai_llm_client.openai.OpenAI')
    def test_check_models_endpoint_unsupported_returns_empty(self, mock_cls):
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.models.retrieve.side_effect = ConnectionError('endpoint not supported')

        client = OpenAICompatLLMClient()
        assert client.check_models(['gpt-4o']) == []

    @patch('lazy_take_notes.l3_interface_adapters.gateways.openai_llm_client.openai.OpenAI')
    def test_custom_base_url(self, mock_cls):
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.models.list.return_value = MagicMock()

        client = OpenAICompatLLMClient(api_key='sk-test', base_url='https://api.groq.com/openai/v1')
        client.check_connectivity()

        mock_cls.assert_called_once_with(api_key='sk-test', base_url='https://api.groq.com/openai/v1')
