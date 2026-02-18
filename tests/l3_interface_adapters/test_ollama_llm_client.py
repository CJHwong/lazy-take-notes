"""Tests for Ollama LLM client gateway — mocks ollama here (L3 boundary)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lazy_take_notes.l1_entities.chat_message import ChatMessage
from lazy_take_notes.l3_interface_adapters.gateways.ollama_llm_client import OllamaLLMClient


class TestOllamaLLMClient:
    @pytest.mark.asyncio
    @patch('lazy_take_notes.l3_interface_adapters.gateways.ollama_llm_client.ollama_sync.AsyncClient')
    async def test_chat_success(self, mock_client_cls):
        mock_client = AsyncMock()
        mock_client_cls.return_value = mock_client

        mock_resp = MagicMock()
        mock_resp.message.content = 'Test response'
        mock_resp.prompt_eval_count = 42
        mock_client.chat.return_value = mock_resp

        client = OllamaLLMClient()
        result = await client.chat('model', [ChatMessage(role='user', content='hi')])

        assert result.content == 'Test response'
        assert result.prompt_tokens == 42

    @pytest.mark.asyncio
    @patch('lazy_take_notes.l3_interface_adapters.gateways.ollama_llm_client.ollama_sync.AsyncClient')
    async def test_chat_single_success(self, mock_client_cls):
        mock_client = AsyncMock()
        mock_client_cls.return_value = mock_client

        mock_resp = MagicMock()
        mock_resp.message.content = 'Summary result'
        mock_client.chat.return_value = mock_resp

        client = OllamaLLMClient()
        result = await client.chat_single('model', 'Summarize')

        assert result == 'Summary result'

    @patch('lazy_take_notes.l3_interface_adapters.gateways.ollama_llm_client.ollama_sync.Client')
    def test_check_connectivity_success(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.list.return_value = MagicMock()

        client = OllamaLLMClient()
        ok, err = client.check_connectivity()
        assert ok is True
        assert err == ''

    @patch('lazy_take_notes.l3_interface_adapters.gateways.ollama_llm_client.ollama_sync.Client')
    def test_check_connectivity_failure(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.list.side_effect = ConnectionError('nope')

        client = OllamaLLMClient()
        ok, err = client.check_connectivity()
        assert ok is False
        assert 'Cannot connect' in err

    @patch('lazy_take_notes.l3_interface_adapters.gateways.ollama_llm_client.ollama_sync.Client')
    def test_check_models_all_present(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.show.return_value = MagicMock()

        client = OllamaLLMClient()
        assert client.check_models(['llama3.2', 'qwen2.5']) == []

    @patch('lazy_take_notes.l3_interface_adapters.gateways.ollama_llm_client.ollama_sync.Client')
    def test_check_models_some_missing(self, mock_client_cls):
        import ollama as ollama_lib

        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        def _show(model):
            if model == 'missing-model':
                raise ollama_lib.ResponseError('model not found')
            return MagicMock()

        mock_client.show.side_effect = _show

        client = OllamaLLMClient()
        assert client.check_models(['llama3.2', 'missing-model']) == ['missing-model']

    @patch('lazy_take_notes.l3_interface_adapters.gateways.ollama_llm_client.ollama_sync.Client')
    def test_check_models_connectivity_failure_returns_empty(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.show.side_effect = ConnectionError('cannot connect')

        client = OllamaLLMClient()
        assert client.check_models(['llama3.2']) == []
