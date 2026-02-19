"""Tests for RunQuickActionUseCase — uses FakeLLMClient."""

import pytest

from lazy_take_notes.l1_entities.transcript import TranscriptSegment
from lazy_take_notes.l2_use_cases.quick_action_use_case import RunQuickActionUseCase
from lazy_take_notes.l3_interface_adapters.gateways.yaml_template_loader import YamlTemplateLoader
from tests.conftest import FakeLLMClient


class TestRunQuickAction:
    @pytest.mark.asyncio
    async def test_existing_key_returns_result(self):
        template = YamlTemplateLoader().load('default_zh_tw')
        fake_llm = FakeLLMClient(response='Action result')
        uc = RunQuickActionUseCase(fake_llm)

        first_key = template.quick_actions[0].key
        result = await uc.execute(
            key=first_key,
            template=template,
            model='test-model',
            latest_digest='Some digest',
            all_segments=[],
        )

        assert result is not None
        text, label = result
        assert text == 'Action result'
        assert label == template.quick_actions[0].label

    @pytest.mark.asyncio
    async def test_unknown_key_returns_none(self):
        template = YamlTemplateLoader().load('default_zh_tw')
        fake_llm = FakeLLMClient()
        uc = RunQuickActionUseCase(fake_llm)

        result = await uc.execute(
            key='nonexistent',
            template=template,
            model='test-model',
            latest_digest=None,
            all_segments=[],
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_uses_recent_segments(self):
        template = YamlTemplateLoader().load('default_zh_tw')
        fake_llm = FakeLLMClient(response='OK')
        uc = RunQuickActionUseCase(fake_llm)

        segments = [TranscriptSegment(text=f'Seg {i}', wall_start=float(i), wall_end=float(i + 1)) for i in range(60)]

        first_key = template.quick_actions[0].key
        await uc.execute(
            key=first_key,
            template=template,
            model='test-model',
            latest_digest='digest',
            all_segments=segments,
        )

        # Should have called chat_single with the last 50 segments
        assert len(fake_llm.chat_single_calls) == 1
        prompt = fake_llm.chat_single_calls[0][1]
        assert 'Seg 59' in prompt
        assert 'Seg 10' in prompt

    @pytest.mark.asyncio
    async def test_user_context_included_in_prompt(self):
        template = YamlTemplateLoader().load('default_zh_tw')
        fake_llm = FakeLLMClient(response='OK')
        uc = RunQuickActionUseCase(fake_llm)

        first_key = template.quick_actions[0].key
        await uc.execute(
            key=first_key,
            template=template,
            model='test-model',
            latest_digest='digest',
            all_segments=[],
            user_context='Speaker A = Alice',
        )

        assert len(fake_llm.chat_single_calls) == 1
        prompt = fake_llm.chat_single_calls[0][1]
        assert 'Speaker A = Alice' in prompt
