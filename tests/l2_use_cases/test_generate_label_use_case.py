"""Tests for GenerateLabelUseCase."""

from __future__ import annotations

import pytest

from lazy_take_notes.l2_use_cases.generate_label_use_case import GenerateLabelUseCase, sanitize_label
from tests.conftest import FakeLLMClient


class TestSanitizeLabel:
    def test_clean_input(self):
        assert sanitize_label('sprint_review') == 'sprint_review'

    def test_strips_quotes_and_whitespace(self):
        assert sanitize_label('  "Sprint Review"  ') == 'sprint_review'

    def test_replaces_special_chars(self):
        assert sanitize_label('Q1 Budget & Planning!') == 'q1_budget_planning'

    def test_truncates_to_max_words(self):
        assert sanitize_label('one_two_three_four_five_six_seven') == 'one_two_three_four_five'

    def test_empty_input(self):
        assert not sanitize_label('')

    def test_only_special_chars(self):
        assert not sanitize_label('!!!')

    def test_single_quoted_word(self):
        assert sanitize_label("'standup'") == 'standup'

    def test_hyphens_converted_to_underscores(self):
        assert sanitize_label('sprint-review-notes') == 'sprint_review_notes'


class TestGenerateLabelUseCase:
    @pytest.mark.asyncio
    async def test_happy_path(self):
        fake_llm = FakeLLMClient(response='sprint_review_notes')
        uc = GenerateLabelUseCase(fake_llm)

        result = await uc.execute('Default', 'General session', '## Sprint Review\nGood progress.', 'llama3:8b')

        assert result == 'sprint_review_notes'
        assert len(fake_llm.chat_single_calls) == 1
        _, prompt = fake_llm.chat_single_calls[0]
        assert 'Default' in prompt
        assert 'General session' in prompt
        assert 'Sprint Review' in prompt

    @pytest.mark.asyncio
    async def test_messy_response_sanitized(self):
        fake_llm = FakeLLMClient(response='"  Q1 Budget & Planning Session!  "')
        uc = GenerateLabelUseCase(fake_llm)

        result = await uc.execute('Default', '', 'digest text', 'model')

        assert result == 'q1_budget_planning_session'

    @pytest.mark.asyncio
    async def test_empty_response(self):
        fake_llm = FakeLLMClient(response='')
        uc = GenerateLabelUseCase(fake_llm)

        result = await uc.execute('Default', '', 'digest text', 'model')

        assert not result
