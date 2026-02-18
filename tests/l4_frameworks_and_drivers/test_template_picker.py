"""Tests for the template picker TUI."""

from __future__ import annotations

from pathlib import Path

import pytest
from textual.widgets import Input, Markdown

from lazy_take_notes.l3_interface_adapters.gateways.yaml_template_loader import all_template_names
from lazy_take_notes.l4_frameworks_and_drivers.template_picker import (
    TemplatePicker,
)


@pytest.fixture(autouse=True)
def _isolate_user_templates(monkeypatch):
    """Ensure user templates dir does not exist so only built-ins show."""
    import lazy_take_notes.l3_interface_adapters.gateways.yaml_template_loader as mod

    monkeypatch.setattr(mod, 'USER_TEMPLATES_DIR', Path('/nonexistent/user/templates'))


class TestTemplatePicker:
    @pytest.mark.asyncio
    async def test_picker_shows_all_templates(self):
        picker = TemplatePicker()
        async with picker.run_test():
            items = picker.query('#template-list TemplateItem')
            assert len(items) == len(all_template_names())

    @pytest.mark.asyncio
    async def test_escape_returns_none(self):
        picker = TemplatePicker()
        async with picker.run_test() as pilot:
            await pilot.press('escape')
            await pilot.pause()

        assert picker.return_value is None

    @pytest.mark.asyncio
    async def test_enter_returns_template_name(self):
        picker = TemplatePicker()
        async with picker.run_test() as pilot:
            # Move focus to the list and select
            await pilot.press('tab')
            await pilot.pause()
            await pilot.press('down')
            await pilot.pause()
            await pilot.press('enter')
            await pilot.pause()

        assert picker.return_value in all_template_names()

    @pytest.mark.asyncio
    async def test_preview_updates_on_highlight(self):
        picker = TemplatePicker()
        async with picker.run_test() as pilot:
            preview = picker.query_one('#preview-md', Markdown)
            await pilot.press('tab')
            await pilot.pause()
            await pilot.press('down')
            await pilot.pause()
            assert preview._markdown != ''

    @pytest.mark.asyncio
    async def test_locale_headers_present(self):
        picker = TemplatePicker()
        async with picker.run_test():
            headers = picker.query('#template-list LocaleHeader')
            assert len(headers) > 0

    @pytest.mark.asyncio
    async def test_search_filters_templates(self):
        picker = TemplatePicker()
        async with picker.run_test() as pilot:
            total = len(picker.query('#template-list TemplateItem'))
            search = picker.query_one('#search-input', Input)
            search.focus()
            await pilot.pause()
            # 'en' should match only the English template(s)
            await pilot.press(*'en')
            await pilot.pause()
            filtered = len(picker.query('#template-list TemplateItem'))
            assert filtered < total
            assert filtered > 0

    @pytest.mark.asyncio
    async def test_search_no_match_shows_empty(self):
        picker = TemplatePicker()
        async with picker.run_test() as pilot:
            search = picker.query_one('#search-input', Input)
            search.focus()
            await pilot.pause()
            for ch in 'xyznonexistent':
                await pilot.press(ch)
            await pilot.pause()
            items = picker.query('#template-list TemplateItem')
            assert len(items) == 0

    @pytest.mark.asyncio
    async def test_search_clear_restores_all(self):
        picker = TemplatePicker()
        async with picker.run_test() as pilot:
            total = len(picker.query('#template-list TemplateItem'))
            search = picker.query_one('#search-input', Input)
            search.focus()
            await pilot.pause()
            # Filter then clear
            await pilot.press(*'en')
            await pilot.pause()
            search.value = ''
            await pilot.pause()
            restored = len(picker.query('#template-list TemplateItem'))
            assert restored == total

    @pytest.mark.asyncio
    async def test_enter_after_search_returns_filtered_name(self):
        picker = TemplatePicker()
        async with picker.run_test() as pilot:
            search = picker.query_one('#search-input', Input)
            search.focus()
            await pilot.pause()
            for ch in 'default_en':
                await pilot.press(ch)
            await pilot.pause()
            await pilot.press('enter')
            await pilot.pause()

        assert picker.return_value == 'default_en'
