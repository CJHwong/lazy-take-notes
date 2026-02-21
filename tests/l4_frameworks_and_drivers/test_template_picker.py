"""Tests for the template picker TUI."""

from __future__ import annotations

from pathlib import Path

import pytest
from textual.widgets import Input, Markdown

import lazy_take_notes.l3_interface_adapters.gateways.yaml_template_loader as yaml_loader_mod
from lazy_take_notes.l1_entities.audio_mode import AudioMode
from lazy_take_notes.l3_interface_adapters.gateways.yaml_template_loader import all_template_names
from lazy_take_notes.l4_frameworks_and_drivers.template_picker import (
    TemplatePicker,
)


@pytest.fixture(autouse=True)
def _isolate_user_templates(monkeypatch):
    """Ensure user templates dir does not exist so only built-ins show."""
    monkeypatch.setattr(yaml_loader_mod, 'USER_TEMPLATES_DIR', Path('/nonexistent/user/templates'))


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

        assert picker.return_value is not None
        name, mode = picker.return_value
        assert name in all_template_names()
        assert isinstance(mode, AudioMode)

    @pytest.mark.asyncio
    async def test_enter_returns_audio_mode_in_result(self):
        picker = TemplatePicker()
        async with picker.run_test() as pilot:
            await pilot.press('enter')
            await pilot.pause()

        assert picker.return_value is not None
        _name, mode = picker.return_value
        assert mode == AudioMode.MIC_ONLY

    @pytest.mark.asyncio
    async def test_d_cycles_audio_mode(self):
        picker = TemplatePicker()
        async with picker.run_test() as pilot:
            # [d] only fires when the list has focus (not the search Input)
            await pilot.press('tab')
            await pilot.pause()
            assert picker._audio_mode == AudioMode.MIC_ONLY
            await pilot.press('d')
            await pilot.pause()
            assert picker._audio_mode == AudioMode.SYSTEM_ONLY
            await pilot.press('d')
            await pilot.pause()
            assert picker._audio_mode == AudioMode.MIX
            await pilot.press('d')
            await pilot.pause()
            assert picker._audio_mode == AudioMode.MIC_ONLY

    @pytest.mark.asyncio
    async def test_d_cycles_and_result_reflects_mode(self):
        picker = TemplatePicker()
        async with picker.run_test() as pilot:
            await pilot.press('tab')  # move focus to list so [d] fires
            await pilot.pause()
            await pilot.press('d')
            await pilot.pause()
            await pilot.press('enter')
            await pilot.pause()

        assert picker.return_value is not None
        _name, mode = picker.return_value
        assert mode == AudioMode.SYSTEM_ONLY

    @pytest.mark.asyncio
    async def test_d_no_op_when_input_focused(self):
        """[d] must not cycle audio mode when the search Input has focus (to allow typing)."""
        picker = TemplatePicker()
        async with picker.run_test() as pilot:
            # Input is focused from on_mount — pressing 'd' types into the search box
            assert isinstance(picker.focused, Input)
            await pilot.press('d')
            await pilot.pause()
            assert picker._audio_mode == AudioMode.MIC_ONLY  # unchanged

    @pytest.mark.asyncio
    async def test_d_hidden_when_show_audio_mode_false(self):
        picker = TemplatePicker(show_audio_mode=False)
        async with picker.run_test() as pilot:
            # Tab to list so any key handling fires — still no-op when audio mode hidden
            await pilot.press('tab')
            await pilot.pause()
            before = picker._audio_mode
            await pilot.press('d')
            await pilot.pause()
            assert picker._audio_mode == before

    @pytest.mark.asyncio
    async def test_preview_updates_on_highlight(self):
        picker = TemplatePicker()
        async with picker.run_test() as pilot:
            preview = picker.query_one('#preview-md', Markdown)
            await pilot.press('tab')
            await pilot.pause()
            await pilot.press('down')
            await pilot.pause()
            assert preview._markdown

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

        assert picker.return_value is not None
        name, _mode = picker.return_value
        assert name == 'default_en'
