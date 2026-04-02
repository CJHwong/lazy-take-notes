"""Tests for WelcomePicker — startup TUI to select Record / Transcribe / View."""

from __future__ import annotations

import pytest

from lazy_take_notes.l4_frameworks_and_drivers.pickers.welcome_picker import (
    ModeItem,
    WelcomePicker,
)


class TestWelcomePicker:
    @pytest.mark.asyncio
    async def test_shows_all_modes(self):
        picker = WelcomePicker()
        async with picker.run_test() as pilot:
            await pilot.pause()
            items = picker.query(ModeItem)
            assert len(items) == 5
            modes = [item.mode for item in items]
            assert modes == ['record', 'transcribe', 'view', 'create-template', 'config']

    @pytest.mark.asyncio
    async def test_escape_returns_none(self):
        picker = WelcomePicker()
        async with picker.run_test() as pilot:
            await pilot.press('escape')
            await pilot.pause()
        assert picker.return_value is None

    @pytest.mark.asyncio
    async def test_q_returns_none(self):
        picker = WelcomePicker()
        async with picker.run_test() as pilot:
            await pilot.press('q')
            await pilot.pause()
        assert picker.return_value is None

    @pytest.mark.asyncio
    async def test_enter_returns_record(self):
        picker = WelcomePicker()
        async with picker.run_test() as pilot:
            await pilot.pause()
            await pilot.press('enter')
            await pilot.pause()
        assert picker.return_value == 'record'

    @pytest.mark.asyncio
    async def test_down_enter_returns_transcribe(self):
        picker = WelcomePicker()
        async with picker.run_test() as pilot:
            await pilot.pause()
            await pilot.press('down')
            await pilot.pause()
            await pilot.press('enter')
            await pilot.pause()
        assert picker.return_value == 'transcribe'

    @pytest.mark.asyncio
    async def test_down_down_enter_returns_view(self):
        picker = WelcomePicker()
        async with picker.run_test() as pilot:
            await pilot.pause()
            await pilot.press('down')
            await pilot.press('down')
            await pilot.pause()
            await pilot.press('enter')
            await pilot.pause()
        assert picker.return_value == 'view'

    @pytest.mark.asyncio
    async def test_theme_applied_on_mount(self):
        from unittest.mock import patch

        with patch(
            'lazy_take_notes.l4_frameworks_and_drivers.pickers.welcome_picker.load_theme', return_value='textual-light'
        ):
            picker = WelcomePicker()
            async with picker.run_test() as pilot:
                await pilot.pause()
                assert picker.theme == 'textual-light'
