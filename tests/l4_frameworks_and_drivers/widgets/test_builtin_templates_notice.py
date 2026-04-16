"""Tests for the built-in templates notice modal widget."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Static

from lazy_take_notes.l4_frameworks_and_drivers.widgets.builtin_templates_notice import (
    NOTICE_BODY,
    BuiltinTemplatesNotice,
)


class ModalHost(App[None]):
    """Minimal app to host the notice for testing."""

    def compose(self) -> ComposeResult:
        yield Static('host')


class TestBuiltinTemplatesNotice:
    @pytest.mark.asyncio
    async def test_composes_with_title_body_and_hint(self):
        app = ModalHost()
        async with app.run_test() as pilot:
            modal = BuiltinTemplatesNotice()
            app.push_screen(modal)
            await pilot.pause()

            assert modal.query_one('#btn-title', Static) is not None
            assert modal.query_one('#btn-hint', Static) is not None

    @pytest.mark.asyncio
    async def test_dismiss_on_escape(self):
        app = ModalHost()
        async with app.run_test() as pilot:
            app.push_screen(BuiltinTemplatesNotice())
            await pilot.pause()
            assert isinstance(app.screen, BuiltinTemplatesNotice)

            await pilot.press('escape')
            await pilot.pause()
            assert not isinstance(app.screen, BuiltinTemplatesNotice)

    @pytest.mark.asyncio
    async def test_dismiss_on_any_key(self):
        app = ModalHost()
        async with app.run_test() as pilot:
            app.push_screen(BuiltinTemplatesNotice())
            await pilot.pause()
            assert isinstance(app.screen, BuiltinTemplatesNotice)

            await pilot.press('enter')
            await pilot.pause()
            assert not isinstance(app.screen, BuiltinTemplatesNotice)

    @pytest.mark.asyncio
    async def test_body_mentions_templates_and_settings(self):
        assert 'example sessions' in NOTICE_BODY.lower()
        assert 'show_builtin_templates' in NOTICE_BODY
        assert '[n]' in NOTICE_BODY

    @pytest.mark.asyncio
    async def test_d_key_calls_on_suppress_and_dismisses(self):
        app = ModalHost()
        async with app.run_test() as pilot:
            callback = MagicMock()
            app.push_screen(BuiltinTemplatesNotice(on_suppress=callback))
            await pilot.pause()
            assert isinstance(app.screen, BuiltinTemplatesNotice)

            await pilot.press('d')
            await pilot.pause()
            callback.assert_called_once()
            assert not isinstance(app.screen, BuiltinTemplatesNotice)

    @pytest.mark.asyncio
    async def test_other_key_does_not_call_on_suppress(self):
        app = ModalHost()
        async with app.run_test() as pilot:
            callback = MagicMock()
            app.push_screen(BuiltinTemplatesNotice(on_suppress=callback))
            await pilot.pause()
            assert isinstance(app.screen, BuiltinTemplatesNotice)

            await pilot.press('enter')
            await pilot.pause()
            callback.assert_not_called()
            assert not isinstance(app.screen, BuiltinTemplatesNotice)

    @pytest.mark.asyncio
    async def test_d_key_without_callback_dismisses_safely(self):
        """Pressing 'd' when no on_suppress callback is set should still dismiss."""
        app = ModalHost()
        async with app.run_test() as pilot:
            app.push_screen(BuiltinTemplatesNotice())
            await pilot.pause()
            assert isinstance(app.screen, BuiltinTemplatesNotice)

            await pilot.press('d')
            await pilot.pause()
            assert not isinstance(app.screen, BuiltinTemplatesNotice)
