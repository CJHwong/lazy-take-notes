"""Tests for the recording consent notice widget."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Static

from lazy_take_notes.l4_frameworks_and_drivers.widgets.consent_notice import ConsentNotice


class ModalHost(App[None]):
    """Minimal app to host the consent notice for testing."""

    def compose(self) -> ComposeResult:
        yield Static('host')


class TestConsentNotice:
    @pytest.mark.asyncio
    async def test_composes_with_title_body_and_hint(self):
        app = ModalHost()
        async with app.run_test() as pilot:
            modal = ConsentNotice()
            app.push_screen(modal)
            await pilot.pause()

            assert modal.query_one('#notice-title', Static) is not None
            assert modal.query_one('#notice-hint', Static) is not None

    @pytest.mark.asyncio
    async def test_dismiss_on_escape(self):
        app = ModalHost()
        async with app.run_test() as pilot:
            app.push_screen(ConsentNotice())
            await pilot.pause()
            assert isinstance(app.screen, ConsentNotice)

            await pilot.press('escape')
            await pilot.pause()
            assert not isinstance(app.screen, ConsentNotice)

    @pytest.mark.asyncio
    async def test_dismiss_on_any_key(self):
        app = ModalHost()
        async with app.run_test() as pilot:
            app.push_screen(ConsentNotice())
            await pilot.pause()
            assert isinstance(app.screen, ConsentNotice)

            await pilot.press('enter')
            await pilot.pause()
            assert not isinstance(app.screen, ConsentNotice)

    @pytest.mark.asyncio
    async def test_dismiss_on_space(self):
        app = ModalHost()
        async with app.run_test() as pilot:
            app.push_screen(ConsentNotice())
            await pilot.pause()
            assert isinstance(app.screen, ConsentNotice)

            await pilot.press('space')
            await pilot.pause()
            assert not isinstance(app.screen, ConsentNotice)

    @pytest.mark.asyncio
    async def test_body_contains_consent_language(self):
        app = ModalHost()
        async with app.run_test() as pilot:
            modal = ConsentNotice()
            app.push_screen(modal)
            await pilot.pause()

            from lazy_take_notes.l4_frameworks_and_drivers.widgets.consent_notice import NOTICE_BODY

            assert 'record audio' in NOTICE_BODY
            assert 'consent' in NOTICE_BODY.lower()
            assert 'participants' in NOTICE_BODY.lower()

    @pytest.mark.asyncio
    async def test_n_key_calls_on_suppress_and_dismisses(self):
        app = ModalHost()
        async with app.run_test() as pilot:
            callback = MagicMock()
            app.push_screen(ConsentNotice(on_suppress=callback))
            await pilot.pause()
            assert isinstance(app.screen, ConsentNotice)

            await pilot.press('n')
            await pilot.pause()
            callback.assert_called_once()
            assert not isinstance(app.screen, ConsentNotice)

    @pytest.mark.asyncio
    async def test_other_key_does_not_call_on_suppress(self):
        app = ModalHost()
        async with app.run_test() as pilot:
            callback = MagicMock()
            app.push_screen(ConsentNotice(on_suppress=callback))
            await pilot.pause()
            assert isinstance(app.screen, ConsentNotice)

            await pilot.press('enter')
            await pilot.pause()
            callback.assert_not_called()
            assert not isinstance(app.screen, ConsentNotice)

    @pytest.mark.asyncio
    async def test_n_key_without_callback_dismisses_safely(self):
        """Pressing 'n' when no on_suppress callback is set should still dismiss."""
        app = ModalHost()
        async with app.run_test() as pilot:
            app.push_screen(ConsentNotice())
            await pilot.pause()
            assert isinstance(app.screen, ConsentNotice)

            await pilot.press('n')
            await pilot.pause()
            assert not isinstance(app.screen, ConsentNotice)
