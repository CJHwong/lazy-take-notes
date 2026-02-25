"""Tests for the query modal widget."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Markdown, Static

from lazy_take_notes.l4_frameworks_and_drivers.widgets.query_modal import QueryModal


class ModalHost(App[None]):
    """Minimal app to host the query modal for testing."""

    def compose(self) -> ComposeResult:
        yield Static('host')


class TestQueryModal:
    @pytest.mark.asyncio
    async def test_compose_renders_title_body_hint(self):
        app = ModalHost()
        async with app.run_test() as pilot:
            modal = QueryModal(title='Blockers', body='**No blockers**')
            app.push_screen(modal)
            await pilot.pause()
            assert modal.query_one('#query-title', Static).content == 'Blockers'
            assert modal.query_one('#query-body', Markdown) is not None
            hint = modal.query_one('#query-hint', Static)
            assert 'Copy' in str(hint.content)

    @pytest.mark.asyncio
    async def test_error_class_applied(self):
        app = ModalHost()
        async with app.run_test() as pilot:
            modal = QueryModal(title='Error', body='oops', is_error=True)
            app.push_screen(modal)
            await pilot.pause()
            assert modal.has_class('error')

    @pytest.mark.asyncio
    async def test_escape_dismisses(self):
        app = ModalHost()
        async with app.run_test() as pilot:
            modal = QueryModal(title='T', body='B')
            app.push_screen(modal)
            await pilot.pause()
            await pilot.press('escape')
            await pilot.pause()
            assert not isinstance(app.screen, QueryModal)

    @pytest.mark.asyncio
    async def test_copy_body_copies_to_clipboard(self):
        app = ModalHost()
        async with app.run_test() as pilot:
            modal = QueryModal(title='Status', body='## All done\n\nNothing left.')
            app.push_screen(modal)
            await pilot.pause()
            with patch('lazy_take_notes.l4_frameworks_and_drivers.widgets.query_modal.pyperclip.copy') as mock_copy:
                await pilot.press('c')
                await pilot.pause()
                mock_copy.assert_called_once_with('## All done\n\nNothing left.')
