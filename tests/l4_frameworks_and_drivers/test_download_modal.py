"""Tests for the download modal widget."""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Static

from lazy_take_notes.l4_frameworks_and_drivers.widgets.download_modal import DownloadModal


class ModalHost(App[None]):
    """Minimal app to host the download modal for testing."""

    def compose(self) -> ComposeResult:
        yield Static('host')


class TestDownloadModal:
    @pytest.mark.asyncio
    async def test_initial_state(self):
        modal = DownloadModal(model_name='large-v3-turbo-q8_0')
        assert modal.model_name == 'large-v3-turbo-q8_0'
        assert modal.phase == 'downloading'
        assert modal.percent == -1

    @pytest.mark.asyncio
    async def test_composes_with_status_and_detail(self):
        app = ModalHost()
        async with app.run_test() as pilot:
            modal = DownloadModal(model_name='large-v3-turbo-q8_0')
            app.push_screen(modal)
            await pilot.pause()
            # Widgets are mounted
            assert modal.query_one('#dl-status', Static) is not None
            assert modal.query_one('#dl-detail', Static) is not None

    @pytest.mark.asyncio
    async def test_update_progress_tracks_percent(self):
        app = ModalHost()
        async with app.run_test() as pilot:
            modal = DownloadModal(model_name='breeze-q8')
            app.push_screen(modal)
            await pilot.pause()
            modal.update_progress(42)
            assert modal.percent == 42
            assert modal.phase == 'downloading'

    @pytest.mark.asyncio
    async def test_switch_to_loading(self):
        app = ModalHost()
        async with app.run_test() as pilot:
            modal = DownloadModal(model_name='large-v3-turbo-q8_0')
            app.push_screen(modal)
            await pilot.pause()
            modal.switch_to_loading()
            assert modal.phase == 'loading'

    @pytest.mark.asyncio
    async def test_dismiss(self):
        app = ModalHost()
        async with app.run_test() as pilot:
            modal = DownloadModal(model_name='test')
            app.push_screen(modal)
            await pilot.pause()
            modal.dismiss()
            await pilot.pause()
            assert not isinstance(app.screen, DownloadModal)
