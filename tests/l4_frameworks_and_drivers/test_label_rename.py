"""Tests for runtime session label rename feature."""

from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import patch

import pytest

from lazy_take_notes.l3_interface_adapters.controllers.session_controller import SessionController
from lazy_take_notes.l3_interface_adapters.gateways.file_persistence import FilePersistenceGateway
from lazy_take_notes.l3_interface_adapters.gateways.yaml_template_loader import YamlTemplateLoader
from lazy_take_notes.l4_frameworks_and_drivers.apps.record import RecordApp
from lazy_take_notes.l4_frameworks_and_drivers.infra_config import build_app_config
from lazy_take_notes.l4_frameworks_and_drivers.widgets.label_modal import LabelModal
from tests.conftest import FakeLLMClient


def _make_app(tmp_path: Path, *, label: str = '', dir_name: str = '2026-02-21_143052') -> RecordApp:
    config = build_app_config({})
    template = YamlTemplateLoader().load('default_zh_tw')
    output_dir = tmp_path / dir_name
    output_dir.mkdir(parents=True)
    fake_llm = FakeLLMClient()
    persistence = FilePersistenceGateway(output_dir)
    controller = SessionController(
        config=config,
        template=template,
        llm_client=fake_llm,
        persistence=persistence,
    )
    return RecordApp(
        config=config,
        template=template,
        output_dir=output_dir,
        controller=controller,
        label=label,
    )


class TestBuildHeaderText:
    @pytest.mark.asyncio
    async def test_header_without_label(self, tmp_path):
        app = _make_app(tmp_path)
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test():
                header = app._build_header_text()
                assert 'lazy-take-notes' in header
                assert '\u2014' not in header

    @pytest.mark.asyncio
    async def test_header_with_label(self, tmp_path):
        app = _make_app(tmp_path, label='sprint-review')
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test():
                header = app._build_header_text()
                assert '\u2014 sprint-review' in header

    @pytest.mark.asyncio
    async def test_header_label_shown_at_startup(self, tmp_path):
        app = _make_app(tmp_path, label='kickoff')
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as pilot:
                await pilot.pause()
                from textual.widgets import Static

                header_widget = app.query_one('#header', Static)
                assert 'kickoff' in str(header_widget.render())


class TestSanitization:
    def test_spaces_become_underscores(self):
        assert re.sub(r'[^\w\-]', '_', 'sprint review') == 'sprint_review'

    def test_special_chars_stripped(self):
        assert re.sub(r'[^\w\-]', '_', 'a/b:c!d') == 'a_b_c_d'

    def test_hyphens_preserved(self):
        assert re.sub(r'[^\w\-]', '_', 'sprint-review') == 'sprint-review'

    def test_unicode_word_chars_preserved(self):
        result = re.sub(r'[^\w\-]', '_', '會議記錄')
        assert result == '會議記錄'


class TestTimestampPrefixPreservation:
    @pytest.mark.asyncio
    async def test_rename_preserves_timestamp(self, tmp_path):
        app = _make_app(tmp_path, dir_name='2026-02-21_143052')
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as pilot:
                await pilot.pause()
                app._on_label_result('standup')

                assert app._output_dir.name == '2026-02-21_143052_standup'

    @pytest.mark.asyncio
    async def test_rename_replaces_old_label(self, tmp_path):
        app = _make_app(tmp_path, dir_name='2026-02-21_143052_old_label', label='old_label')
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as pilot:
                await pilot.pause()
                app._on_label_result('new-label')

                assert app._output_dir.name == '2026-02-21_143052_new-label'


class TestRenameCallback:
    @pytest.mark.asyncio
    async def test_none_label_is_noop(self, tmp_path):
        app = _make_app(tmp_path)
        original_dir = app._output_dir
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as pilot:
                await pilot.pause()
                app._on_label_result(None)
                assert app._output_dir == original_dir

    @pytest.mark.asyncio
    async def test_empty_label_is_noop(self, tmp_path):
        app = _make_app(tmp_path)
        original_dir = app._output_dir
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as pilot:
                await pilot.pause()
                app._on_label_result('')
                assert app._output_dir == original_dir

    @pytest.mark.asyncio
    async def test_whitespace_only_is_noop(self, tmp_path):
        app = _make_app(tmp_path)
        original_dir = app._output_dir
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as pilot:
                await pilot.pause()
                app._on_label_result('   ')
                assert app._output_dir == original_dir

    @pytest.mark.asyncio
    async def test_rename_updates_output_dir(self, tmp_path):
        app = _make_app(tmp_path)
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as pilot:
                await pilot.pause()
                app._on_label_result('retro')

                assert app._output_dir.name == '2026-02-21_143052_retro'
                assert app._output_dir.exists()

    @pytest.mark.asyncio
    async def test_rename_updates_persistence(self, tmp_path):
        app = _make_app(tmp_path)
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as pilot:
                await pilot.pause()
                app._on_label_result('retro')

                persistence = app._controller._persistence
                assert isinstance(persistence, FilePersistenceGateway)
                assert persistence.output_dir == app._output_dir

    @pytest.mark.asyncio
    async def test_rename_updates_header(self, tmp_path):
        app = _make_app(tmp_path)
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as pilot:
                await pilot.pause()
                app._on_label_result('retro')

                from textual.widgets import Static

                header = app.query_one('#header', Static)
                assert 'retro' in str(header.render())

    @pytest.mark.asyncio
    async def test_persistence_writes_to_new_dir_after_rename(self, tmp_path):
        app = _make_app(tmp_path)
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as pilot:
                await pilot.pause()
                app._on_label_result('retro')

                persistence = app._controller._persistence
                path = persistence.save_digest_md('After rename', 1)
                assert 'retro' in str(path.parent)
                assert path.exists()


class TestLabelModal:
    @pytest.mark.asyncio
    async def test_r_opens_label_modal(self, tmp_path):
        app = _make_app(tmp_path)
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as pilot:
                await pilot.press('l')
                await pilot.pause()
                assert isinstance(app.screen, LabelModal)

    @pytest.mark.asyncio
    async def test_enter_submits_text(self, tmp_path):
        app = _make_app(tmp_path)
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as pilot:
                await pilot.press('l')
                await pilot.pause()
                assert isinstance(app.screen, LabelModal)

                modal_input = app.screen.query_one('#label-input')
                modal_input.value = 'my-session'
                await pilot.press('enter')
                await pilot.pause()
                assert not isinstance(app.screen, LabelModal)
                assert 'my-session' in app._output_dir.name

    @pytest.mark.asyncio
    async def test_enter_empty_returns_none(self, tmp_path):
        app = _make_app(tmp_path)
        original_dir = app._output_dir
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as pilot:
                await pilot.press('l')
                await pilot.pause()
                assert isinstance(app.screen, LabelModal)

                modal_input = app.screen.query_one('#label-input')
                modal_input.value = ''
                await pilot.press('enter')
                await pilot.pause()
                assert not isinstance(app.screen, LabelModal)
                # Empty text → None → no rename
                assert app._output_dir == original_dir

    @pytest.mark.asyncio
    async def test_escape_dismisses_without_rename(self, tmp_path):
        app = _make_app(tmp_path)
        original_dir = app._output_dir
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as pilot:
                await pilot.press('l')
                await pilot.pause()
                assert isinstance(app.screen, LabelModal)

                await pilot.press('escape')
                await pilot.pause()
                assert not isinstance(app.screen, LabelModal)
                assert app._output_dir == original_dir
