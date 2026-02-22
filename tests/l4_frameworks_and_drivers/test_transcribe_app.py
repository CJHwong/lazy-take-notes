"""Tests for TranscribeApp — file transcription TUI."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from lazy_take_notes.l1_entities.transcript import TranscriptSegment
from lazy_take_notes.l3_interface_adapters.controllers.session_controller import SessionController
from lazy_take_notes.l3_interface_adapters.gateways.yaml_template_loader import YamlTemplateLoader
from lazy_take_notes.l4_frameworks_and_drivers.apps.transcribe import TranscribeApp
from lazy_take_notes.l4_frameworks_and_drivers.infra_config import build_app_config
from lazy_take_notes.l4_frameworks_and_drivers.messages import (
    AudioWorkerStatus,
    ModelDownloadProgress,
    TranscriptChunk,
)
from lazy_take_notes.l4_frameworks_and_drivers.widgets.status_bar import StatusBar
from tests.conftest import FakeLLMClient, FakePersistence


def make_app(tmp_path: Path) -> TranscribeApp:
    config = build_app_config({})
    template = YamlTemplateLoader().load('default_zh_tw')
    output_dir = tmp_path / 'output'
    output_dir.mkdir()
    fake_llm = FakeLLMClient()
    fake_persist = FakePersistence(output_dir)
    controller = SessionController(
        config=config,
        template=template,
        llm_client=fake_llm,
        persistence=fake_persist,
    )
    audio_file = tmp_path / 'audio.wav'
    audio_file.touch()
    return TranscribeApp(
        config=config,
        template=template,
        output_dir=output_dir,
        controller=controller,
        audio_path=audio_file,
    )


class TestTranscribeAppComposition:
    @pytest.mark.asyncio
    async def test_has_required_widgets(self, tmp_path):
        app = make_app(tmp_path)
        with patch.object(app, '_start_file_worker'):
            async with app.run_test():
                assert app.query_one('#transcript-panel')
                assert app.query_one('#digest-panel')
                assert app.query_one('#status-bar', StatusBar)


class TestTranscribeAppStatus:
    @pytest.mark.asyncio
    async def test_recording_status_updates_bar(self, tmp_path):
        app = make_app(tmp_path)
        with patch.object(app, '_start_file_worker'):
            async with app.run_test() as pilot:
                app.post_message(AudioWorkerStatus(status='recording'))
                await pilot.pause()

                bar = app.query_one('#status-bar', StatusBar)
                assert bar.recording is True

    @pytest.mark.asyncio
    async def test_stopped_triggers_final_digest(self, tmp_path):
        app = make_app(tmp_path)
        with patch.object(app, '_start_file_worker'):
            async with app.run_test() as pilot:
                # Seed buffer
                segments = [TranscriptSegment(text='Line', wall_start=0.0, wall_end=1.0)]
                app.post_message(TranscriptChunk(segments=segments))
                await pilot.pause()

                with patch.object(app, '_run_final_digest') as mock_final:
                    app.post_message(AudioWorkerStatus(status='stopped'))
                    await pilot.pause()
                    mock_final.assert_called_once()

    @pytest.mark.asyncio
    async def test_stopped_no_content_no_digest(self, tmp_path):
        app = make_app(tmp_path)
        with patch.object(app, '_start_file_worker'):
            async with app.run_test() as pilot:
                with patch.object(app, '_run_final_digest') as mock_final:
                    app.post_message(AudioWorkerStatus(status='stopped'))
                    await pilot.pause()
                    mock_final.assert_not_called()

    @pytest.mark.asyncio
    async def test_error_status(self, tmp_path):
        app = make_app(tmp_path)
        with patch.object(app, '_start_file_worker'):
            async with app.run_test() as pilot:
                app.post_message(AudioWorkerStatus(status='error', error='bad file'))
                await pilot.pause()
                assert app._worker_done is True


class TestTranscribeAppActions:
    @pytest.mark.asyncio
    async def test_stop_sets_shutdown_event(self, tmp_path):
        app = make_app(tmp_path)
        with patch.object(app, '_start_file_worker'):
            async with app.run_test() as pilot:
                await pilot.press('s')
                await pilot.pause()
                assert app._file_shutdown.is_set()

    @pytest.mark.asyncio
    async def test_stop_after_done_is_noop(self, tmp_path):
        app = make_app(tmp_path)
        with patch.object(app, '_start_file_worker'):
            async with app.run_test() as pilot:
                app._worker_done = True
                app._file_shutdown.clear()
                await pilot.press('s')
                await pilot.pause()
                assert not app._file_shutdown.is_set()

    @pytest.mark.asyncio
    async def test_quit_when_worker_running_sets_pending(self, tmp_path):
        app = make_app(tmp_path)
        with patch.object(app, '_start_file_worker'):
            async with app.run_test() as pilot:
                app._worker_done = False
                await pilot.press('q')
                await pilot.pause()
                assert app._pending_quit is True
                assert app._file_shutdown.is_set()

    @pytest.mark.asyncio
    async def test_quit_after_done_no_content_exits(self, tmp_path):
        app = make_app(tmp_path)
        with patch.object(app, '_start_file_worker'):
            async with app.run_test() as pilot:
                app._worker_done = True
                with patch.object(app, 'exit') as mock_exit:
                    await pilot.press('q')
                    await pilot.pause()
                    mock_exit.assert_called_once()

    @pytest.mark.asyncio
    async def test_quit_after_done_with_content_runs_final_digest(self, tmp_path):
        app = make_app(tmp_path)
        with patch.object(app, '_start_file_worker'):
            async with app.run_test() as pilot:
                segments = [TranscriptSegment(text='Data', wall_start=0.0, wall_end=1.0)]
                app.post_message(TranscriptChunk(segments=segments))
                await pilot.pause()

                app._worker_done = True
                with patch.object(app, '_run_final_digest') as mock_final:
                    await pilot.press('q')
                    await pilot.pause()
                    mock_final.assert_called_once()
                    assert app._pending_quit is True

    @pytest.mark.asyncio
    async def test_quit_blocked_while_digest_running(self, tmp_path):
        app = make_app(tmp_path)
        with patch.object(app, '_start_file_worker'):
            async with app.run_test() as pilot:
                app._digest_running = True
                with patch.object(app, 'exit') as mock_exit:
                    await pilot.press('q')
                    await pilot.pause()
                    mock_exit.assert_not_called()

    @pytest.mark.asyncio
    async def test_force_digest_noop_when_pending_quit(self, tmp_path):
        app = make_app(tmp_path)
        with patch.object(app, '_start_file_worker'):
            async with app.run_test() as pilot:
                segments = [TranscriptSegment(text='Data', wall_start=0.0, wall_end=1.0)]
                app.post_message(TranscriptChunk(segments=segments))
                await pilot.pause()

                app._pending_quit = True
                with patch.object(app, '_run_digest_worker') as mock_digest:
                    await pilot.press('d')
                    await pilot.pause()
                    mock_digest.assert_not_called()

    @pytest.mark.asyncio
    async def test_stopped_with_pending_quit_and_no_content_exits(self, tmp_path):
        app = make_app(tmp_path)
        with patch.object(app, '_start_file_worker'):
            async with app.run_test() as pilot:
                app._pending_quit = True
                with patch.object(app, 'exit') as mock_exit:
                    app.post_message(AudioWorkerStatus(status='stopped'))
                    await pilot.pause()
                    mock_exit.assert_called_once()


class TestTranscribeAppHints:
    @pytest.mark.asyncio
    async def test_hints_when_transcribing(self, tmp_path):
        app = make_app(tmp_path)
        with patch.object(app, '_start_file_worker'):
            async with app.run_test() as pilot:
                app.post_message(AudioWorkerStatus(status='recording'))
                await pilot.pause()
                bar = app.query_one('#status-bar', StatusBar)
                assert 'stop' in bar.keybinding_hints

    @pytest.mark.asyncio
    async def test_hints_when_stopped(self, tmp_path):
        app = make_app(tmp_path)
        with patch.object(app, '_start_file_worker'):
            async with app.run_test() as pilot:
                app.post_message(AudioWorkerStatus(status='stopped'))
                await pilot.pause()
                bar = app.query_one('#status-bar', StatusBar)
                assert 'quit' in bar.keybinding_hints

    @pytest.mark.asyncio
    async def test_hints_when_loading_model(self, tmp_path):
        app = make_app(tmp_path)
        with patch.object(app, '_start_file_worker'):
            async with app.run_test() as pilot:
                app.post_message(AudioWorkerStatus(status='loading_model'))
                await pilot.pause()
                bar = app.query_one('#status-bar', StatusBar)
                # Falls back to idle hints (base default)
                assert 'help' in bar.keybinding_hints

    @pytest.mark.asyncio
    async def test_hints_when_model_ready(self, tmp_path):
        app = make_app(tmp_path)
        with patch.object(app, '_start_file_worker'):
            async with app.run_test() as pilot:
                app.post_message(AudioWorkerStatus(status='model_ready'))
                await pilot.pause()
                bar = app.query_one('#status-bar', StatusBar)
                assert 'help' in bar.keybinding_hints

    @pytest.mark.asyncio
    async def test_help_keybindings_includes_stop(self, tmp_path):
        app = make_app(tmp_path)
        with patch.object(app, '_start_file_worker'):
            async with app.run_test() as pilot:
                await pilot.press('h')
                await pilot.pause()
                assert app.screen.__class__.__name__ == 'HelpModal'


class TestTranscribeAppDownloadModal:
    @pytest.mark.asyncio
    async def test_loading_model_switches_download_modal(self, tmp_path):
        app = make_app(tmp_path)
        with patch.object(app, '_start_file_worker'):
            async with app.run_test() as pilot:
                # Trigger download modal via ModelDownloadProgress
                app.post_message(ModelDownloadProgress(percent=50, model_name='test-model'))
                await pilot.pause()
                assert app._download_modal is not None

                # loading_model should call switch_to_loading on the modal
                with patch.object(app._download_modal, 'switch_to_loading') as mock_switch:
                    app.post_message(AudioWorkerStatus(status='loading_model'))
                    await pilot.pause()
                    mock_switch.assert_called_once()


class TestTranscribeAppForceDigest:
    @pytest.mark.asyncio
    async def test_force_digest_calls_super(self, tmp_path):
        app = make_app(tmp_path)
        with patch.object(app, '_start_file_worker'):
            async with app.run_test() as pilot:
                segments = [
                    TranscriptSegment(text='Line', wall_start=0.0, wall_end=1.0),
                ]
                app.post_message(TranscriptChunk(segments=segments))
                await pilot.pause()

                app._pending_quit = False
                with patch.object(app, '_run_digest_worker') as mock_digest:
                    await pilot.press('d')
                    await pilot.pause()
                    mock_digest.assert_called_once()
