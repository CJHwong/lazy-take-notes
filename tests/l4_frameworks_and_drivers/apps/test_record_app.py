"""Tests for the Textual TUI app using headless Pilot."""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from textual.widgets import TextArea

from lazy_take_notes.l1_entities.transcript import TranscriptSegment
from lazy_take_notes.l3_interface_adapters.controllers.session_controller import SessionController
from lazy_take_notes.l3_interface_adapters.gateways.yaml_template_loader import YamlTemplateLoader
from lazy_take_notes.l4_frameworks_and_drivers.apps.record import RecordApp
from lazy_take_notes.l4_frameworks_and_drivers.config import build_app_config
from lazy_take_notes.l4_frameworks_and_drivers.messages import (
    AudioLevel,
    AudioWorkerStatus,
    DigestError,
    DigestReady,
    ModelDownloadProgress,
    QueryResult,
    TranscriptChunk,
)
from lazy_take_notes.l4_frameworks_and_drivers.widgets.digest_panel import DigestPanel
from lazy_take_notes.l4_frameworks_and_drivers.widgets.download_modal import DownloadModal
from lazy_take_notes.l4_frameworks_and_drivers.widgets.query_modal import QueryModal
from lazy_take_notes.l4_frameworks_and_drivers.widgets.status_bar import StatusBar
from lazy_take_notes.l4_frameworks_and_drivers.widgets.transcript_panel import TranscriptPanel
from tests.conftest import FakeLLMClient, FakePersistence


def make_app(
    tmp_path: Path,
    missing_digest_models: list[str] | None = None,
    missing_interactive_models: list[str] | None = None,
) -> RecordApp:
    config = build_app_config({})
    template = YamlTemplateLoader().load('default_en')
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
    return RecordApp(
        config=config,
        template=template,
        output_dir=output_dir,
        controller=controller,
        missing_digest_models=missing_digest_models,
        missing_interactive_models=missing_interactive_models,
    )


class TestAppComposition:
    @pytest.mark.asyncio
    async def test_app_has_required_widgets(self, tmp_path):
        app = make_app(tmp_path)
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test():
                assert app.query_one('#transcript-panel', TranscriptPanel)
                assert app.query_one('#digest-panel', DigestPanel)
                assert app.query_one('#status-bar', StatusBar)


class TestTranscriptChunkHandling:
    @pytest.mark.asyncio
    async def test_transcript_chunk_updates_panel(self, tmp_path):
        app = make_app(tmp_path)
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as pilot:
                segments = [
                    TranscriptSegment(text='Hello world', wall_start=1.0, wall_end=2.0),
                    TranscriptSegment(text='Testing', wall_start=2.0, wall_end=3.0),
                ]
                app.post_message(TranscriptChunk(segments=segments))
                await pilot.pause()

                bar = app.query_one('#status-bar', StatusBar)
                assert bar.buf_count == 2


class TestAudioWorkerStatusHandling:
    @pytest.mark.asyncio
    async def test_recording_status(self, tmp_path):
        app = make_app(tmp_path)
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as pilot:
                app.post_message(AudioWorkerStatus(status='recording'))
                await pilot.pause()

                bar = app.query_one('#status-bar', StatusBar)
                assert bar.recording is True
                assert bar.paused is False
                assert bar.stopped is False


class TestDigestReadyHandling:
    @pytest.mark.asyncio
    async def test_digest_updates_panel(self, tmp_path):
        app = make_app(tmp_path)
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as pilot:
                markdown = '## Current Topic\nTesting the app\n'
                app.post_message(DigestReady(markdown=markdown, digest_number=1))
                await pilot.pause()

                bar = app.query_one('#status-bar', StatusBar)
                assert not bar.activity


class TestPauseResume:
    @pytest.mark.asyncio
    async def test_pause_sets_event_and_updates_bar(self, tmp_path):
        app = make_app(tmp_path)
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as pilot:
                app.post_message(AudioWorkerStatus(status='recording'))
                await pilot.pause()

                bar = app.query_one('#status-bar', StatusBar)
                assert bar.recording is True
                assert not app._audio_paused.is_set()

                await pilot.press('space')
                await pilot.pause()

                assert app._audio_paused.is_set()
                assert bar.paused is True
                assert bar.recording is False

    @pytest.mark.asyncio
    async def test_resume_clears_event(self, tmp_path):
        app = make_app(tmp_path)
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as pilot:
                app.post_message(AudioWorkerStatus(status='recording'))
                await pilot.pause()

                await pilot.press('space')
                await pilot.pause()
                assert app._audio_paused.is_set()

                await pilot.press('space')
                await pilot.pause()
                assert not app._audio_paused.is_set()

                bar = app.query_one('#status-bar', StatusBar)
                assert bar.paused is False
                assert bar.recording is True


class TestStopRecording:
    @pytest.mark.asyncio
    async def test_stop_sets_stopped_state(self, tmp_path):
        app = make_app(tmp_path)
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as pilot:
                app.post_message(AudioWorkerStatus(status='recording'))
                await pilot.pause()

                await pilot.press('s')
                await pilot.pause()

                bar = app.query_one('#status-bar', StatusBar)
                assert bar.stopped is True
                assert bar.recording is False
                assert app._audio_stopped is True

    @pytest.mark.asyncio
    async def test_pause_after_stop_is_noop(self, tmp_path):
        app = make_app(tmp_path)
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as pilot:
                app.post_message(AudioWorkerStatus(status='recording'))
                await pilot.pause()

                await pilot.press('s')
                await pilot.pause()

                bar = app.query_one('#status-bar', StatusBar)
                assert bar.stopped is True

                await pilot.press('space')
                await pilot.pause()
                assert bar.stopped is True
                assert bar.paused is False

    @pytest.mark.asyncio
    async def test_stop_is_idempotent(self, tmp_path):
        app = make_app(tmp_path)
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as pilot:
                await pilot.press('s')
                await pilot.pause()
                await pilot.press('s')
                await pilot.pause()

                assert app._audio_stopped is True


class TestHelpModal:
    @pytest.mark.asyncio
    async def test_h_opens_help_modal(self, tmp_path):
        app = make_app(tmp_path)
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as pilot:
                await pilot.press('h')
                await pilot.pause()

                assert app.screen.__class__.__name__ == 'HelpModal'

    @pytest.mark.asyncio
    async def test_h_toggles_help_closed(self, tmp_path):
        app = make_app(tmp_path)
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as pilot:
                await pilot.press('h')
                await pilot.pause()
                assert app.screen.__class__.__name__ == 'HelpModal'

                await pilot.press('h')
                await pilot.pause()
                assert app.screen.__class__.__name__ != 'HelpModal'

    @pytest.mark.asyncio
    async def test_escape_dismisses_help(self, tmp_path):
        app = make_app(tmp_path)
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as pilot:
                await pilot.press('h')
                await pilot.pause()
                assert app.screen.__class__.__name__ == 'HelpModal'

                await pilot.press('escape')
                await pilot.pause()
                assert app.screen.__class__.__name__ != 'HelpModal'


class TestCopyContent:
    @pytest.mark.asyncio
    async def test_copy_digest_content(self, tmp_path):
        app = make_app(tmp_path)
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as pilot:
                markdown = '## Topic\nTest content\n'
                panel = app.query_one('#digest-panel', DigestPanel)
                panel.update_digest(markdown)
                await pilot.pause()

                panel.focus()
                await pilot.pause()

                with patch('lazy_take_notes.l4_frameworks_and_drivers.widgets.digest_panel.pyperclip') as mock_clip:
                    await pilot.press('c')
                    await pilot.pause()
                    mock_clip.copy.assert_called_once_with(markdown)

    @pytest.mark.asyncio
    async def test_copy_transcript_content(self, tmp_path):
        app = make_app(tmp_path)
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as pilot:
                segments = [
                    TranscriptSegment(text='Line one', wall_start=1.0, wall_end=2.0),
                    TranscriptSegment(text='Line two', wall_start=2.0, wall_end=3.0),
                ]
                panel = app.query_one('#transcript-panel', TranscriptPanel)
                panel.append_segments(segments)
                await pilot.pause()

                panel.focus()
                await pilot.pause()

                with patch('lazy_take_notes.l4_frameworks_and_drivers.widgets.transcript_panel.pyperclip') as mock_clip:
                    await pilot.press('c')
                    await pilot.pause()
                    mock_clip.copy.assert_called_once()
                    copied = mock_clip.copy.call_args[0][0]
                    assert 'Line one' in copied
                    assert 'Line two' in copied

    @pytest.mark.asyncio
    async def test_copy_empty_digest_warns(self, tmp_path):
        app = make_app(tmp_path)
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as pilot:
                panel = app.query_one('#digest-panel', DigestPanel)
                panel.focus()
                await pilot.pause()

                with patch('lazy_take_notes.l4_frameworks_and_drivers.widgets.digest_panel.pyperclip') as mock_clip:
                    await pilot.press('c')
                    await pilot.pause()
                    mock_clip.copy.assert_not_called()


class TestStopFlush:
    @pytest.mark.asyncio
    async def test_stop_does_not_trigger_digest_immediately(self, tmp_path):
        app = make_app(tmp_path)
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as pilot:
                segments = [
                    TranscriptSegment(text='Buffered', wall_start=0.0, wall_end=1.0),
                ]
                app.post_message(TranscriptChunk(segments=segments))
                await pilot.pause()
                assert len(app._controller.digest_state.buffer) > 0

                with patch.object(app, '_run_digest_worker') as mock_digest:
                    await pilot.press('s')
                    await pilot.pause()
                    mock_digest.assert_not_called()

    @pytest.mark.asyncio
    async def test_audio_stopped_status_triggers_digest(self, tmp_path):
        app = make_app(tmp_path)
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as pilot:
                segments = [
                    TranscriptSegment(text='Buffered line', wall_start=0.0, wall_end=1.0),
                ]
                app.post_message(TranscriptChunk(segments=segments))
                await pilot.pause()

                app._audio_stopped = True
                app._cancel_audio_workers()

                with patch.object(app, '_run_digest_worker') as mock_digest:
                    app.post_message(AudioWorkerStatus(status='stopped'))
                    await pilot.pause()
                    mock_digest.assert_called_once_with(is_final=True)


class TestTimerFreeze:
    @pytest.mark.asyncio
    async def test_timer_freezes_on_stop(self, tmp_path):
        app = make_app(tmp_path)
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as pilot:
                bar = app.query_one('#status-bar', StatusBar)
                assert bar._frozen_elapsed is None

                await pilot.press('s')
                await pilot.pause()

                assert bar._frozen_elapsed is not None

    @pytest.mark.asyncio
    async def test_frozen_timer_does_not_change(self, tmp_path):
        app = make_app(tmp_path)
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as pilot:
                bar = app.query_one('#status-bar', StatusBar)

                await pilot.press('s')
                await pilot.pause()

                frozen_val = bar._frozen_elapsed
                time.sleep(0.05)
                now = time.monotonic()
                assert bar._format_elapsed(now) == bar._format_elapsed(now)
                assert bar._frozen_elapsed == frozen_val


class TestMissingModels:
    @pytest.mark.asyncio
    async def test_digest_panel_shows_warning_when_digest_model_missing(self, tmp_path):
        app = make_app(tmp_path, missing_digest_models=['llama3.2'])
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as pilot:
                await pilot.pause()
                panel = app.query_one('#digest-panel', DigestPanel)
                assert 'llama3.2' in panel._current_markdown
                assert 'ollama pull' in panel._current_markdown

    @pytest.mark.asyncio
    async def test_digest_panel_empty_when_no_missing_models(self, tmp_path):
        app = make_app(tmp_path)
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as pilot:
                await pilot.pause()
                panel = app.query_one('#digest-panel', DigestPanel)
                assert not panel._current_markdown

    @pytest.mark.asyncio
    async def test_digest_panel_empty_when_only_interactive_model_missing(self, tmp_path):
        app = make_app(tmp_path, missing_interactive_models=['qwen2.5:0.5b'])
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as pilot:
                await pilot.pause()
                panel = app.query_one('#digest-panel', DigestPanel)
                assert not panel._current_markdown

    @pytest.mark.asyncio
    async def test_digest_panel_shows_all_missing_digest_models(self, tmp_path):
        app = make_app(tmp_path, missing_digest_models=['llama3.2', 'qwen2.5:0.5b'])
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as pilot:
                await pilot.pause()
                panel = app.query_one('#digest-panel', DigestPanel)
                assert 'llama3.2' in panel._current_markdown
                assert 'qwen2.5:0.5b' in panel._current_markdown


class TestStatusBarHints:
    @pytest.mark.asyncio
    async def test_hints_on_mount(self, tmp_path):
        app = make_app(tmp_path)
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as pilot:
                await pilot.pause()
                bar = app.query_one('#status-bar', StatusBar)
                assert 'help' in bar.keybinding_hints
                assert 'quit' in bar.keybinding_hints

    @pytest.mark.asyncio
    async def test_hints_when_recording(self, tmp_path):
        app = make_app(tmp_path)
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as pilot:
                app.post_message(AudioWorkerStatus(status='recording'))
                await pilot.pause()
                bar = app.query_one('#status-bar', StatusBar)
                assert 'pause' in bar.keybinding_hints
                assert 'stop' in bar.keybinding_hints

    @pytest.mark.asyncio
    async def test_hints_when_paused(self, tmp_path):
        app = make_app(tmp_path)
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as pilot:
                app.post_message(AudioWorkerStatus(status='recording'))
                await pilot.pause()
                await pilot.press('space')
                await pilot.pause()
                bar = app.query_one('#status-bar', StatusBar)
                assert 'resume' in bar.keybinding_hints

    @pytest.mark.asyncio
    async def test_hints_when_stopped(self, tmp_path):
        app = make_app(tmp_path)
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as pilot:
                app.post_message(AudioWorkerStatus(status='recording'))
                await pilot.pause()
                await pilot.press('s')
                await pilot.pause()
                bar = app.query_one('#status-bar', StatusBar)
                assert 'quit' in bar.keybinding_hints


class TestQuitWithFinalDigest:
    @pytest.mark.asyncio
    async def test_quit_with_data_sets_pending_quit(self, tmp_path):
        app = make_app(tmp_path)
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as pilot:
                segments = [
                    TranscriptSegment(text='Some data', wall_start=0.0, wall_end=1.0),
                ]
                app.post_message(TranscriptChunk(segments=segments))
                await pilot.pause()

                with patch.object(app, '_run_final_digest'):
                    await pilot.press('q')
                    await pilot.pause()

                    assert app._pending_quit is True
                    assert app._audio_stopped is True

    @pytest.mark.asyncio
    async def test_quit_no_data_exits_after_audio_stops(self, tmp_path):
        app = make_app(tmp_path)
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as pilot:
                with patch.object(app, 'exit') as mock_exit:
                    await pilot.press('q')
                    await pilot.pause()
                    # 'q' defers exit until the audio worker confirms its flush is done
                    assert app._pending_quit is True
                    mock_exit.assert_not_called()

                    app.post_message(AudioWorkerStatus(status='stopped'))
                    await pilot.pause()
                    mock_exit.assert_called_once()

    @pytest.mark.asyncio
    async def test_second_q_exits(self, tmp_path):
        app = make_app(tmp_path)
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as pilot:
                app._pending_quit = True

                with patch.object(app, 'exit') as mock_exit:
                    await pilot.press('q')
                    await pilot.pause()
                    mock_exit.assert_called_once()


class TestForceDigest:
    @pytest.mark.asyncio
    async def test_force_digest_triggers_worker_when_buffer_not_empty(self, tmp_path):
        app = make_app(tmp_path)
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as pilot:
                segments = [TranscriptSegment(text='Line one', wall_start=0.0, wall_end=1.0)]
                app.post_message(TranscriptChunk(segments=segments))
                await pilot.pause()

                with patch.object(app, '_run_digest_worker') as mock_digest:
                    await pilot.press('d')
                    await pilot.pause()
                    mock_digest.assert_called_once_with(is_final=False)

    @pytest.mark.asyncio
    async def test_force_digest_notifies_when_buffer_empty(self, tmp_path):
        app = make_app(tmp_path)
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as pilot:
                with patch.object(app, '_run_digest_worker') as mock_digest:
                    await pilot.press('d')
                    await pilot.pause()
                    mock_digest.assert_not_called()

    @pytest.mark.asyncio
    async def test_force_digest_is_noop_when_digest_already_running(self, tmp_path):
        app = make_app(tmp_path)
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as pilot:
                segments = [TranscriptSegment(text='Line one', wall_start=0.0, wall_end=1.0)]
                app.post_message(TranscriptChunk(segments=segments))
                await pilot.pause()

                app._digest_running = True
                with patch.object(app, '_run_digest_worker') as mock_digest:
                    await pilot.press('d')
                    await pilot.pause()
                    mock_digest.assert_not_called()


class TestStatusBarLastDigestTime:
    @pytest.mark.asyncio
    async def test_last_digest_time_set_on_digest_ready(self, tmp_path):
        app = make_app(tmp_path)
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as pilot:
                bar = app.query_one('#status-bar', StatusBar)
                assert bar.last_digest_time == 0.0

                app.post_message(DigestReady(markdown='## Topic\n', digest_number=1))
                await pilot.pause()

                assert bar.last_digest_time > 0.0


class TestQueryErrorModal:
    @pytest.mark.asyncio
    async def test_error_result_opens_modal_with_is_error(self, tmp_path):
        app = make_app(tmp_path)
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as pilot:
                app.post_message(
                    QueryResult(result='Error: connection refused', action_label='Catch Up', is_error=True)
                )
                await pilot.pause()

                assert isinstance(app.screen, QueryModal)
                assert app.screen._is_error is True

    @pytest.mark.asyncio
    async def test_success_result_opens_modal_without_error(self, tmp_path):
        app = make_app(tmp_path)
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as pilot:
                app.post_message(QueryResult(result='Here is your summary', action_label='Summary', is_error=False))
                await pilot.pause()

                assert isinstance(app.screen, QueryModal)
                assert app.screen._is_error is False


class TestContextEdit:
    @pytest.mark.asyncio
    async def test_context_input_present_in_layout(self, tmp_path):
        app = make_app(tmp_path)
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test():
                assert app.query_one('#context-input', TextArea) is not None

    @pytest.mark.asyncio
    async def test_text_change_updates_controller(self, tmp_path):
        app = make_app(tmp_path)
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as pilot:
                text_area = app.query_one('#context-input', TextArea)
                text_area.focus()
                await pilot.pause()

                text_area.insert('Speaker A = Alice')
                await pilot.pause()

                assert app._controller.user_context == 'Speaker A = Alice'


class TestCopyEmptyTranscript:
    @pytest.mark.asyncio
    async def test_copy_empty_transcript_warns(self, tmp_path):
        app = make_app(tmp_path)
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as pilot:
                panel = app.query_one('#transcript-panel', TranscriptPanel)
                panel.focus()
                await pilot.pause()

                with patch('lazy_take_notes.l4_frameworks_and_drivers.widgets.transcript_panel.pyperclip') as mock_clip:
                    await pilot.press('c')
                    await pilot.pause()
                    mock_clip.copy.assert_not_called()


class TestModelDownloadProgress:
    @pytest.mark.asyncio
    async def test_first_progress_creates_download_modal(self, tmp_path):
        app = make_app(tmp_path)
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as pilot:
                app._audio_model_name = 'breeze-q5'
                with patch.object(app, 'push_screen'):
                    app.post_message(ModelDownloadProgress(percent=10, model_name='breeze-q5'))
                    await pilot.pause()

                assert app._download_modal is not None
                assert isinstance(app._download_modal, DownloadModal)

    @pytest.mark.asyncio
    async def test_second_progress_updates_existing_modal(self, tmp_path):
        app = make_app(tmp_path)
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as pilot:
                mock_modal = MagicMock()
                app._download_modal = mock_modal

                app.post_message(ModelDownloadProgress(percent=50, model_name='breeze-q5'))
                await pilot.pause()
                mock_modal.update_progress.assert_called_once_with(50)


class TestAudioWorkerStatusBranches:
    @pytest.mark.asyncio
    async def test_loading_model_with_download_modal_switches_to_loading(self, tmp_path):
        app = make_app(tmp_path)
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as pilot:
                mock_modal = MagicMock()
                app._download_modal = mock_modal

                app.post_message(AudioWorkerStatus(status='loading_model'))
                await pilot.pause()
                mock_modal.switch_to_loading.assert_called_once()

    @pytest.mark.asyncio
    async def test_model_ready_dismisses_download_modal(self, tmp_path):
        app = make_app(tmp_path)
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as pilot:
                mock_modal = MagicMock()
                app._download_modal = mock_modal

                app.post_message(AudioWorkerStatus(status='model_ready'))
                await pilot.pause()
                mock_modal.dismiss.assert_called_once()
                assert app._download_modal is None

    @pytest.mark.asyncio
    async def test_error_status_dismisses_modal_and_notifies(self, tmp_path):
        app = make_app(tmp_path)
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as pilot:
                app.post_message(AudioWorkerStatus(status='error', error='mic not found'))
                await pilot.pause()

                bar = app.query_one('#status-bar', StatusBar)
                assert bar.audio_status == 'error'


class TestDigestErrorHandler:
    @pytest.mark.asyncio
    async def test_digest_error_clears_activity(self, tmp_path):
        app = make_app(tmp_path)
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as pilot:
                bar = app.query_one('#status-bar', StatusBar)
                bar.activity = 'Digesting...'

                app.post_message(DigestError(error='LLM timeout', consecutive_failures=1))
                await pilot.pause()

                assert not bar.activity


class TestQuickActionGuards:
    @pytest.mark.asyncio
    async def test_blocked_while_digest_running(self, tmp_path):
        app = make_app(tmp_path)
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as pilot:
                app._digest_running = True
                with patch.object(app, '_run_query_worker') as mock_query:
                    app.action_quick_action('1')
                    await pilot.pause()
                    mock_query.assert_not_called()

    @pytest.mark.asyncio
    async def test_blocked_while_query_running(self, tmp_path):
        app = make_app(tmp_path)
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as pilot:
                app._query_running = True
                with patch.object(app, '_run_query_worker') as mock_query:
                    app.action_quick_action('1')
                    await pilot.pause()
                    mock_query.assert_not_called()


class TestForceDigestGuard:
    @pytest.mark.asyncio
    async def test_force_digest_noop_when_pending_quit(self, tmp_path):
        app = make_app(tmp_path)
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as pilot:
                segments = [TranscriptSegment(text='Data', wall_start=0.0, wall_end=1.0)]
                app.post_message(TranscriptChunk(segments=segments))
                await pilot.pause()

                app._pending_quit = True
                with patch.object(app, '_run_digest_worker') as mock_digest:
                    await pilot.press('d')
                    await pilot.pause()
                    mock_digest.assert_not_called()


class TestQuitGuards:
    @pytest.mark.asyncio
    async def test_q_disabled_while_recording(self, tmp_path):
        app = make_app(tmp_path)
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as pilot:
                app.post_message(AudioWorkerStatus(status='recording'))
                await pilot.pause()

                with patch.object(app, 'exit') as mock_exit:
                    await pilot.press('q')
                    await pilot.pause()
                    mock_exit.assert_not_called()
                    assert app._pending_quit is False

    @pytest.mark.asyncio
    async def test_q_disabled_while_paused(self, tmp_path):
        app = make_app(tmp_path)
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as pilot:
                app.post_message(AudioWorkerStatus(status='recording'))
                await pilot.pause()
                await pilot.press('space')  # pause
                await pilot.pause()

                bar = app.query_one('#status-bar', StatusBar)
                assert bar.paused is True

                with patch.object(app, 'exit') as mock_exit:
                    await pilot.press('q')
                    await pilot.pause()
                    mock_exit.assert_not_called()

    @pytest.mark.asyncio
    async def test_second_q_with_pending_quit_and_digest_running_warns(self, tmp_path):
        app = make_app(tmp_path)
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as pilot:
                app._pending_quit = True
                app._digest_running = True

                with patch.object(app, 'exit') as mock_exit:
                    await pilot.press('q')
                    await pilot.pause()
                    mock_exit.assert_not_called()

    @pytest.mark.asyncio
    async def test_q_when_digest_running_warns(self, tmp_path):
        app = make_app(tmp_path)
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as pilot:
                app._audio_stopped = True
                app._digest_running = True

                with patch.object(app, 'exit') as mock_exit:
                    await pilot.press('q')
                    await pilot.pause()
                    mock_exit.assert_not_called()


class TestAudioLevel:
    @pytest.mark.asyncio
    async def test_audio_level_updates_status_bar(self, tmp_path):
        app = make_app(tmp_path)
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as pilot:
                app.post_message(AudioLevel(rms=0.05))
                await pilot.pause()

                bar = app.query_one('#status-bar', StatusBar)
                assert bar.audio_level == 0.05


class TestStatusBarRender:
    @pytest.mark.asyncio
    async def test_render_download_state(self, tmp_path):
        app = make_app(tmp_path)
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as _pilot:
                bar = app.query_one('#status-bar', StatusBar)
                bar.download_percent = 50
                bar.download_model = 'breeze-q5'
                rendered = bar.render()
                assert 'Downloading' in rendered
                assert '50%' in rendered

    @pytest.mark.asyncio
    async def test_render_loading_model_state(self, tmp_path):
        app = make_app(tmp_path)
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as _pilot:
                bar = app.query_one('#status-bar', StatusBar)
                bar.audio_status = 'loading_model'
                rendered = bar.render()
                assert 'Loading model' in rendered

    @pytest.mark.asyncio
    async def test_render_error_state(self, tmp_path):
        app = make_app(tmp_path)
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as _pilot:
                bar = app.query_one('#status-bar', StatusBar)
                bar.audio_status = 'error'
                rendered = bar.render()
                assert 'Error' in rendered

    @pytest.mark.asyncio
    async def test_render_activity_indicator(self, tmp_path):
        app = make_app(tmp_path)
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as _pilot:
                bar = app.query_one('#status-bar', StatusBar)
                bar.activity = 'Digesting...'
                rendered = bar.render()
                assert 'Digesting' in rendered

    @pytest.mark.asyncio
    async def test_render_last_digest_time_seconds(self, tmp_path):
        app = make_app(tmp_path)
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as _pilot:
                bar = app.query_one('#status-bar', StatusBar)
                # Pin monotonic so the test works on fresh CI runners
                # where uptime < 30 s would make (monotonic() - 30) negative.
                with patch('time.monotonic', return_value=1000.0):
                    bar.last_digest_time = 1000.0 - 30
                    rendered = bar.render()
                assert 'last' in rendered
                assert 'ago' in rendered

    @pytest.mark.asyncio
    async def test_render_last_digest_time_minutes(self, tmp_path):
        app = make_app(tmp_path)
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as _pilot:
                bar = app.query_one('#status-bar', StatusBar)
                # Pin monotonic so the test works on fresh CI runners
                # where uptime < 120 s would make (monotonic() - 120) negative.
                with patch('time.monotonic', return_value=1000.0):
                    bar.last_digest_time = 1000.0 - 120
                    rendered = bar.render()
                assert 'last' in rendered
                assert 'm ago' in rendered

    @pytest.mark.asyncio
    async def test_render_recording_with_wave(self, tmp_path):
        app = make_app(tmp_path)
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as _pilot:
                bar = app.query_one('#status-bar', StatusBar)
                bar.recording = True
                rendered = bar.render()
                assert 'Rec' in rendered

    @pytest.mark.asyncio
    async def test_render_quick_action_hints(self, tmp_path):
        app = make_app(tmp_path)
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as _pilot:
                bar = app.query_one('#status-bar', StatusBar)
                bar.quick_action_hints = r'\[1] Catch Up  \[2] Todo'
                rendered = bar.render()
                assert 'Catch Up' in rendered


class TestFinalDigest:
    @pytest.mark.asyncio
    async def test_digest_ready_with_is_final_sets_flag(self, tmp_path):
        app = make_app(tmp_path)
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as pilot:
                app.post_message(DigestReady(markdown='## Final', digest_number=1, is_final=True))
                await pilot.pause()

                assert app._final_digest_done is True

    @pytest.mark.asyncio
    async def test_digest_ready_non_final_does_not_set_flag(self, tmp_path):
        app = make_app(tmp_path)
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as pilot:
                app.post_message(DigestReady(markdown='## Not final', digest_number=1, is_final=False))
                await pilot.pause()

                assert app._final_digest_done is False


# ---------------------------------------------------------------------------
# Step 4: Async worker tests (_run_digest_worker, _run_query_worker, _run_final_digest)
# ---------------------------------------------------------------------------


class TestDigestWorkerAsync:
    @pytest.mark.asyncio
    async def test_digest_success_posts_digest_ready(self, tmp_path):
        app = make_app(tmp_path)
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as pilot:
                # Seed buffer so digest has content
                segments = [TranscriptSegment(text='Line', wall_start=0.0, wall_end=1.0)]
                app.post_message(TranscriptChunk(segments=segments))
                await pilot.pause()

                # Trigger the real digest worker (FakeLLMClient returns instantly)
                app._run_digest_worker()
                # Allow async worker + message handler to complete
                await pilot.pause()
                await pilot.pause()

                assert app._digest_running is False
                panel = app.query_one('#digest-panel', DigestPanel)
                assert panel._current_markdown  # digest populated

    @pytest.mark.asyncio
    async def test_digest_failure_posts_digest_error(self, tmp_path):
        app = make_app(tmp_path)
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as pilot:
                segments = [TranscriptSegment(text='Line', wall_start=0.0, wall_end=1.0)]
                app.post_message(TranscriptChunk(segments=segments))
                await pilot.pause()

                # Make LLM return empty → triggers error path
                app._controller._digest_uc._llm.set_response('')  # type: ignore[union-attr]
                app._run_digest_worker()
                await pilot.pause()
                await pilot.pause()

                assert app._digest_running is False


class TestQueryWorkerAsync:
    @pytest.mark.asyncio
    async def test_query_success_posts_query_result(self, tmp_path):
        app = make_app(tmp_path)
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as pilot:
                # key='1' corresponds to first quick action in default_en
                app._run_query_worker('1')
                await pilot.pause()
                await pilot.pause()

                assert app._query_running is False
                assert isinstance(app.screen, QueryModal)

    @pytest.mark.asyncio
    async def test_query_exception_posts_error_result(self, tmp_path):
        app = make_app(tmp_path)
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as pilot:
                # Make the controller's run_quick_action raise
                with patch.object(app._controller, 'run_quick_action', side_effect=RuntimeError('LLM timeout')):
                    app._run_query_worker('1')
                    await pilot.pause()
                    await pilot.pause()

                assert app._query_running is False
                assert isinstance(app.screen, QueryModal)
                assert app.screen._is_error is True

    @pytest.mark.asyncio
    async def test_query_returns_none_no_modal(self, tmp_path):
        app = make_app(tmp_path)
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as pilot:
                # key='99' won't match any quick action → returns None
                app._run_query_worker('99')
                await pilot.pause()
                await pilot.pause()

                assert app._query_running is False
                # No QueryModal pushed for None result
                assert not isinstance(app.screen, QueryModal)

    @pytest.mark.asyncio
    async def test_query_label_resolution(self, tmp_path):
        app = make_app(tmp_path)
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as _pilot:
                bar = app.query_one('#status-bar', StatusBar)
                app._run_query_worker('1')
                # The status bar should show the quick action label, not '1'
                expected_label = app._template.quick_actions[0].label
                assert expected_label in bar.activity


class TestFinalDigestWorkerAsync:
    @pytest.mark.asyncio
    async def test_final_digest_success_posts_and_notifies(self, tmp_path):
        app = make_app(tmp_path)
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as pilot:
                segments = [TranscriptSegment(text='Final line', wall_start=0.0, wall_end=1.0)]
                app.post_message(TranscriptChunk(segments=segments))
                await pilot.pause()

                app._run_final_digest()
                await pilot.pause()
                await pilot.pause()

                assert app._digest_running is False
                assert app._final_digest_done is True

    @pytest.mark.asyncio
    async def test_final_digest_failure_posts_error(self, tmp_path):
        app = make_app(tmp_path)
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as pilot:
                segments = [TranscriptSegment(text='Final line', wall_start=0.0, wall_end=1.0)]
                app.post_message(TranscriptChunk(segments=segments))
                await pilot.pause()

                app._controller._digest_uc._llm.set_response('')  # type: ignore[union-attr]
                app._run_final_digest()
                await pilot.pause()
                await pilot.pause()

                assert app._digest_running is False
                assert app._final_digest_done is False  # error → not marked as done


# ---------------------------------------------------------------------------
# Step 6: Quit-when-already-stopped branches
# ---------------------------------------------------------------------------


class TestQuitWhenAlreadyStopped:
    @pytest.mark.asyncio
    async def test_quit_already_stopped_with_content_runs_final_digest(self, tmp_path):
        app = make_app(tmp_path)
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as pilot:
                segments = [TranscriptSegment(text='data', wall_start=0.0, wall_end=1.0)]
                app.post_message(TranscriptChunk(segments=segments))
                await pilot.pause()

                # Pre-stop audio so quit takes the was_already_stopped path
                app._audio_stopped = True
                bar = app.query_one('#status-bar', StatusBar)
                bar.stopped = True

                with patch.object(app, '_run_final_digest') as mock_final:
                    await pilot.press('q')
                    await pilot.pause()
                    mock_final.assert_called_once()
                    assert app._pending_quit is True

    @pytest.mark.asyncio
    async def test_quit_already_stopped_no_content_exits(self, tmp_path):
        app = make_app(tmp_path)
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as pilot:
                app._audio_stopped = True
                bar = app.query_one('#status-bar', StatusBar)
                bar.stopped = True

                with patch.object(app, 'exit') as mock_exit:
                    await pilot.press('q')
                    await pilot.pause()
                    mock_exit.assert_called_once()

    @pytest.mark.asyncio
    async def test_quit_already_stopped_final_done_exits(self, tmp_path):
        app = make_app(tmp_path)
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as pilot:
                segments = [TranscriptSegment(text='data', wall_start=0.0, wall_end=1.0)]
                app.post_message(TranscriptChunk(segments=segments))
                await pilot.pause()

                app._audio_stopped = True
                app._final_digest_done = True
                bar = app.query_one('#status-bar', StatusBar)
                bar.stopped = True

                with patch.object(app, 'exit') as mock_exit:
                    await pilot.press('q')
                    await pilot.pause()
                    mock_exit.assert_called_once()


# ---------------------------------------------------------------------------
# Step 7: Transcript chunk triggers digest
# ---------------------------------------------------------------------------


class TestTranscriptChunkDigestTrigger:
    @pytest.mark.asyncio
    async def test_digest_triggered_when_controller_says_should_digest(self, tmp_path):
        app = make_app(tmp_path)
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as pilot:
                app._digest_running = False
                with (
                    patch.object(app._controller, 'on_transcript_segments', return_value=True),
                    patch.object(app, '_run_digest_worker') as mock_digest,
                ):
                    app.post_message(
                        TranscriptChunk(segments=[TranscriptSegment(text='hello', wall_start=0.0, wall_end=1.0)])
                    )
                    await pilot.pause()
                    mock_digest.assert_called_once()

    @pytest.mark.asyncio
    async def test_digest_not_triggered_when_already_running(self, tmp_path):
        app = make_app(tmp_path)
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as pilot:
                app._digest_running = True
                with (
                    patch.object(app._controller, 'on_transcript_segments', return_value=True),
                    patch.object(app, '_run_digest_worker') as mock_digest,
                ):
                    app.post_message(
                        TranscriptChunk(segments=[TranscriptSegment(text='hello', wall_start=0.0, wall_end=1.0)])
                    )
                    await pilot.pause()
                    mock_digest.assert_not_called()


# ---------------------------------------------------------------------------
# Misc uncovered lines
# ---------------------------------------------------------------------------


class TestQuickActionCallsWorker:
    @pytest.mark.asyncio
    async def test_action_quick_action_calls_run_query_worker(self, tmp_path):
        """Line 487: action_quick_action delegates to _run_query_worker."""
        app = make_app(tmp_path)
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as pilot:
                with patch.object(app, '_run_query_worker') as mock_query:
                    app.action_quick_action('1')
                    await pilot.pause()
                    mock_query.assert_called_once_with('1')


class TestReportDownloadProgress:
    @pytest.mark.asyncio
    async def test_report_download_progress_posts_message(self, tmp_path):
        """Line 184: _report_download_progress posts ModelDownloadProgress."""
        app = make_app(tmp_path)
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as pilot:
                app._audio_model_name = 'test-model'
                app._report_download_progress(42)
                await pilot.pause()

                bar = app.query_one('#status-bar', StatusBar)
                assert bar.download_percent == 42


class TestStatusBarStopWhilePaused:
    @pytest.mark.asyncio
    async def test_stop_while_paused_includes_paused_time(self, tmp_path):
        """Line 75: watch_stopped when _pause_start is not None.

        action_stop_recording sets paused=False before stopped=True, which
        clears _pause_start. To hit line 75, set stopped=True directly while
        _pause_start is still active.
        """
        app = make_app(tmp_path)
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as pilot:
                bar = app.query_one('#status-bar', StatusBar)
                # Simulate that recording had started, then push _start_time
                # back so elapsed is large enough to absorb the paused time.
                bar._recording_started = True
                bar._start_time = time.monotonic() - 20.0
                bar._pause_start = time.monotonic() - 5.0
                bar._paused_total = 2.0
                # Set stopped directly — _pause_start still active
                bar.stopped = True
                await pilot.pause()

                # frozen = now - (now-20) - (2.0 + (now - (now-5))) ≈ 13s
                assert bar._frozen_elapsed is not None
                assert bar._frozen_elapsed >= 10.0


class TestStatusBarNoQuickActionHints:
    @pytest.mark.asyncio
    async def test_render_without_quick_action_hints(self, tmp_path):
        """Line 148: return left when quick_action_hints is empty."""
        app = make_app(tmp_path)
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as _pilot:
                bar = app.query_one('#status-bar', StatusBar)
                bar.quick_action_hints = ''
                rendered = bar.render()
                # Should return a single line (no \n from qa_line + '\n' + left)
                assert '\n' not in rendered


class TestPendingQuitWithContentRunsFinalDigest:
    @pytest.mark.asyncio
    async def test_stopped_status_with_pending_quit_and_content_runs_final_digest(self, tmp_path):
        """Line 283: AudioWorkerStatus(stopped) + pending_quit + has_content → _run_final_digest."""
        app = make_app(tmp_path)
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as pilot:
                segments = [TranscriptSegment(text='data', wall_start=0.0, wall_end=1.0)]
                app.post_message(TranscriptChunk(segments=segments))
                await pilot.pause()

                app._pending_quit = True
                with patch.object(app, '_run_final_digest') as mock_final:
                    app.post_message(AudioWorkerStatus(status='stopped'))
                    await pilot.pause()
                    mock_final.assert_called_once()


class TestBaseAppDefaults:
    """Exercise BaseApp methods that subclasses normally override."""

    @pytest.mark.asyncio
    async def test_base_hints_for_state(self, tmp_path):
        """BaseApp._hints_for_state returns default hint string."""
        app = make_app(tmp_path)
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test():
                from lazy_take_notes.l4_frameworks_and_drivers.apps.base import BaseApp

                base_hints = BaseApp._hints_for_state(app, 'idle')
                assert 'quit' in base_hints

    @pytest.mark.asyncio
    async def test_base_help_keybindings(self, tmp_path):
        """BaseApp._help_keybindings returns table rows."""
        app = make_app(tmp_path)
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test():
                from lazy_take_notes.l4_frameworks_and_drivers.apps.base import BaseApp

                rows = BaseApp._help_keybindings(app)
                assert any('Quit' in row for row in rows)

    @pytest.mark.asyncio
    async def test_base_action_quit_app(self, tmp_path):
        """BaseApp.action_quit_app calls self.exit()."""
        app = make_app(tmp_path)
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test():
                from lazy_take_notes.l4_frameworks_and_drivers.apps.base import BaseApp

                with patch.object(app, 'exit') as mock_exit:
                    BaseApp.action_quit_app(app)
                    mock_exit.assert_called_once()
