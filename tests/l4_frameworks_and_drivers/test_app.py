"""Tests for the Textual TUI app using headless Pilot."""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import patch

import pytest

from lazy_take_notes.l1_entities.transcript import TranscriptSegment
from lazy_take_notes.l3_interface_adapters.controllers.session_controller import SessionController
from lazy_take_notes.l3_interface_adapters.gateways.yaml_template_loader import YamlTemplateLoader
from lazy_take_notes.l4_frameworks_and_drivers.app import App
from lazy_take_notes.l4_frameworks_and_drivers.infra_config import build_app_config
from lazy_take_notes.l4_frameworks_and_drivers.messages import AudioWorkerStatus, DigestReady, TranscriptChunk
from lazy_take_notes.l4_frameworks_and_drivers.widgets.digest_panel import DigestPanel
from lazy_take_notes.l4_frameworks_and_drivers.widgets.status_bar import StatusBar
from lazy_take_notes.l4_frameworks_and_drivers.widgets.transcript_panel import TranscriptPanel
from tests.conftest import FakeLLMClient, FakePersistence


def make_app(
    tmp_path: Path,
    missing_digest_models: list[str] | None = None,
    missing_interactive_models: list[str] | None = None,
) -> App:
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
    return App(
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
                    mock_digest.assert_called_once_with(is_final=False)


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
