"""Fault tests for a live recording session.

A Textual handler or worker that raises ends the whole app, so each test drives
one failure into a running RecordApp and checks the session survives it.
Failures cover each boundary the event loop touches: the disk, the LLM, the
transcript text, the speaker bar, the template, and the session folder.
"""

from __future__ import annotations

import asyncio
import errno
import logging
import signal
from pathlib import Path
from unittest.mock import patch

import pytest
from textual.widgets import Input

from lazy_take_notes.l1_entities.chat_message import ChatMessage
from lazy_take_notes.l1_entities.speaker_turn import SpeakerTurn
from lazy_take_notes.l1_entities.transcript import TranscriptSegment
from lazy_take_notes.l2_use_cases.ports.llm_client import ChatResponse
from lazy_take_notes.l3_interface_adapters.controllers.session_controller import SessionController
from lazy_take_notes.l3_interface_adapters.gateways.yaml_template_loader import YamlTemplateLoader
from lazy_take_notes.l4_frameworks_and_drivers.apps.record import RecordApp
from lazy_take_notes.l4_frameworks_and_drivers.config import build_app_config
from lazy_take_notes.l4_frameworks_and_drivers.messages import (
    AudioLevel,
    AudioWorkerStatus,
    SpeakerTurns,
    TranscriptChunk,
)
from lazy_take_notes.l4_frameworks_and_drivers.screen_base import SessionHostApp
from lazy_take_notes.l4_frameworks_and_drivers.widgets.speaker_bar import SpeakerBar
from lazy_take_notes.l4_frameworks_and_drivers.widgets.status_bar import StatusBar
from lazy_take_notes.l4_frameworks_and_drivers.widgets.transcript_panel import TranscriptPanel
from tests.conftest import FailingPersistence, FakeLLMClient, FakePersistence

_ENOSPC = OSError(errno.ENOSPC, 'No space left on device')


@pytest.fixture(autouse=True)
def _no_real_exit_watchdog():
    """A signal arms a watchdog that calls os._exit; it must never fire inside pytest."""
    from lazy_take_notes.l4_frameworks_and_drivers import screen_base

    with patch.object(screen_base.threading, 'Timer'):
        yield


class GatedLLM(FakeLLMClient):
    """LLM whose chat calls wait until the test opens their gate, in call order."""

    def __init__(self) -> None:
        super().__init__(response='## Notes')
        self.gates: list[asyncio.Event] = []

    async def chat(self, model: str, messages: list[ChatMessage]) -> ChatResponse:
        gate = asyncio.Event()
        self.gates.append(gate)
        self.chat_calls.append((model, list(messages)))
        try:
            await gate.wait()
        except asyncio.CancelledError:
            # A real HTTP client needs a few loop turns to unwind a cancelled request.
            for _ in range(5):
                await asyncio.sleep(0)
            raise
        return ChatResponse(content=f'## Notes {len(self.gates)}', prompt_tokens=10)


def _host(screen) -> SessionHostApp:
    """The app hosting a screen under test; signal handling lives there."""
    host = screen.app
    assert isinstance(host, SessionHostApp)
    return host


def make_app(tmp_path: Path, *, llm=None, persistence=None, template=None, dir_name: str = 'output') -> RecordApp:
    config = build_app_config({'output': {'auto_label': False}})
    template = template or YamlTemplateLoader().load('default_en')
    output_dir = tmp_path / dir_name
    output_dir.mkdir(parents=True)
    controller = SessionController(
        config=config,
        template=template,
        llm_client=llm or FakeLLMClient(),
        persistence=persistence or FakePersistence(output_dir),
    )
    return RecordApp(config=config, template=template, output_dir=output_dir, controller=controller)


def _segment(text: str, start: float = 1.0) -> TranscriptSegment:
    return TranscriptSegment(text=text, wall_start=start, wall_end=start + 1.0)


def _notices(app: RecordApp) -> list[str]:
    return [str(notification.message) for notification in app.app._notifications]  # noqa: SLF001 -- read-only check of shown toasts


async def _settle(pilot, rounds: int = 3) -> None:
    for _ in range(rounds):
        await pilot.pause()


class TestTranscriptText:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'text',
        ['[/INST] hello', '[/]', '[bold]unclosed', '[link=x]y[/link]', '[[[', '\\', '[Music]', '\x00\x1b[31m'],
        ids=['closing-tag', 'bare-close', 'unclosed', 'link', 'brackets', 'backslash', 'music', 'control-chars'],
    )
    async def test_markup_like_text_is_shown_verbatim(self, tmp_path, text):
        app = make_app(tmp_path)
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as pilot:
                app.post_message(TranscriptChunk(segments=[_segment(text)]))
                await _settle(pilot)

                panel = app.query_one('#transcript-panel', TranscriptPanel)
                assert text in panel._all_text[-1]  # noqa: SLF001 -- plain-text mirror of the log

    @pytest.mark.asyncio
    async def test_markup_like_text_survives_a_speaker_redraw(self, tmp_path):
        app = make_app(tmp_path)
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as pilot:
                app.post_message(TranscriptChunk(segments=[_segment('[/INST] hi')]))
                app.post_message(SpeakerTurns(turns=[SpeakerTurn(start=0.0, end=5.0, speaker=1)]))
                await _settle(pilot)

                panel = app.query_one('#transcript-panel', TranscriptPanel)
                assert 'S1: [/INST] hi' in panel._all_text[-1]  # noqa: SLF001 -- plain-text mirror of the log


class TestSpeakerBar:
    @pytest.mark.asyncio
    async def test_a_click_on_the_bar_padding_does_nothing(self, tmp_path):
        app = make_app(tmp_path)
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as pilot:
                app.post_message(SpeakerTurns(turns=[SpeakerTurn(start=0.0, end=5.0, speaker=1)]))
                await _settle(pilot)

                await pilot.click('#speaker-bar', offset=(0, 0))
                await _settle(pilot)

                assert app.app.screen is app

    @pytest.mark.asyncio
    async def test_a_markup_like_speaker_name_is_shown_verbatim(self, tmp_path):
        app = make_app(tmp_path)
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as pilot:
                app.post_message(TranscriptChunk(segments=[_segment('hello')]))
                app.post_message(SpeakerTurns(turns=[SpeakerTurn(start=0.0, end=5.0, speaker=1)]))
                await _settle(pilot)

                await pilot.click('#speaker-1')
                await pilot.pause()
                app.app.screen.query_one('#label-input', Input).value = '[/b] Bob'
                await pilot.press('enter')
                await _settle(pilot)

                chip = app.query_one('#speaker-1')
                assert '[/b] Bob' in str(chip.render())
                panel = app.query_one('#transcript-panel', TranscriptPanel)
                assert '[/b] Bob: hello' in panel._all_text[-1]  # noqa: SLF001 -- plain-text mirror of the log
                assert isinstance(app.query_one('#speaker-bar'), SpeakerBar)

    @pytest.mark.asyncio
    async def test_a_failed_speakers_write_keeps_the_session(self, tmp_path):
        app = make_app(tmp_path)
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as pilot:
                with patch.object(Path, 'write_text', side_effect=_ENOSPC):
                    app.post_message(SpeakerTurns(turns=[SpeakerTurn(start=0.0, end=5.0, speaker=1)]))
                    await _settle(pilot)

                assert app.app.screen is app
                assert any('speakers' in notice for notice in _notices(app))


class TestDiskFailures:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'error',
        [_ENOSPC, PermissionError(errno.EACCES, 'Permission denied'), FileNotFoundError(errno.ENOENT, 'gone')],
        ids=['enospc', 'eacces', 'enoent'],
    )
    async def test_a_failed_transcript_write_keeps_recording_and_warns_once(self, tmp_path, error):
        app = make_app(tmp_path, persistence=FailingPersistence(error, {'save_transcript_lines'}))
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as pilot:
                for index in range(3):
                    app.post_message(TranscriptChunk(segments=[_segment(f'line {index}', start=float(index))]))
                    await _settle(pilot)

                panel = app.query_one('#transcript-panel', TranscriptPanel)
                assert len(panel._all_text) == 3  # noqa: SLF001 -- plain-text mirror of the log
                warnings = [notice for notice in _notices(app) if 'Could not save transcript' in notice]
                assert len(warnings) == 1

    @pytest.mark.asyncio
    @pytest.mark.parametrize('method', ['save_digest_md', 'save_history'])
    async def test_a_failed_notes_write_still_shows_the_digest(self, tmp_path, method):
        app = make_app(tmp_path, persistence=FailingPersistence(_ENOSPC, {method}))
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as pilot:
                app.post_message(TranscriptChunk(segments=[_segment('hello')]))
                await _settle(pilot)
                app.action_force_digest()
                await _settle(pilot, 6)

                assert app._controller.latest_digest == 'Fake LLM response'  # noqa: SLF001 -- controller state
                assert app._digest_running is False  # noqa: SLF001 -- worker flag
                assert any('Could not save notes' in notice for notice in _notices(app))


class TestDigestFailures:
    @pytest.mark.asyncio
    async def test_a_malformed_template_fails_the_digest_not_the_session(self, tmp_path):
        template = YamlTemplateLoader().load('default_en')
        broken = template.model_copy(update={'digest_user_template': 'stray { brace'})
        app = make_app(tmp_path, template=broken)
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as pilot:
                app.post_message(TranscriptChunk(segments=[_segment('hello')]))
                await _settle(pilot)
                app.action_force_digest()
                await _settle(pilot, 6)

                assert app._digest_running is False  # noqa: SLF001 -- worker flag
                assert any('Template error' in notice for notice in _notices(app))

    @pytest.mark.asyncio
    async def test_an_unexpected_digest_exception_fails_the_digest_not_the_session(self, tmp_path):
        app = make_app(tmp_path)
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as pilot:
                app.post_message(TranscriptChunk(segments=[_segment('hello')]))
                await _settle(pilot)
                with patch.object(app._controller, 'run_digest', side_effect=AssertionError('bug')):  # noqa: SLF001 -- inject a bug
                    app.action_force_digest()
                    await _settle(pilot, 6)

                assert app._digest_running is False  # noqa: SLF001 -- worker flag
                assert any('bug' in notice for notice in _notices(app))

    @pytest.mark.asyncio
    async def test_the_final_digest_replaces_a_running_digest_cleanly(self, tmp_path):
        llm = GatedLLM()
        app = make_app(tmp_path, llm=llm)
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as pilot:
                app.post_message(TranscriptChunk(segments=[_segment('hello')]))
                await _settle(pilot)
                app.action_force_digest()
                await _settle(pilot)
                assert len(llm.gates) == 1

                # The audio worker stops on its own while the periodic digest waits.
                app.post_message(AudioWorkerStatus(status='stopped'))
                await _settle(pilot)
                assert len(llm.gates) == 2
                assert app._digest_running is True  # noqa: SLF001 -- the final digest is still running

                sent_roles = [message.role for message in llm.chat_calls[-1][1]]
                assert sent_roles == ['system', 'user'], 'the final request must not carry the cancelled prompt'

                llm.gates[1].set()
                await _settle(pilot, 6)

                state = app._controller.digest_state  # noqa: SLF001 -- controller state
                roles = [message.role for message in state.messages]
                assert roles == ['system', 'user', 'assistant']
                assert app._digest_running is False  # noqa: SLF001 -- worker flag
                assert app._final_digest_done is True  # noqa: SLF001 -- final digest landed

    @pytest.mark.asyncio
    async def test_quit_waits_while_the_final_digest_replaces_a_running_digest(self, tmp_path):
        llm = GatedLLM()
        app = make_app(tmp_path, llm=llm)
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as pilot:
                app.post_message(TranscriptChunk(segments=[_segment('hello')]))
                await _settle(pilot)
                app.action_force_digest()
                await _settle(pilot)
                app.action_quit_app()  # first q: stop audio, wait for the flush
                app.post_message(AudioWorkerStatus(status='stopped'))
                await _settle(pilot)

                app.action_quit_app()  # second q while the final digest runs
                await _settle(pilot)

                assert app.app.screen is app
                llm.gates[-1].set()
                await _settle(pilot, 6)


class TestRename:
    @pytest.mark.asyncio
    async def test_a_long_cjk_label_fits_the_file_name_limit(self, tmp_path):
        app = make_app(tmp_path, dir_name='2026-02-21_143052')
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as pilot:
                app._on_label_result('會議' * 100)  # noqa: SLF001 -- the rename callback
                await pilot.pause()

                assert app._output_dir.exists()  # noqa: SLF001 -- renamed folder
                assert len(app._output_dir.name.encode()) <= 255  # noqa: SLF001 -- renamed folder

    @pytest.mark.asyncio
    async def test_the_rename_moves_the_folder_the_audio_writer_uses(self, tmp_path):
        app = make_app(tmp_path, dir_name='2026-02-21_143052')
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as pilot:
                app._on_label_result('standup')  # noqa: SLF001 -- the rename callback
                await pilot.pause()

                folder = app._session_folder  # noqa: SLF001 -- shared with the WAV writer
                assert folder.path == tmp_path / '2026-02-21_143052_standup'
                assert app._output_dir == folder.path  # noqa: SLF001 -- renamed folder
                assert not (tmp_path / '2026-02-21_143052').exists()

    @pytest.mark.asyncio
    async def test_an_existing_target_is_never_overwritten(self, tmp_path):
        app = make_app(tmp_path, dir_name='2026-02-21_143052')
        taken = tmp_path / '2026-02-21_143052_standup'
        taken.mkdir()
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as pilot:
                app._on_label_result('standup')  # noqa: SLF001 -- the rename callback
                await pilot.pause()

                assert app._output_dir.name == '2026-02-21_143052'  # noqa: SLF001 -- folder unchanged
                assert taken.exists()
                assert any('already exists' in notice for notice in _notices(app))

    @pytest.mark.asyncio
    async def test_a_failed_rename_keeps_the_session(self, tmp_path):
        app = make_app(tmp_path, dir_name='2026-02-21_143052')
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as pilot:
                with patch.object(Path, 'rename', side_effect=PermissionError(errno.EACCES, 'denied')):
                    app._on_label_result('standup')  # noqa: SLF001 -- the rename callback
                await pilot.pause()

                assert app._output_dir.name == '2026-02-21_143052'  # noqa: SLF001 -- folder unchanged
                assert not app._session_label  # noqa: SLF001 -- label not applied
                assert any('Could not rename' in notice for notice in _notices(app))


class TestCrashCapture:
    @pytest.mark.asyncio
    async def test_an_unexpected_callback_error_is_logged_and_the_session_continues(self, tmp_path, caplog):
        app = make_app(tmp_path)
        with patch.object(app, '_start_audio_worker'), caplog.at_level(logging.CRITICAL, logger='ltn'):
            async with app.run_test() as pilot:
                app.call_later(lambda: 1 / 0)
                await _settle(pilot)

                assert app.app.is_running
                assert any('ZeroDivisionError' in notice for notice in _notices(app))
                assert any('The session continues.' in notice for notice in _notices(app))

        records = [record for record in caplog.records if record.levelno == logging.CRITICAL]
        assert records[0].exc_info[0] is ZeroDivisionError

    @pytest.mark.asyncio
    async def test_a_raising_handler_does_not_stop_the_screen_from_handling_messages(self, tmp_path):
        app = make_app(tmp_path)
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as pilot:
                with patch.object(app._controller, 'on_transcript_segments', side_effect=[ValueError('bug'), False]):  # noqa: SLF001 -- inject a bug
                    app.post_message(TranscriptChunk(segments=[_segment('first')]))
                    await _settle(pilot)
                    app.post_message(TranscriptChunk(segments=[_segment('second', start=2.0)]))
                    await _settle(pilot)

                panel = app.query_one('#transcript-panel', TranscriptPanel)
                assert any('second' in line for line in panel._all_text)  # noqa: SLF001 -- plain-text mirror of the log
                assert app.app.is_running

    @pytest.mark.asyncio
    @pytest.mark.parametrize('thread', [False, True], ids=['async-worker', 'thread-worker'])
    async def test_a_worker_crash_logs_the_stack_and_the_session_continues(self, tmp_path, caplog, thread):
        import traceback

        def raising_worker_body():
            raise ZeroDivisionError('worker bug')

        async def raising_async_body():
            raising_worker_body()

        app = make_app(tmp_path)
        with patch.object(app, '_start_audio_worker'), caplog.at_level(logging.CRITICAL, logger='ltn'):
            async with app.run_test() as pilot:
                if thread:
                    app.run_worker(raising_worker_body, thread=True)
                else:
                    app.run_worker(raising_async_body)
                await _settle(pilot, 6)
                assert app.app.is_running

        records = [record for record in caplog.records if record.levelno == logging.CRITICAL]
        assert records
        logged = ''.join(traceback.format_exception(*records[0].exc_info))
        assert 'raising_worker_body' in logged
        assert 'worker bug' in logged

    @pytest.mark.asyncio
    async def test_an_error_storm_raises_one_alert_and_keeps_recording(self, tmp_path, caplog):
        from lazy_take_notes.l4_frameworks_and_drivers.screen_base import ERROR_BURST_LIMIT

        caplog.set_level('CRITICAL', logger='ltn.app')
        app = make_app(tmp_path)
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as pilot:
                with (
                    patch('lazy_take_notes.l4_frameworks_and_drivers.screen_base.SHUTDOWN_GRACE_S', 0.05),
                    patch.object(app.app, 'exit') as exit_app,
                ):
                    for _ in range(ERROR_BURST_LIMIT * 3):
                        app.call_later(lambda: 1 / 0)
                    await _settle(pilot, 4)
                    await asyncio.sleep(0.2)
                    await _settle(pilot)

                    assert not app._audio_shutdown.is_set()  # noqa: SLF001 -- the recording goes on
                    exit_app.assert_not_called()
                    alerts = [n for n in app.app._notifications if 'Errors keep repeating' in n.message]  # noqa: SLF001 -- Textual keeps the shown toasts here
                    assert len(alerts) == 1
                    tracebacks = [r for r in caplog.records if r.exc_info]
                    assert len(tracebacks) == ERROR_BURST_LIMIT + 1  # then only counts, so the log stays small

                    with patch('lazy_take_notes.l4_frameworks_and_drivers.screen_base.STORM_LOG_INTERVAL_S', 0.0):
                        app.call_later(lambda: 1 / 0)
                        await _settle(pilot)
                    assert 'more errors' in caplog.records[-1].getMessage()
                    assert 'ZeroDivisionError' in caplog.records[-1].getMessage()


class TestShutdownSignals:
    @pytest.mark.asyncio
    @pytest.mark.parametrize('signum', [signal.SIGTERM, signal.SIGHUP], ids=['sigterm', 'sighup'])
    async def test_a_signal_stops_audio_saves_context_and_exits_after_the_flush(self, tmp_path, signum):
        persistence = FakePersistence(tmp_path / 'output')
        app = make_app(tmp_path, persistence=persistence)
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as pilot:
                app.post_message(AudioWorkerStatus(status='recording'))
                app.post_message(TranscriptChunk(segments=[_segment('hello')]))
                await _settle(pilot)
                app._controller.user_context = 'agenda'  # noqa: SLF001 -- typed context

                with patch.object(app.app, 'exit') as exit_app:
                    _host(app).handle_shutdown_signal(signum)
                    await _settle(pilot)

                    assert app._audio_shutdown.is_set()  # noqa: SLF001 -- audio worker told to stop
                    assert persistence.context_calls == ['agenda']
                    exit_app.assert_not_called()  # waits for the audio flush

                    app.post_message(AudioWorkerStatus(status='stopped'))
                    await _settle(pilot, 6)

                    exit_app.assert_called()
                    assert persistence.digest_calls == []  # a signal skips the final digest

    @pytest.mark.asyncio
    async def test_a_signal_exits_even_if_the_audio_worker_never_stops(self, tmp_path):
        app = make_app(tmp_path)
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as pilot:
                with (
                    patch('lazy_take_notes.l4_frameworks_and_drivers.screen_base.SHUTDOWN_GRACE_S', 0.05),
                    patch.object(app.app, 'exit') as exit_app,
                ):
                    _host(app).handle_shutdown_signal(signal.SIGTERM)
                    await asyncio.sleep(0.2)
                    await _settle(pilot)

                    exit_app.assert_called()

    @pytest.mark.asyncio
    async def test_a_signal_with_no_session_exits_at_once(self, tmp_path):
        from textual.screen import Screen

        from lazy_take_notes.l4_frameworks_and_drivers.screen_base import ShellScreen

        class Idle(ShellScreen):
            def compose(self):
                return []

        screen = Idle()
        async with screen.run_test() as pilot:
            with patch.object(screen.app, 'exit') as exit_app:
                _host(screen).handle_shutdown_signal(signal.SIGTERM)
                await pilot.pause()

                exit_app.assert_called_once()
        assert isinstance(screen, Screen)

    @pytest.mark.asyncio
    async def test_the_host_installs_and_restores_signal_handlers(self, tmp_path):
        app = make_app(tmp_path)
        before = {signum: signal.getsignal(signum) for signum in (signal.SIGTERM, signal.SIGHUP)}
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as pilot:
                await pilot.pause()
                for signum in before:
                    assert signal.getsignal(signum) == _host(app)._on_os_signal  # noqa: SLF001 -- installed handler
        assert {signum: signal.getsignal(signum) for signum in before} == before


class TestQuitKeys:
    @pytest.mark.asyncio
    async def test_ctrl_q_stops_the_recording_before_the_app_exits(self, tmp_path):
        # Textual's ctrl+q only ends the app; the audio thread then records on with no screen.
        persistence = FakePersistence(tmp_path / 'output')
        app = make_app(tmp_path, persistence=persistence)
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as pilot:
                app.post_message(AudioWorkerStatus(status='recording'))
                await _settle(pilot)
                app._controller.user_context = 'agenda'  # noqa: SLF001 -- typed context

                with patch.object(app.app, 'exit') as exit_app:
                    await pilot.press('ctrl+q')
                    await _settle(pilot)

                    assert app._audio_shutdown.is_set()  # noqa: SLF001 -- audio worker told to stop
                    assert persistence.context_calls == ['agenda']
                    exit_app.assert_not_called()  # waits for the audio flush

                    app.post_message(AudioWorkerStatus(status='stopped'))
                    await _settle(pilot, 6)

                    exit_app.assert_called()

    @pytest.mark.asyncio
    async def test_ctrl_c_during_a_recording_names_the_stop_key(self, tmp_path):
        app = make_app(tmp_path)
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as pilot:
                app.post_message(AudioWorkerStatus(status='recording'))
                await _settle(pilot)
                await pilot.press('ctrl+c')
                await _settle(pilot)

                notice = _notices(app)[-1]
                assert 'Press s to stop and save' in notice
                assert 'ctrl+q' in notice

    @pytest.mark.asyncio
    async def test_ctrl_c_when_not_recording_keeps_textual_s_hint(self, tmp_path):
        app = make_app(tmp_path)
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as pilot:
                app.post_message(AudioWorkerStatus(status='stopped'))
                await _settle(pilot)
                await pilot.press('ctrl+c')
                await _settle(pilot)

                assert 'to quit the app' in _notices(app)[-1]


class TestQuitDuringADigest:
    @pytest.mark.asyncio
    async def test_q_during_the_final_digest_quits_when_it_ends(self, tmp_path):
        llm = GatedLLM()
        app = make_app(tmp_path, llm=llm)
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as pilot:
                app.post_message(AudioWorkerStatus(status='recording'))
                app.post_message(TranscriptChunk(segments=[_segment('hello')]))
                await _settle(pilot)
                await pilot.press('s')
                app.post_message(AudioWorkerStatus(status='stopped'))
                await _settle(pilot)
                assert app._digest_running  # noqa: SLF001 -- the final digest waits on the gate

                with patch.object(app, 'dismiss') as dismiss:
                    await pilot.press('q')
                    await _settle(pilot)
                    dismiss.assert_not_called()
                    assert any('Quitting when the digest finishes' in n for n in _notices(app))

                    llm.gates[-1].set()
                    await _settle(pilot, 6)
                    dismiss.assert_called_once()

    @pytest.mark.asyncio
    async def test_q_during_a_periodic_digest_still_writes_the_final_notes(self, tmp_path):
        llm = GatedLLM()
        app = make_app(tmp_path, llm=llm)
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as pilot:
                app.post_message(AudioWorkerStatus(status='recording'))
                app.post_message(TranscriptChunk(segments=[_segment('hello')]))
                await _settle(pilot)
                await pilot.press('d')  # a periodic digest
                await _settle(pilot)
                app._audio_stopped = True  # noqa: SLF001 -- stopped while the digest runs
                app._audio_worker_done = True  # noqa: SLF001 -- and the flush finished
                bar = app.query_one('#status-bar', StatusBar)
                bar.recording = False
                bar.stopped = True

                with patch.object(app, 'dismiss') as dismiss:
                    await pilot.press('q')
                    llm.gates[0].set()
                    await _settle(pilot, 6)
                    assert len(llm.gates) == 2  # the final digest started after the periodic one
                    dismiss.assert_not_called()
                    llm.gates[1].set()
                    await _settle(pilot, 6)
                    dismiss.assert_called_once()

    @pytest.mark.asyncio
    async def test_a_failed_final_digest_still_quits_once(self, tmp_path):
        class Failing(GatedLLM):
            async def chat(self, model, messages):
                await super().chat(model, messages)
                raise RuntimeError('model gone')

        llm = Failing()
        app = make_app(tmp_path, llm=llm)
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as pilot:
                app.post_message(AudioWorkerStatus(status='recording'))
                app.post_message(TranscriptChunk(segments=[_segment('hello')]))
                await _settle(pilot)
                await pilot.press('s')
                app.post_message(AudioWorkerStatus(status='stopped'))
                await _settle(pilot)
                with patch.object(app, 'dismiss') as dismiss:
                    await pilot.press('q')
                    llm.gates[-1].set()
                    await _settle(pilot, 8)
                    assert len(llm.gates) == 1  # no retry loop
                    dismiss.assert_called_once()


class TestCloseSession:
    @pytest.mark.asyncio
    async def test_it_closes_once_even_when_called_again(self, tmp_path):
        # A second q and the label worker's end both reach close_session.
        app = make_app(tmp_path)
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as pilot:
                with patch.object(app, 'dismiss') as dismiss:
                    app.close_session()
                    app.close_session()
                    app._pending_quit = True  # noqa: SLF001 -- a late label worker end
                    app._continue_pending_quit()  # noqa: SLF001
                    await pilot.pause()
                    dismiss.assert_called_once()

    @pytest.mark.asyncio
    async def test_a_modal_on_top_is_closed_first(self, tmp_path):
        from lazy_take_notes.l4_frameworks_and_drivers.widgets.help_modal import HelpModal

        app = make_app(tmp_path)
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as pilot:
                app.app.push_screen(HelpModal(body_md='help'))
                await pilot.pause()
                tops: list = []
                with patch.object(app, 'dismiss', side_effect=lambda *_: tops.append(app.app.screen)):
                    app.close_session()
                    await pilot.pause()
                assert tops == [app]  # dismiss pops the top screen; it must be this one

    @pytest.mark.asyncio
    async def test_q_waits_for_the_audio_flush_before_the_final_digest(self, tmp_path):
        llm = GatedLLM()
        app = make_app(tmp_path, llm=llm)
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as pilot:
                app.post_message(AudioWorkerStatus(status='recording'))
                app.post_message(TranscriptChunk(segments=[_segment('hello')]))
                await _settle(pilot)
                await pilot.press('d')
                await pilot.press('s')
                await _settle(pilot)
                with patch.object(app, 'dismiss') as dismiss:
                    await pilot.press('q')
                    llm.gates[0].set()  # the periodic digest ends during the flush
                    await _settle(pilot, 6)
                    assert len(llm.gates) == 1  # no final digest before the last lines arrive

                    app.post_message(TranscriptChunk(segments=[_segment('last words', 2.0)]))
                    app.post_message(AudioWorkerStatus(status='stopped'))
                    await _settle(pilot, 6)
                    assert len(llm.gates) == 2
                    assert 'last words' in str(llm.chat_calls[-1][1])
                    llm.gates[1].set()
                    await _settle(pilot, 6)
                    dismiss.assert_called_once()

    @pytest.mark.asyncio
    async def test_ctrl_c_with_a_modal_open_still_warns_about_ctrl_q(self, tmp_path):
        from lazy_take_notes.l4_frameworks_and_drivers.widgets.help_modal import HelpModal

        app = make_app(tmp_path)
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as pilot:
                app.post_message(AudioWorkerStatus(status='recording'))
                await _settle(pilot)
                app.app.push_screen(HelpModal(body_md='help'))
                await pilot.pause()
                _host(app).action_help_quit()
                await pilot.pause()
                assert 'Press s to stop and save' in _notices(app)[-1]


class TestSignalPortability:
    @pytest.mark.asyncio
    async def test_a_host_that_cannot_install_handlers_still_runs_the_session(self, tmp_path):
        app = make_app(tmp_path)
        with (
            patch.object(app, '_start_audio_worker'),
            patch(
                'lazy_take_notes.l4_frameworks_and_drivers.screen_base.signal.signal',
                side_effect=ValueError('not main thread'),
            ),
        ):
            async with app.run_test() as pilot:
                await pilot.pause()
                assert app.app.screen is app

    def test_only_signals_the_platform_has_are_used(self, monkeypatch):
        from lazy_take_notes.l4_frameworks_and_drivers.screen_base import shutdown_signals

        monkeypatch.delattr(signal, 'SIGHUP')

        assert shutdown_signals() == (signal.SIGTERM,)


class TestForcedExit:
    def test_a_signal_arms_a_daemon_watchdog(self):
        from lazy_take_notes.l4_frameworks_and_drivers import screen_base

        with patch.object(screen_base.threading, 'Timer') as timer:
            screen_base.arm_exit_watchdog(signal.SIGHUP)

        delay, target = timer.call_args.args[:2]
        assert delay == screen_base.SHUTDOWN_GRACE_S + screen_base.FORCE_EXIT_AFTER_S
        assert timer.return_value.daemon is True
        timer.return_value.start.assert_called_once()
        with patch.object(screen_base.os, '_exit') as hard_exit:
            target(*timer.call_args.args[2])
        hard_exit.assert_called_once_with(128 + signal.SIGHUP)

    @pytest.mark.asyncio
    async def test_the_signal_handler_arms_the_watchdog(self, tmp_path):
        app = make_app(tmp_path)
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as pilot:
                with (
                    patch('lazy_take_notes.l4_frameworks_and_drivers.screen_base.arm_exit_watchdog') as arm,
                    patch.object(app.app, 'exit'),
                ):
                    _host(app).handle_shutdown_signal(signal.SIGHUP)
                    await pilot.pause()

                arm.assert_called_once_with(signal.SIGHUP)

    @pytest.mark.asyncio
    async def test_the_os_handler_arms_the_watchdog_before_the_loop_runs(self, tmp_path):
        # A hung event loop never runs queued callbacks; the watchdog must not depend on it.
        app = make_app(tmp_path)
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as pilot:
                with (
                    patch('lazy_take_notes.l4_frameworks_and_drivers.screen_base.arm_exit_watchdog') as arm,
                    patch.object(app.app, 'end_sessions') as end_sessions,
                ):
                    _host(app)._on_os_signal(signal.SIGHUP, None)  # noqa: SLF001 -- the installed OS handler
                    arm.assert_called_once_with(signal.SIGHUP)
                    end_sessions.assert_not_called()  # the clean stop waits for the loop
                    await pilot.pause()
                    end_sessions.assert_called_once()
                    arm.assert_called_once()  # armed once, not again by the loop callback


class TestAudioHealthDisplay:
    """A lost source is visible for as long as it is lost, not for one toast."""

    @pytest.mark.asyncio
    async def test_a_lost_mic_stays_on_the_status_bar_until_it_returns(self, tmp_path):
        from lazy_take_notes.l4_frameworks_and_drivers.messages import AudioHealth
        from lazy_take_notes.l4_frameworks_and_drivers.widgets.status_bar import StatusBar

        app = make_app(tmp_path)
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as pilot:
                bar = app.query_one('#status-bar', StatusBar)
                app.post_message(AudioHealth(mic='lost', system='ok', detail='Microphone: unplugged'))
                await _settle(pilot)
                assert 'MIC LOST' in bar.render()
                assert any('Microphone' in notice and 'lost' in notice.lower() for notice in _notices(app))

                app.post_message(AudioHealth(mic='ok', system='lost'))
                await _settle(pilot)
                rendered = bar.render()
                assert 'MIC LOST' not in rendered
                assert 'SYS LOST' in rendered
                assert any('Microphone is back' in notice for notice in _notices(app))

                app.post_message(AudioHealth(mic='silent', system='ok'))
                await _settle(pilot)
                assert 'MIC SILENT' in bar.render()
                assert 'SYS LOST' not in bar.render()

                app.post_message(AudioHealth(mic='ok', system='ok', processing='error', detail='bad chunk'))
                await _settle(pilot)
                assert 'AUDIO ERROR' in bar.render()
                assert any('Nothing is recorded' in notice for notice in _notices(app))
                app.post_message(AudioHealth(mic='ok', system='ok'))
                await _settle(pilot)
                assert 'AUDIO ERROR' not in bar.render()

    @pytest.mark.asyncio
    async def test_a_raising_status_bar_render_does_not_freeze_the_alert(self, tmp_path):
        from lazy_take_notes.l4_frameworks_and_drivers.messages import AudioHealth
        from lazy_take_notes.l4_frameworks_and_drivers.widgets.status_bar import StatusBar

        app = make_app(tmp_path)
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as pilot:
                bar = app.query_one('#status-bar', StatusBar)
                with patch.object(StatusBar, 'watch_audio_level', side_effect=ValueError('bug')):
                    app.post_message(AudioLevel(rms=0.5))
                    await _settle(pilot)
                original_render = StatusBar.render
                calls = {'count': 0}

                def render_fails_once(widget):
                    calls['count'] += 1
                    if calls['count'] == 1:
                        raise ValueError('render bug')
                    return original_render(widget)

                with patch.object(StatusBar, 'render', render_fails_once):
                    bar.refresh()
                    await _settle(pilot)
                    app.post_message(AudioHealth(mic='lost', system='ok'))
                    await _settle(pilot)
                    await pilot.pause(0.1)
                assert calls['count'] >= 2  # the failing render ran, and later renders still happened
                screen_text = app.app.export_screenshot().replace('&#160;', ' ')
                assert 'MIC' in screen_text
                assert 'LOST' in screen_text
                assert app.app.is_running
