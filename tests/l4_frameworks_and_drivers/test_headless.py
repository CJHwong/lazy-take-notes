"""Tests for the headless session runner.

Drives HeadlessSession with a fake worker launcher (no real audio/whisper) and
a real SessionController wired to FakeLLMClient + FakePersistence.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast
from unittest.mock import MagicMock

import click
import pytest

from lazy_take_notes.l1_entities import session_status as st
from lazy_take_notes.l1_entities.transcript import TranscriptSegment
from lazy_take_notes.l3_interface_adapters.controllers.session_controller import SessionController
from lazy_take_notes.l3_interface_adapters.gateways.session_status import read_status
from lazy_take_notes.l4_frameworks_and_drivers.config import build_app_config
from lazy_take_notes.l4_frameworks_and_drivers.headless import (
    HeadlessSession,
    _banner_lines,  # noqa: PLC2701 -- testing module-private formatter
    _clock,  # noqa: PLC2701 -- testing module-private formatter
    _heartbeat_line,  # noqa: PLC2701 -- testing module-private formatter
    _select_headless_template,  # noqa: PLC2701 -- testing module-private selector
    _shared_worker_kwargs,  # noqa: PLC2701 -- testing module-private helper
    _summary_lines,  # noqa: PLC2701 -- testing module-private formatter
    _warn_missing_models,  # noqa: PLC2701 -- testing module-private helper
)
from lazy_take_notes.l4_frameworks_and_drivers.messages import (
    AudioWorkerStatus,
    ModelDownloadProgress,
    TranscriptChunk,
)


@pytest.fixture(autouse=True)
def _no_caffeinate(monkeypatch):
    """Never spawn the real `caffeinate` inhibitor during tests."""
    monkeypatch.setattr(
        'lazy_take_notes.l4_frameworks_and_drivers.keep_awake.inhibit_sleep',
        lambda: None,
    )


def _seg(text: str) -> TranscriptSegment:
    return TranscriptSegment(text=text, wall_start=0.0, wall_end=1.0)


def _read(tmp_path: Path) -> st.SessionStatus:
    status = read_status(tmp_path)
    assert status is not None
    return status


def _controller(template, persistence, fake_llm, *, eager_digest: bool = False) -> SessionController:
    raw = {'digest': {'min_lines': 1, 'min_interval': 0.0, 'max_lines': 1}} if eager_digest else {}
    config = build_app_config(raw)
    return SessionController(config=config, template=template, llm_client=fake_llm, persistence=persistence)


class TestHeadlessSession:
    @pytest.mark.asyncio
    async def test_runs_to_stopped_and_persists(self, tmp_path: Path, default_template, fake_persistence, fake_llm):
        controller = _controller(default_template, fake_persistence, fake_llm)

        def launcher(post_message, _is_cancelled):
            post_message(AudioWorkerStatus(status='model_ready'))
            post_message(TranscriptChunk(segments=[_seg('hello'), _seg('world')]))
            post_message(AudioWorkerStatus(status='stopped'))

        session = HeadlessSession(controller, tmp_path, launcher)
        await session.run()

        status = read_status(tmp_path)
        assert status is not None
        assert status.state == st.STOPPED
        assert status.segment_count == 2
        # Transcript was persisted, and a final digest ran on completion.
        assert fake_persistence.transcript_calls
        assert fake_persistence.digest_calls

    @pytest.mark.asyncio
    async def test_worker_error_sets_error_state(self, tmp_path: Path, default_template, fake_persistence, fake_llm):
        controller = _controller(default_template, fake_persistence, fake_llm)

        def launcher(post_message, _is_cancelled):
            post_message(AudioWorkerStatus(status='error', error='device gone'))

        session = HeadlessSession(controller, tmp_path, launcher)
        await session.run()

        status = _read(tmp_path)
        assert status.state == st.ERROR
        assert status.error == 'device gone'
        # No final digest after an error.
        assert not fake_persistence.digest_calls

    @pytest.mark.asyncio
    async def test_incremental_digest_triggers_on_chunk(
        self, tmp_path: Path, default_template, fake_persistence, fake_llm
    ):
        controller = _controller(default_template, fake_persistence, fake_llm, eager_digest=True)

        def launcher(post_message, _is_cancelled):
            post_message(TranscriptChunk(segments=[_seg('trigger me')]))
            post_message(AudioWorkerStatus(status='stopped'))

        session = HeadlessSession(controller, tmp_path, launcher)
        await session.run()

        # One incremental digest (from the chunk) + one final digest.
        assert len(fake_persistence.digest_calls) >= 2
        assert _read(tmp_path).state == st.STOPPED

    @pytest.mark.asyncio
    async def test_progress_messages_update_state(self, tmp_path: Path, default_template, fake_persistence, fake_llm):
        controller = _controller(default_template, fake_persistence, fake_llm)

        def launcher(post_message, _is_cancelled):
            post_message(AudioWorkerStatus(status='loading_model'))
            post_message(ModelDownloadProgress(percent=50, model_name='whatever'))
            post_message(AudioWorkerStatus(status='recording'))
            post_message(AudioWorkerStatus(status='stopped'))

        session = HeadlessSession(controller, tmp_path, launcher)
        await session.run()

        # No transcript, no digest content -> ends stopped without a digest.
        assert _read(tmp_path).state == st.STOPPED
        assert not fake_persistence.digest_calls

    def test_request_stop_sets_cancel_and_hints_once(
        self, tmp_path: Path, default_template, fake_persistence, fake_llm
    ):
        controller = _controller(default_template, fake_persistence, fake_llm)
        reported: list[str] = []
        session = HeadlessSession(controller, tmp_path, lambda *_: None, report=reported.append)

        session.request_stop()
        session.request_stop()  # mashing Ctrl-C must not repeat the hint

        assert session._cancel.is_set()  # noqa: SLF001 -- asserting internal stop flag
        assert sum('stopping' in line for line in reported) == 1

    def test_toggle_mic_flips_and_reports(self, tmp_path: Path, default_template, fake_persistence, fake_llm):
        class _Src:
            mic_muted = False

        controller = _controller(default_template, fake_persistence, fake_llm)
        src = _Src()
        reported: list[str] = []
        session = HeadlessSession(controller, tmp_path, lambda *_: None, report=reported.append, audio_source=src)

        session._toggle_mic()  # noqa: SLF001 -- exercising the SIGUSR1 handler
        assert src.mic_muted is True
        assert any('muted' in line for line in reported)

        session._toggle_mic()  # noqa: SLF001 -- toggle back
        assert src.mic_muted is False
        assert any('unmuted' in line for line in reported)

    @pytest.mark.asyncio
    async def test_run_with_audio_source_installs_mic_toggle(
        self, tmp_path: Path, default_template, fake_persistence, fake_llm
    ):
        class _Src:
            mic_muted = False

        controller = _controller(default_template, fake_persistence, fake_llm)

        def launcher(post_message, _is_cancelled):
            post_message(AudioWorkerStatus(status='stopped'))

        session = HeadlessSession(controller, tmp_path, launcher, audio_source=_Src())
        await session.run()  # exercises _install_mic_toggle with a source present

        assert _read(tmp_path).state == st.STOPPED


class TestHeadlessEntryPoints:
    def _ctx(self):
        ctx = MagicMock()
        ctx.obj = {'config_path': None, 'output_dir': None}
        return ctx

    def _patch_common(self, monkeypatch, tmp_path, template):
        from lazy_take_notes.l4_frameworks_and_drivers import headless as hl

        config = build_app_config({'output': {'directory': str(tmp_path)}})
        loader = MagicMock()
        loader.load.return_value = template
        monkeypatch.setattr(hl, 'load_config', lambda *a, **k: (config, MagicMock(), loader))
        monkeypatch.setattr(hl, 'preflight_llm', lambda *a, **k: ([], []))
        monkeypatch.setattr(hl, '_build_container', lambda *a, **k: MagicMock())
        captured: dict = {}
        monkeypatch.setattr(hl, '_run', lambda out_dir, cfg, session: captured.update(out_dir=out_dir, session=session))
        return hl, captured

    def test_run_record_headless_wires_session(self, tmp_path, default_template, monkeypatch):
        hl, captured = self._patch_common(monkeypatch, tmp_path, default_template)

        hl.run_record_headless(self._ctx(), template_name='default_en', label='demo')

        assert captured['out_dir'].exists()
        assert isinstance(captured['session'], hl.HeadlessSession)

    def test_run_record_headless_mute_mic(self, tmp_path, default_template, monkeypatch):
        from unittest.mock import MagicMock as _MagicMock

        hl, _ = self._patch_common(monkeypatch, tmp_path, default_template)
        container = _MagicMock()
        monkeypatch.setattr(hl, '_build_container', lambda *a, **k: container)

        hl.run_record_headless(self._ctx(), template_name='default_en', mute_mic=True)

        assert container.audio_source.mic_muted is True

    def test_run_transcribe_headless_wires_session(self, tmp_path, default_template, monkeypatch):
        hl, captured = self._patch_common(monkeypatch, tmp_path, default_template)
        audio = tmp_path / 'in.wav'
        audio.touch()

        hl.run_transcribe_headless(self._ctx(), audio_path=audio, template_name='default_en')

        assert captured['out_dir'].exists()
        assert isinstance(captured['session'], hl.HeadlessSession)

    def test_unknown_template_raises_click_exception(self, monkeypatch):
        from lazy_take_notes.l4_frameworks_and_drivers import headless as hl

        loader = MagicMock()
        loader.load.side_effect = FileNotFoundError('nope')
        loader.list_templates.return_value = [MagicMock(key='default_en')]
        monkeypatch.setattr(hl, 'load_config', lambda *a, **k: (build_app_config({}), MagicMock(), loader))

        with pytest.raises(click.ClickException, match='Unknown template'):
            hl.run_record_headless(self._ctx(), template_name='bogus')


class TestHeadlessHelpers:
    def test_build_container_constructs(self, default_template, monkeypatch):
        from lazy_take_notes.l4_frameworks_and_drivers import headless as hl

        sentinel = object()
        monkeypatch.setattr(
            'lazy_take_notes.l4_frameworks_and_drivers.container.DependencyContainer',
            lambda *a, **k: sentinel,
        )

        result = hl._build_container(build_app_config({}), default_template, Path('/tmp/x'), MagicMock())  # noqa: SLF001 -- testing module helper

        assert result is sentinel

    def test_run_launches_session(self, tmp_path: Path, monkeypatch):
        from lazy_take_notes.l4_frameworks_and_drivers import headless as hl

        monkeypatch.setattr(
            'lazy_take_notes.l4_frameworks_and_drivers.logging_setup.setup_file_logging',
            lambda *a, **k: None,
        )
        ran: dict = {}

        class _FakeSession:
            status = st.SessionStatus(
                state=st.STOPPED, pid=1, started_at='2026-06-06T09:00:00', updated_at='2026-06-06T09:00:30'
            )

            async def run(self):
                ran['ok'] = True

        hl._run(tmp_path, build_app_config({}), cast(hl.HeadlessSession, _FakeSession()))  # noqa: SLF001 -- testing module helper

        assert ran['ok'] is True


class TestHeadlessReporting:
    @pytest.mark.asyncio
    async def test_milestones_reported(self, tmp_path: Path, default_template, fake_persistence, fake_llm):
        from lazy_take_notes.l4_frameworks_and_drivers.messages import ModelDownloadProgress

        controller = _controller(default_template, fake_persistence, fake_llm)
        reported: list[str] = []

        def launcher(post_message, _is_cancelled):
            post_message(AudioWorkerStatus(status='loading_model'))
            post_message(ModelDownloadProgress(percent=10, model_name='ggml'))
            post_message(ModelDownloadProgress(percent=15, model_name='ggml'))  # same bucket -> not re-reported
            post_message(AudioWorkerStatus(status='recording'))
            post_message(AudioWorkerStatus(status='stopped'))

        session = HeadlessSession(controller, tmp_path, launcher, report=reported.append)
        await session.run()

        assert '… loading model' in reported
        assert '✓ model ready, recording' in reported
        assert sum('downloading' in line for line in reported) == 1  # throttled to one per 20% bucket

    @pytest.mark.asyncio
    async def test_transcription_error_does_not_end_session(
        self, tmp_path: Path, default_template, fake_persistence, fake_llm
    ):
        # A recoverable per-chunk error after recording started must NOT end the session.
        controller = _controller(default_template, fake_persistence, fake_llm)
        reported: list[str] = []

        def launcher(post_message, _is_cancelled):
            post_message(AudioWorkerStatus(status='recording'))  # marks started
            post_message(AudioWorkerStatus(status='error', error='chunk failed'))  # recoverable
            post_message(TranscriptChunk(segments=[_seg('still recording')]))
            post_message(AudioWorkerStatus(status='stopped'))

        session = HeadlessSession(controller, tmp_path, launcher, report=reported.append)
        await session.run()

        assert any('transcription error (continuing)' in line for line in reported)
        assert _read(tmp_path).state == st.STOPPED  # not ERROR
        assert len(controller.all_segments) == 1  # kept processing after the error

    @pytest.mark.asyncio
    async def test_pump_times_out_after_cancel(
        self, tmp_path: Path, default_template, fake_persistence, fake_llm, monkeypatch
    ):
        import lazy_take_notes.l4_frameworks_and_drivers.headless as hl

        monkeypatch.setattr(hl, 'STOP_GRACE', 0.01)
        controller = _controller(default_template, fake_persistence, fake_llm)
        reported: list[str] = []

        def launcher(post_message, _is_cancelled):
            pass  # wedged worker: never posts 'stopped'

        session = HeadlessSession(controller, tmp_path, launcher, report=reported.append)
        session.request_stop()  # cancel before the pump starts
        await session.run()

        assert any('did not stop in time' in line for line in reported)
        assert _read(tmp_path).state == st.STOPPED

    @pytest.mark.asyncio
    async def test_run_records_error_on_exception(
        self, tmp_path: Path, default_template, fake_persistence, fake_llm, monkeypatch
    ):
        controller = _controller(default_template, fake_persistence, fake_llm)

        def boom(_segments):
            raise RuntimeError('disk full')

        monkeypatch.setattr(controller, 'on_transcript_segments', boom)  # force _pump to raise

        def launcher(post_message, _is_cancelled):
            post_message(TranscriptChunk(segments=[_seg('x')]))
            post_message(AudioWorkerStatus(status='stopped'))

        session = HeadlessSession(controller, tmp_path, launcher)
        with pytest.raises(RuntimeError, match='disk full'):
            await session.run()

        assert _read(tmp_path).state == st.ERROR

    @pytest.mark.asyncio
    async def test_no_sound_warning_reported(self, tmp_path: Path, default_template, fake_persistence, fake_llm):
        controller = _controller(default_template, fake_persistence, fake_llm)
        reported: list[str] = []

        def launcher(post_message, _is_cancelled):
            post_message(AudioWorkerStatus(status='warning', error='Audio signal lost: no sound from source'))
            post_message(AudioWorkerStatus(status='stopped'))

        session = HeadlessSession(controller, tmp_path, launcher, report=reported.append)
        await session.run()

        assert any('⚠' in line and 'no sound' in line for line in reported)

    @pytest.mark.asyncio
    async def test_error_reported(self, tmp_path: Path, default_template, fake_persistence, fake_llm):
        controller = _controller(default_template, fake_persistence, fake_llm)
        reported: list[str] = []

        def launcher(post_message, _is_cancelled):
            post_message(AudioWorkerStatus(status='error', error='boom'))

        session = HeadlessSession(controller, tmp_path, launcher, report=reported.append)
        await session.run()

        assert any('✗ error: boom' in line for line in reported)

    @pytest.mark.asyncio
    async def test_error_during_shutdown_is_graceful(
        self, tmp_path: Path, default_template, fake_persistence, fake_llm
    ):
        # A teardown error after the user asked to stop must not flip the session to 'error'.
        controller = _controller(default_template, fake_persistence, fake_llm)
        reported: list[str] = []

        def launcher(post_message, _is_cancelled):
            post_message(AudioWorkerStatus(status='error', error='[Errno 32] Broken pipe'))

        session = HeadlessSession(controller, tmp_path, launcher, report=reported.append)
        session.request_stop()  # user already pressed Ctrl-C
        await session.run()

        assert _read(tmp_path).state == st.STOPPED
        assert not any('✗ error' in line for line in reported)

    @pytest.mark.asyncio
    async def test_digest_failure_reported(self, tmp_path: Path, default_template, fake_persistence, fake_llm):
        fake_llm.set_response('')  # empty response -> digest use case returns an error
        controller = _controller(default_template, fake_persistence, fake_llm, eager_digest=True)
        reported: list[str] = []

        def launcher(post_message, _is_cancelled):
            post_message(TranscriptChunk(segments=[_seg('x')]))
            post_message(AudioWorkerStatus(status='stopped'))

        session = HeadlessSession(controller, tmp_path, launcher, report=reported.append)
        await session.run()

        assert any('✗ digest failed' in line for line in reported)

    @pytest.mark.asyncio
    async def test_heartbeat_emits_line(self, tmp_path: Path, default_template, fake_persistence, fake_llm):
        import asyncio

        controller = _controller(default_template, fake_persistence, fake_llm)
        reported: list[str] = []
        session = HeadlessSession(
            controller, tmp_path, lambda *_: None, status_sink=reported.append, activity='recording'
        )

        beat = asyncio.create_task(session._heartbeat(0.01))  # noqa: SLF001 -- exercising the heartbeat loop
        await asyncio.sleep(0.05)
        beat.cancel()

        assert any(line.startswith('⏺ recording') for line in reported)

    @pytest.mark.asyncio
    async def test_duplicate_warnings_deduped(self, tmp_path: Path, default_template, fake_persistence, fake_llm):
        controller = _controller(default_template, fake_persistence, fake_llm)
        reported: list[str] = []
        warn = AudioWorkerStatus(status='warning', error='Audio signal lost: no sound from source')

        def launcher(post_message, _is_cancelled):
            post_message(warn)
            post_message(warn)
            post_message(warn)
            post_message(AudioWorkerStatus(status='stopped'))

        session = HeadlessSession(controller, tmp_path, launcher, report=reported.append)
        await session.run()

        assert sum('Audio signal lost' in line for line in reported) == 1

    @pytest.mark.asyncio
    async def test_warning_after_recovery_shown_again(
        self, tmp_path: Path, default_template, fake_persistence, fake_llm
    ):
        controller = _controller(default_template, fake_persistence, fake_llm)
        reported: list[str] = []
        warn = AudioWorkerStatus(status='warning', error='Audio signal lost: no sound from source')

        def launcher(post_message, _is_cancelled):
            post_message(warn)
            post_message(TranscriptChunk(segments=[_seg('audio is back')]))  # recovery
            post_message(warn)
            post_message(AudioWorkerStatus(status='stopped'))

        session = HeadlessSession(controller, tmp_path, launcher, report=reported.append)
        await session.run()

        assert sum('Audio signal lost' in line for line in reported) == 2


class _FakeStream:
    def __init__(self, tty: bool):
        self.buf = ''
        self._tty = tty

    def write(self, s: str) -> None:
        self.buf += s

    def flush(self) -> None:
        pass

    def isatty(self) -> bool:
        return self._tty


class TestStatusWriter:
    def test_tty_status_in_place_then_commits_on_line(self):
        from lazy_take_notes.l4_frameworks_and_drivers.headless import _StatusWriter  # noqa: PLC2701

        stream = _FakeStream(tty=True)
        writer = _StatusWriter(stream)

        writer.status('⏺ 01:00')
        assert stream.buf == '\r⏺ 01:00\x1b[K'

        writer.status('⏺ 02:00')  # overwrites the same line in place
        assert stream.buf == '\r⏺ 01:00\x1b[K\r⏺ 02:00\x1b[K'

        stream.buf = ''
        writer.line('✓ digest')  # commits the heartbeat snapshot (\n), then prints the line
        assert stream.buf == '\n✓ digest\n'

        stream.buf = ''
        writer.status('⏺ 03:00')  # fresh in-place line
        writer.finish()  # commits it on exit
        assert stream.buf == '\r⏺ 03:00\x1b[K\n'

    def test_non_tty_suppresses_status_and_uses_plain_lines(self):
        from lazy_take_notes.l4_frameworks_and_drivers.headless import _StatusWriter  # noqa: PLC2701

        stream = _FakeStream(tty=False)
        writer = _StatusWriter(stream)

        writer.status('⏺ 01:00')
        assert not stream.buf  # heartbeat suppressed off-TTY

        writer.line('✓ digest')
        assert stream.buf == '✓ digest\n'

        writer.finish()  # no-op
        assert stream.buf == '✓ digest\n'


class TestHeadlessTemplateSelection:
    def _loader(self):
        from lazy_take_notes.l3_interface_adapters.gateways.yaml_template_loader import YamlTemplateLoader

        return YamlTemplateLoader()

    def test_language_wins(self):
        assert _select_headless_template(self._loader(), None, 'zh-TW').metadata.key == 'default_zh_tw'

    def test_template_key(self):
        assert _select_headless_template(self._loader(), 'sprint_retro_en', None).metadata.key == 'sprint_retro_en'

    def test_default_when_nothing_given(self):
        assert _select_headless_template(self._loader(), None, None).metadata.key == 'default_en'


class TestHeadlessHelpers2:
    def test_shared_worker_kwargs(self, default_template):
        kw = _shared_worker_kwargs(build_app_config({}), default_template)
        assert kw['language'] == default_template.metadata.locale.split('-')[0].lower()
        assert 'chunk_duration' in kw
        assert 'recognition_hints' in kw

    def test_warn_missing_models(self, capsys):
        _warn_missing_models(['digest-model'], ['action-model'])
        err = capsys.readouterr().err
        assert 'digest-model' in err
        assert 'action-model' in err

    def test_warn_missing_models_empty_is_silent(self, capsys):
        _warn_missing_models([], [])
        assert not capsys.readouterr().err


class TestHeadlessFormatters:
    def test_clock(self):
        assert _clock(0) == '00:00'
        assert _clock(65) == '01:05'
        assert _clock(-5) == '00:00'

    def test_heartbeat_line_pluralizes(self):
        assert _heartbeat_line('recording', 60, 8, 1) == '⏺ recording · 01:00 · 8 seg · 1 digest'
        assert _heartbeat_line('recording', 0, 0, 0) == '⏺ recording · 00:00 · 0 seg · 0 digests'

    def test_banner_lines(self):
        lines = _banner_lines('recording', Path('/x/sess'), 8123)
        assert lines[0] == '▶ recording → /x/sess'
        assert 'kill -INT 8123' in lines[1]
        assert 'lazy-take-notes status' in lines[2]
        assert not any('USR1' in line for line in lines)

    def test_banner_lines_with_mute(self):
        lines = _banner_lines('recording', Path('/x/sess'), 8123, show_mute=True)
        assert any('kill -USR1 8123' in line for line in lines)

    def test_summary_lines_includes_paths(self):
        status = st.SessionStatus(
            state=st.STOPPED,
            pid=1,
            started_at='2026-06-06T09:00:00',
            updated_at='2026-06-06T09:12:30',
            segment_count=45,
            digest_count=3,
        )
        lines = _summary_lines(Path('/x/sess'), status)
        assert lines[0] == '■ stopped · 12:30 · 45 seg · 3 digests'
        assert any('transcript: /x/sess/transcript.txt' in line for line in lines)
        assert any('notes:      /x/sess/notes.md' in line for line in lines)

    def test_summary_lines_with_error(self):
        status = st.SessionStatus(
            state=st.ERROR,
            pid=1,
            started_at='2026-06-06T09:00:00',
            updated_at='2026-06-06T09:00:03',
            error='device gone',
        )
        lines = _summary_lines(Path('/x/sess'), status)
        assert any('error:      device gone' in line for line in lines)

    def test_summary_lines_minimal_when_empty(self):
        status = st.SessionStatus(
            state=st.STOPPED, pid=1, started_at='2026-06-06T09:00:00', updated_at='2026-06-06T09:00:05'
        )
        lines = _summary_lines(Path('/x/sess'), status)
        assert lines == ['■ stopped · 00:05 · 0 seg · 0 digests']
