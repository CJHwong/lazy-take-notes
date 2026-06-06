"""Headless session runner — drives the workers and controller without a TUI.

Reuses the same worker functions and ``SessionController`` as the Textual apps.
A worker runs in a background thread and posts messages through a thread-safe
sink into an asyncio queue drained on the main thread, so all controller
mutation stays single-threaded — the same invariant the TUI relies on.

Output is files only (transcript + notes, written by the controller). Progress
is exposed through a ``.status.json`` file that the ``status`` CLI command reads.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
import threading
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import click

from lazy_take_notes.l1_entities import session_status as st
from lazy_take_notes.l1_entities.session_files import NOTES, TRANSCRIPT
from lazy_take_notes.l1_entities.session_status import SessionStatus
from lazy_take_notes.l3_interface_adapters.gateways.session_status import write_status
from lazy_take_notes.l4_frameworks_and_drivers.cli_helpers import (
    load_config,
    load_template_key,
    make_session_dir,
    preflight_llm,
    preflight_microphone,
    resolve_base_dir,
    resolve_language,
    set_mic_muted,
)
from lazy_take_notes.l4_frameworks_and_drivers.messages import (
    AudioWorkerStatus,
    ModelDownloadProgress,
    TranscriptChunk,
)

if TYPE_CHECKING:
    from lazy_take_notes.l3_interface_adapters.controllers.session_controller import SessionController

log = logging.getLogger('ltn.headless')

HEARTBEAT_INTERVAL = 60.0

# After a stop is requested, don't wait forever for a wedged worker to post
# 'stopped' — give up after this grace period so Ctrl-C always returns control.
STOP_GRACE = 30.0

# Worker status string -> our session state.
_WORKER_STATE = {
    'loading_model': st.LOADING_MODEL,
    'model_ready': st.RECORDING,
    'recording': st.RECORDING,
    'warning': st.RECORDING,
}


def _now() -> str:
    return datetime.now().isoformat(timespec='seconds')


class _StatusWriter:
    """Renders permanent log lines plus a single in-place status (heartbeat) line.

    On a TTY the heartbeat rewrites one line in place via ``\\r``. A permanent
    line first commits whatever heartbeat is currently shown (a ``\\n`` leaves
    that last snapshot in scrollback) and prints below it, so milestones never
    overwrite live status and the status line never shows stale numbers — the
    committed snapshot was accurate when it was frozen, and the next heartbeat
    starts a fresh in-place line. When the stream is not a TTY (piped / log
    file) the heartbeat is suppressed; liveness comes from the milestone lines
    and the ``status`` command, keeping logs clean.
    """

    def __init__(self, stream) -> None:
        self._stream = stream
        self._tty = stream.isatty()
        self._pending = False  # an in-place heartbeat line is currently shown

    def line(self, message: str) -> None:
        if self._tty and self._pending:
            self._stream.write('\n')  # commit the in-place heartbeat snapshot as-is
            self._pending = False
        self._stream.write(message + '\n')
        self._stream.flush()

    def status(self, message: str) -> None:
        if not self._tty:
            return
        self._stream.write(f'\r{message}\x1b[K')
        self._pending = True
        self._stream.flush()

    def finish(self) -> None:
        if self._tty and self._pending:
            self._stream.write('\n')
            self._stream.flush()
            self._pending = False


def _clock(seconds: float) -> str:
    total = max(int(seconds), 0)
    minutes, secs = divmod(total, 60)
    return f'{minutes:02d}:{secs:02d}'


def _heartbeat_line(activity: str, elapsed: float, segments: int, digests: int) -> str:
    plural = '' if digests == 1 else 's'
    return f'⏺ {activity} · {_clock(elapsed)} · {segments} seg · {digests} digest{plural}'


def _banner_lines(activity: str, out_dir: Path, pid: int, *, show_mute: bool = False) -> list[str]:
    lines = [
        f'▶ {activity} → {out_dir}',
        f'  stop:  Ctrl-C  (or kill -INT {pid})',
    ]
    if show_mute:
        lines.append(f'  mute:  kill -USR1 {pid}')
    lines.append('  check: lazy-take-notes status')
    return lines


def _iso_elapsed(start_iso: str, end_iso: str) -> float:
    try:
        return (datetime.fromisoformat(end_iso) - datetime.fromisoformat(start_iso)).total_seconds()
    except ValueError:  # pragma: no cover -- timestamps always ISO from _now()
        return 0.0


def _summary_lines(out_dir: Path, status: SessionStatus) -> list[str]:
    plural = '' if status.digest_count == 1 else 's'
    clock = _clock(_iso_elapsed(status.started_at, status.updated_at))
    head = f'■ {status.state} · {clock} · {status.segment_count} seg · {status.digest_count} digest{plural}'
    lines = [head]
    if status.segment_count:
        lines.append(f'  transcript: {out_dir / TRANSCRIPT.name}')
    if status.digest_count:
        lines.append(f'  notes:      {out_dir / NOTES.name}')
    if status.error:
        lines.append(f'  error:      {status.error}')
    return lines


class HeadlessSession:
    """Pumps worker messages into the controller and tracks status on disk.

    *worker_launcher* is a blocking callable ``(post_message, is_cancelled)``
    run in a background thread; it drives one of the existing worker functions.
    """

    def __init__(
        self,
        controller: SessionController,
        output_dir: Path,
        worker_launcher: Callable[[Callable, Callable[[], bool]], None],
        report: Callable[[str], None] | None = None,
        activity: str = 'recording',
        audio_source=None,
        status_sink: Callable[[str], None] | None = None,
    ) -> None:
        self._controller = controller
        self._output_dir = Path(output_dir)
        self._launch_worker = worker_launcher
        self._writer = _StatusWriter(sys.stderr)
        self._report = report or self._writer.line
        self._status_sink = status_sink or self._writer.status
        self._activity = activity
        self._audio_source = audio_source
        self._last_dl_bucket = -1
        self._last_warning: str | None = None
        self._started = False  # worker reached model_ready/recording
        self._cancel = threading.Event()
        self._status = SessionStatus(
            state=st.STARTING,
            pid=os.getpid(),
            started_at=_now(),
            updated_at=_now(),
        )

    @property
    def status(self) -> SessionStatus:
        return self._status

    def request_stop(self) -> None:
        """Signal the worker to wind down (wired to SIGINT)."""
        if not self._cancel.is_set():
            self._report('⏹ stopping… finishing the current chunk, then the final digest')
        self._cancel.set()

    def _elapsed_seconds(self) -> float:
        return _iso_elapsed(self._status.started_at, _now())

    def _save(self, state: str | None = None, error: str | None = None) -> None:
        if state is not None:
            self._status.state = state
        if error is not None:
            self._status.error = error
        self._status.segment_count = len(self._controller.all_segments)
        self._status.digest_count = self._controller.digest_state.digest_count
        self._status.updated_at = _now()
        write_status(self._output_dir, self._status)

    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def sink(message) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, message)

        self._install_signal_handler(loop)
        self._install_mic_toggle(loop)
        self._save()

        thread = threading.Thread(
            target=self._launch_worker,
            args=(sink, self._cancel.is_set),
            daemon=True,
        )
        thread.start()

        beat = asyncio.create_task(self._heartbeat(HEARTBEAT_INTERVAL))
        try:
            await self._pump(queue)
            thread.join(timeout=5)
            await self._finalize()
        except Exception as exc:
            log.error('headless session aborted: %s', exc, exc_info=True)
            try:
                self._save(state=st.ERROR, error=str(exc))
            except Exception:  # noqa: S110 -- best-effort; original error matters # pragma: no cover
                pass
            raise
        finally:
            beat.cancel()
            self._writer.finish()

    async def _heartbeat(self, interval: float) -> None:
        while True:
            await asyncio.sleep(interval)
            self._status_sink(
                _heartbeat_line(
                    self._activity,
                    self._elapsed_seconds(),
                    len(self._controller.all_segments),
                    self._controller.digest_state.digest_count,
                )
            )

    def _install_signal_handler(self, loop: asyncio.AbstractEventLoop) -> None:
        try:
            loop.add_signal_handler(signal.SIGINT, self.request_stop)
        except (NotImplementedError, RuntimeError):  # pragma: no cover -- platform fallback (e.g. Windows)
            signal.signal(signal.SIGINT, lambda *_: self.request_stop())

    def _install_mic_toggle(self, loop: asyncio.AbstractEventLoop) -> None:
        if self._audio_source is None:
            return
        if not hasattr(signal, 'SIGUSR1'):  # pragma: no cover -- non-POSIX (e.g. Windows)
            return
        try:
            loop.add_signal_handler(signal.SIGUSR1, self._toggle_mic)
        except (NotImplementedError, RuntimeError):  # pragma: no cover -- platform fallback
            signal.signal(signal.SIGUSR1, lambda *_: self._toggle_mic())

    def _toggle_mic(self) -> None:
        muted = not getattr(self._audio_source, 'mic_muted', False)
        set_mic_muted(self._audio_source, muted)
        self._report('⏸ mic muted' if muted else '▶ mic unmuted')

    async def _pump(self, queue: asyncio.Queue) -> None:
        while True:
            try:
                # After a stop request, don't block forever on a wedged worker.
                timeout = STOP_GRACE if self._cancel.is_set() else None
                message = await asyncio.wait_for(queue.get(), timeout)
            except TimeoutError:
                self._report('⏹ worker did not stop in time; ending session')
                return
            if isinstance(message, TranscriptChunk):
                await self._on_chunk(message)
            elif isinstance(message, AudioWorkerStatus):
                if message.status == 'stopped':
                    return
                if message.status == 'error':
                    if self._handle_error(message):
                        return  # fatal -> stop the pump
                    continue  # recoverable -> keep recording
                if message.status in ('model_ready', 'recording'):
                    self._started = True
                if message.status == 'warning':
                    if message.error != self._last_warning:
                        self._report(f'⚠ {message.error}')
                        self._last_warning = message.error
                else:
                    self._report_status(message.status)
                self._save(state=_WORKER_STATE.get(message.status, self._status.state))
            elif isinstance(message, ModelDownloadProgress):
                self._report_download(message)
                self._save(state=st.DOWNLOADING)
            # TranscriptionStatus / AudioLevel carry no headless-relevant state.

    def _handle_error(self, message: AudioWorkerStatus) -> bool:
        """Handle a worker 'error'. Returns True if the session should stop.

        The audio worker emits 'error' for both fatal startup failures (model
        load / device init, before recording begins) and recoverable mid-session
        failures (e.g. a single chunk failed to transcribe — the worker keeps
        recording). A recoverable error must NOT end the session, matching the
        TUI which only notifies and carries on.
        """
        if self._cancel.is_set():
            # Teardown noise during a user-requested stop — log, then finalize cleanly.
            log.info('worker error during shutdown: %s', message.error)
            return True
        if not self._started:
            self._report(f'✗ error: {message.error}')
            self._save(state=st.ERROR, error=message.error)
            return True
        self._report(f'✗ transcription error (continuing): {message.error}')
        return False

    def _report_status(self, status: str) -> None:
        if status == 'loading_model':
            self._report('… loading model')
        elif status == 'recording':
            self._report(f'✓ model ready, {self._activity}')

    def _report_download(self, message: ModelDownloadProgress) -> None:
        bucket = message.percent // 20
        if bucket != self._last_dl_bucket:
            self._last_dl_bucket = bucket
            self._report(f'… downloading {message.model_name} {message.percent}%')

    async def _on_chunk(self, message: TranscriptChunk) -> None:
        self._last_warning = None  # audio is flowing again — let a later loss warn afresh
        should_digest = self._controller.on_transcript_segments(message.segments)
        if should_digest:
            await self._digest(is_final=False)
        else:
            self._save(state=st.RECORDING)

    async def _digest(self, *, is_final: bool) -> None:
        self._save(state=st.DIGESTING)
        result = await self._controller.run_digest(is_final=is_final)
        if result.data is not None:
            self._report(f'✓ notes.md updated (digest #{self._controller.digest_state.digest_count})')
            self._save(state=st.RECORDING)
        else:
            error = result.error or 'digest failed'
            self._report(f'✗ digest failed: {error}')
            self._save(state=st.RECORDING, error=error)

    async def _finalize(self) -> None:
        if self._status.state == st.ERROR:
            return
        state = self._controller.digest_state
        if state.buffer or state.digest_count > 0:
            await self._digest(is_final=True)
        self._save(state=st.STOPPED)


def _shared_worker_kwargs(config, template) -> dict:
    """Transcription kwargs common to both the live-audio and file workers."""
    tc = config.transcription
    return {
        'language': template.metadata.locale.split('-')[0].lower(),
        'chunk_duration': tc.chunk_duration,
        'overlap': tc.overlap,
        'silence_threshold': tc.silence_threshold,
        'pause_duration': tc.pause_duration,
        'recognition_hints': list(dict.fromkeys(config.recognition_hints + template.recognition_hints)),
    }


def _audio_launcher(config, template, output_dir, model_resolver_factory, transcriber, audio_source):
    """Build the blocking launcher that runs the live audio worker."""

    def launch(post_message, is_cancelled):  # pragma: no cover -- thread body; drives real audio/whisper
        from lazy_take_notes.l3_interface_adapters.gateways.hf_model_resolver import (  # noqa: PLC0415 -- deferred: fallback resolver
            HfModelResolver,
        )
        from lazy_take_notes.l4_frameworks_and_drivers.workers.audio_worker import (  # noqa: PLC0415 -- deferred: audio stack loaded only when session starts
            run_audio_worker,
        )

        model_name = config.transcription.model_for_locale(template.metadata.locale)

        def on_progress(percent: int) -> None:
            post_message(ModelDownloadProgress(percent=percent, model_name=model_name))

        try:
            resolver = (model_resolver_factory or (lambda cb: HfModelResolver(on_progress=cb)))(on_progress)
            model_path = resolver.resolve(model_name)
        except Exception as exc:  # noqa: BLE001 -- surface model-resolution failure as a worker error
            post_message(AudioWorkerStatus(status='error', error=str(exc)))
            return

        run_audio_worker(
            post_message=post_message,
            is_cancelled=is_cancelled,
            model_path=model_path,
            output_dir=output_dir,
            save_audio=config.output.save_audio,
            transcriber=transcriber,
            audio_source=audio_source,
            **_shared_worker_kwargs(config, template),
        )

    return launch


def _file_launcher(config, template, audio_path, model_resolver_factory, transcriber):
    """Build the blocking launcher that transcribes an audio file."""

    def launch(post_message, is_cancelled):  # pragma: no cover -- thread body; drives ffmpeg/whisper
        from lazy_take_notes.l4_frameworks_and_drivers.workers.file_transcription_worker import (  # noqa: PLC0415 -- deferred: loaded only when session starts
            run_file_transcription,
        )

        run_file_transcription(
            post_message=post_message,
            is_cancelled=is_cancelled,
            audio_path=audio_path,
            model_name=config.transcription.model_for_locale(template.metadata.locale),
            transcriber=transcriber,
            model_resolver_factory=model_resolver_factory,
            **_shared_worker_kwargs(config, template),
        )

    return launch


def _select_headless_template(template_loader, template_name: str | None, language: str | None):
    """Pick the template for a headless session: --language, then --template, then default_en."""
    if language:
        return resolve_language(template_loader, language)
    return load_template_key(template_loader, template_name or 'default_en')


def _build_container(config, template, out_dir, infra, **overrides):
    from lazy_take_notes.l4_frameworks_and_drivers.container import (  # noqa: PLC0415 -- deferred: container not loaded for --help
        DependencyContainer,
    )

    return DependencyContainer(config, template, out_dir, infra=infra, **overrides)


def _run(out_dir: Path, config, session: HeadlessSession) -> None:
    from lazy_take_notes.l4_frameworks_and_drivers.logging_setup import (  # noqa: PLC0415 -- deferred
        setup_file_logging,
    )

    setup_file_logging(out_dir, enabled=config.output.save_debug_log)
    asyncio.run(session.run())
    for line in _summary_lines(out_dir, session.status):
        click.echo(line, err=True)
    click.echo(str(out_dir))


def _emit_banner(activity: str, out_dir: Path, pid: int, *, show_mute: bool = False) -> None:
    for line in _banner_lines(activity, out_dir, pid, show_mute=show_mute):
        click.echo(line, err=True)


def _warn_missing_models(missing_digest: list[str], missing_interactive: list[str]) -> None:
    """Warn up front about reachable-but-unavailable models (headless has no TUI to show them)."""
    for model in dict.fromkeys([*missing_digest, *missing_interactive]):
        click.echo(f'⚠ model not available: {model} (cycles using it will fail)', err=True)


def run_record_headless(
    ctx: click.Context,
    *,
    template_name: str | None = None,
    language: str | None = None,
    label: str | None = None,
    mute_mic: bool = False,
) -> None:
    """Run a live recording session headlessly (no TUI)."""
    from lazy_take_notes.l4_frameworks_and_drivers.keep_awake import keep_awake  # noqa: PLC0415 -- deferred

    config, infra, template_loader = load_config(ctx.obj['config_path'], ctx.obj['output_dir'])
    template = _select_headless_template(template_loader, template_name, language)

    out_dir = make_session_dir(resolve_base_dir(ctx.obj['output_dir'], config), label)
    _warn_missing_models(*preflight_llm(infra, config))
    preflight_microphone()

    container = _build_container(config, template, out_dir, infra)
    if mute_mic:
        set_mic_muted(container.audio_source, True)
    launcher = _audio_launcher(
        config, template, out_dir, container.model_resolver_factory, container.transcriber, container.audio_source
    )
    session = HeadlessSession(
        container.controller, out_dir, launcher, activity='recording', audio_source=container.audio_source
    )
    _emit_banner('recording', out_dir, session.status.pid, show_mute=True)
    with keep_awake():
        _run(out_dir, config, session)


def run_transcribe_headless(
    ctx: click.Context,
    *,
    audio_path: Path,
    template_name: str | None = None,
    language: str | None = None,
    label: str | None = None,
) -> None:
    """Transcribe an audio file headlessly (no TUI)."""
    config, infra, template_loader = load_config(ctx.obj['config_path'], ctx.obj['output_dir'])
    template = _select_headless_template(template_loader, template_name, language)

    out_dir = make_session_dir(resolve_base_dir(ctx.obj['output_dir'], config), label)
    _warn_missing_models(*preflight_llm(infra, config))

    container = _build_container(config, template, out_dir, infra, build_audio=False)
    launcher = _file_launcher(config, template, audio_path, container.model_resolver_factory, container.transcriber)
    session = HeadlessSession(container.controller, out_dir, launcher, activity='transcribing')
    _emit_banner('transcribing', out_dir, session.status.pid)
    _run(out_dir, config, session)
