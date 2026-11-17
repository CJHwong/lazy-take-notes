"""Gateway: whisper transcriber in a subprocess — avoids GIL contention."""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
from multiprocessing.connection import Connection
from typing import Any

import numpy as np

from lazy_take_notes.l1_entities.transcript import TranscriptSegment

log = logging.getLogger('ltn.whisper')


class _ChildDiedError(RuntimeError):
    """The whisper child exited or its pipe broke before it answered."""


def _subprocess_entry(model_path: str, conn: Any) -> None:
    """Subprocess main: load model via WhisperTranscriber, loop on requests.

    Permanently redirects C-level stdout/stderr to /dev/null so whisper.cpp's
    fprintf() calls do not escape to the parent TUI.
    """
    # Detach into our own session so a terminal Ctrl-C (SIGINT to the foreground
    # process group) can't kill this worker mid-transcribe in headless mode. The
    # parent owns our shutdown via the None sentinel / terminate().
    try:
        os.setsid()
    except OSError:  # pragma: no cover -- already a session leader (rare for spawn children)
        pass

    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)
    os.close(devnull)

    try:
        from lazy_take_notes.l3_interface_adapters.gateways.whisper_transcriber import (  # noqa: PLC0415 -- deferred: subprocess only
            WhisperTranscriber,
        )

        transcriber = WhisperTranscriber()
        transcriber.load_model(model_path)
    except Exception as e:
        conn.send({'status': 'error', 'error': str(e)})
        conn.close()
        return

    conn.send({'status': 'ready'})

    while True:
        req = conn.recv()
        if req is None:
            break
        try:
            segments = transcriber.transcribe(
                audio=req['audio'],
                language=req['language'],
                hints=req.get('hints', []),
            )
            conn.send({'status': 'ok', 'segments': segments})
        except Exception as e:
            conn.send({'status': 'error', 'error': str(e)})

    transcriber.close()
    conn.close()


class SubprocessWhisperTranscriber:
    """Whisper transcriber that runs inference in a child process.

    whisper.cpp's C extension holds the Python GIL for the full duration of
    inference. Running it in a subprocess means the parent process's asyncio
    event loop (and the Textual TUI) stays responsive during transcription.

    Uses multiprocessing.Pipe (raw socket pair) instead of Queue to avoid
    the resource tracker, which fails when Textual has replaced sys.stderr
    with a stream that returns an invalid fileno().
    """

    def __init__(self) -> None:
        self._process: Any = None  # SpawnProcess; typed as Any — context returns a subclass
        self._conn: Connection | None = None
        self._model_path: str | None = None

    def load_model(self, model_path: str) -> None:
        if self._model_ready(model_path):
            log.info('whisper model already loaded: %s', model_path)
            return
        self._kill()
        self._model_path = model_path
        ctx = mp.get_context('spawn')
        parent_conn, child_conn = ctx.Pipe(duplex=True)
        self._process = ctx.Process(
            target=_subprocess_entry,
            args=(model_path, child_conn),
            daemon=True,
        )
        self._process.start()
        child_conn.close()  # parent only needs its own end
        self._conn = parent_conn

        try:
            if not self._conn.poll(timeout=120):
                raise RuntimeError('Timeout waiting for model load')
            result = self._conn.recv()
        except EOFError as e:
            raise RuntimeError('Whisper subprocess exited unexpectedly during model load') from e

        if result.get('status') != 'ready':
            raise RuntimeError(f'Whisper subprocess failed to init: {result.get("error", "unknown")}')

    def transcribe(
        self,
        audio: np.ndarray,
        language: str,
        hints: list[str] | None = None,
    ) -> list[TranscriptSegment]:
        try:
            return self._transcribe_once(audio, language, hints)
        except _ChildDiedError as e:
            # The child was gone before it answered, so no late reply can exist.
            # Retry the same chunk once on a fresh child instead of losing it.
            log.warning('%s; retrying the chunk on a fresh subprocess', e)
            return self._transcribe_once(audio, language, hints)

    def _transcribe_once(
        self,
        audio: np.ndarray,
        language: str,
        hints: list[str] | None,
    ) -> list[TranscriptSegment]:
        if self._conn is None:
            if self._model_path is None:
                raise RuntimeError('Model not loaded. Call load_model() first.')
            self._respawn()
        conn = self._conn
        assert conn is not None  # noqa: S101 -- narrowed for the type checker; _respawn sets it or raises
        # Any failure leaves the pipe in an unknown state: a late reply would be
        # read as the answer to the next chunk. Kill the child; the next call
        # starts a fresh one.
        try:
            conn.send({'audio': audio, 'language': language, 'hints': hints or []})
            if not conn.poll(timeout=120):
                self._kill()
                raise RuntimeError('Timeout waiting for transcription result')
            result = conn.recv()
        except EOFError as e:
            self._kill()
            raise _ChildDiedError('Whisper subprocess exited unexpectedly during transcription') from e
        except OSError as e:
            self._kill()
            raise _ChildDiedError(f'Whisper subprocess pipe failed: {e}') from e

        if result.get('status') == 'error':
            raise RuntimeError(result['error'])
        return result.get('segments', [])

    def _model_ready(self, model_path: str) -> bool:
        """Preflight loads the model first; the session reuses that child."""
        return (
            model_path == self._model_path
            and self._conn is not None
            and self._process is not None
            and self._process.is_alive()
        )

    def _respawn(self) -> None:
        log.warning('Restarting the whisper subprocess')
        try:
            self.load_model(self._model_path or '')
        except Exception:
            self._kill()
            raise

    def _kill(self) -> None:
        conn, self._conn = self._conn, None
        process, self._process = self._process, None
        if conn is not None:
            try:
                conn.close()
            except OSError:
                log.debug('closing a failed whisper pipe', exc_info=True)
        if process is not None:
            process.terminate()
            process.join(timeout=1)

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.send(None)
            except Exception:  # noqa: S110 — best-effort shutdown signal; pipe may already be closed
                pass
            try:
                self._conn.close()
            except Exception:  # noqa: S110 — best-effort; ignore double-close
                pass
            self._conn = None
        if self._process is not None:
            self._process.join(timeout=5)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=1)  # reap zombie after SIGTERM
            self._process = None
