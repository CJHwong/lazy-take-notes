"""Gateway: whisper transcriber in a subprocess — avoids GIL contention."""

from __future__ import annotations

import multiprocessing as mp
import os
from multiprocessing.connection import Connection
from typing import Any

import numpy as np

from lazy_take_notes.l1_entities.transcript import TranscriptSegment


def _subprocess_entry(model_path: str, conn: Any) -> None:
    """Subprocess main: load model via WhisperTranscriber, loop on requests.

    Permanently redirects C-level stdout/stderr to /dev/null so whisper.cpp's
    fprintf() calls do not escape to the parent TUI.
    """
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
                initial_prompt=req.get('prompt', ''),
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

    def load_model(self, model_path: str) -> None:
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
        initial_prompt: str = '',
    ) -> list[TranscriptSegment]:
        if self._conn is None:
            raise RuntimeError('Model not loaded. Call load_model() first.')
        self._conn.send({'audio': audio, 'language': language, 'prompt': initial_prompt})
        try:
            if not self._conn.poll(timeout=120):
                raise RuntimeError('Timeout waiting for transcription result')
            result = self._conn.recv()
        except EOFError as e:
            raise RuntimeError('Whisper subprocess exited unexpectedly during transcription') from e

        if result.get('status') == 'error':
            raise RuntimeError(result['error'])
        return result.get('segments', [])

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
