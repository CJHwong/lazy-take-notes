"""Gateway: CoreAudioTapSource — reads system audio via the native coreaudio-tap subprocess."""

from __future__ import annotations

import logging
import queue
import subprocess  # noqa: S404 -- intentional: fixed arg list, not shell=True
import sys
import threading
from pathlib import Path

import numpy as np

from lazy_take_notes.l1_entities.audio_constants import SAMPLE_RATE

log = logging.getLogger('ltn.audio.coreaudio')

_BINARY = Path(__file__).parent.parent.parent / '_native' / 'bin' / 'coreaudio-tap'
_BYTES_PER_SAMPLE = 4  # float32
_CHUNK_FRAMES = SAMPLE_RATE // 10  # 100ms chunks


class CoreAudioTapSource:
    """Implements AudioSource by reading float32 PCM from the Swift coreaudio-tap subprocess."""

    def __init__(self) -> None:
        self._proc: subprocess.Popen | None = None
        self._queue: queue.Queue[np.ndarray] = queue.Queue()
        self._stop = threading.Event()
        self._exhausted = threading.Event()
        self._thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._error: RuntimeError | None = None

    @property
    def exhausted(self) -> bool:
        """True when the audio stream ended unexpectedly (EOF without explicit stop)."""
        return self._exhausted.is_set()

    def open(self, sample_rate: int, channels: int) -> None:
        if sys.platform != 'darwin':
            raise RuntimeError('CoreAudioTapSource is macOS only')
        if not _BINARY.exists():
            raise RuntimeError(f'Native binary not found: {_BINARY}\nRun: bash scripts/build_native.sh')
        self._error = None
        self._stop.clear()
        self._exhausted.clear()
        self._proc = subprocess.Popen(  # noqa: S603 -- fixed arg list, not shell=True
            [str(_BINARY)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.PIPE,
        )
        log.info('coreaudio-tap started (pid=%d, binary=%s)', self._proc.pid, _BINARY)
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()
        self._stderr_thread = threading.Thread(target=self._stderr_reader, daemon=True)
        self._stderr_thread.start()

    def _stderr_reader(self) -> None:
        """Stream stderr from the Swift binary so diagnostic lines appear in the debug log."""
        proc = self._proc
        if proc is None or proc.stderr is None:  # pragma: no cover
            return
        for raw_line in proc.stderr:
            line = raw_line.decode('utf-8', errors='replace').rstrip()
            if line:
                log.info('coreaudio-tap: %s', line)

    def _reader(self) -> None:
        log.debug('stdout reader thread started')
        chunk_bytes = _CHUNK_FRAMES * _BYTES_PER_SAMPLE
        proc = self._proc
        if proc is None or proc.stdout is None:  # pragma: no cover -- guard for ScreenCaptureKit process state
            return
        while not self._stop.is_set():
            raw = proc.stdout.read(chunk_bytes)
            if not raw:
                break
            self._queue.put(np.frombuffer(raw, dtype=np.float32).copy())
        if not self._stop.is_set():
            log.warning('coreaudio-tap stdout EOF (binary stopped writing)')
            self._exhausted.set()
        # Detect abnormal exit so read() can surface it instead of silently returning None.
        if not self._stop.is_set() and proc.poll() != 0:
            rc = proc.wait()
            stderr_text = ''
            if proc.stderr is not None:
                try:
                    stderr_text = proc.stderr.read().decode('utf-8', errors='replace').strip()
                except Exception:  # noqa: BLE001, S110
                    pass
            msg = f'coreaudio-tap exited with code {rc}'
            if stderr_text:
                msg += f': {stderr_text}'
            log.error(msg)
            self._error = RuntimeError(msg)

    def read(self, timeout: float = 0.1) -> np.ndarray | None:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            if self._error is not None:
                raise self._error from None
            return None

    def close(self) -> None:
        self._stop.set()
        if self._proc is not None:
            log.debug('closing coreaudio-tap (pid=%d)', self._proc.pid)
            self._proc.terminate()
            try:
                self._proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None
        if self._stderr_thread is not None:
            self._stderr_thread.join(timeout=2)
            self._stderr_thread = None
