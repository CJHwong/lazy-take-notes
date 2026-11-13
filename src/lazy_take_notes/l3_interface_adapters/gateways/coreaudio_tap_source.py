"""Read system audio from the native Core Audio process tap."""

from __future__ import annotations

import logging
import os
import queue
import subprocess  # noqa: S404 -- fixed native executable, no shell
import sys
import threading
from collections import deque
from pathlib import Path

import numpy as np

from lazy_take_notes.l1_entities.audio_constants import SAMPLE_RATE
from lazy_take_notes.l3_interface_adapters.gateways._audio_buffer import AudioBuffer

log = logging.getLogger('ltn.audio.coreaudio')

_BINARY = Path(__file__).parent.parent.parent / '_native' / 'bin' / 'coreaudio-tap'
_BYTES_PER_SAMPLE = 4
_CHUNK_FRAMES = SAMPLE_RATE // 10


class CoreAudioTapSource:
    """Keep raw float32 output separate from helper diagnostics and lifetime."""

    continuous_capture = True

    def __init__(self, *, ensure_permission: bool = False) -> None:
        self._proc: subprocess.Popen | None = None
        self._ensure_permission = ensure_permission
        self._queue = AudioBuffer(max_frames=SAMPLE_RATE, sample_rate=SAMPLE_RATE, label='system tap')
        self._stop = threading.Event()
        self._exhausted = threading.Event()
        self._stderr_done = threading.Event()
        self._stderr_tail: deque[str] = deque(maxlen=20)
        self._stderr_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._error: RuntimeError | None = None
        self._signal_seen = False
        self._awaiting_permission = False
        self.mic_muted = False

    @property
    def exhausted(self) -> bool:
        return self._exhausted.is_set()

    @property
    def dropped_frames(self) -> int:
        return self._queue.dropped_frames

    @property
    def system_signal_seen(self) -> bool:
        """Report observed signal; zeros alone cannot identify permission denial."""
        return self._signal_seen

    @property
    def awaiting_permission(self) -> bool:
        return self._awaiting_permission

    def open(self, sample_rate: int, channels: int) -> None:
        if sys.platform != 'darwin':
            raise RuntimeError('CoreAudioTapSource is macOS only')
        if sample_rate != SAMPLE_RATE or channels != 1:
            raise ValueError('CoreAudioTapSource requires 16000 Hz mono output')
        if not _BINARY.exists():
            raise RuntimeError(f'Native binary not found: {_BINARY}\nRun: bash scripts/build_native.sh')
        self.close()
        self._error = None
        self._signal_seen = False
        self._awaiting_permission = False
        self._stop.clear()
        self._exhausted.clear()
        self._stderr_done.clear()
        self._stderr_tail.clear()
        self._queue = AudioBuffer(max_frames=SAMPLE_RATE, sample_rate=SAMPLE_RATE, label='system tap')
        arguments = [str(_BINARY)]
        if self._ensure_permission or os.environ.get('LTN_KIOSK') == '1':
            arguments.append('--ensure-permission')
        self._proc = subprocess.Popen(  # noqa: S603 -- fixed argument list, no shell
            arguments,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.PIPE,
            start_new_session=True,
        )
        log.info('coreaudio-tap started (pid=%d, binary=%s)', self._proc.pid, _BINARY)
        self._stderr_thread = threading.Thread(target=self._stderr_reader, daemon=True)
        self._thread = threading.Thread(target=self._reader, daemon=True)
        try:
            self._stderr_thread.start()
            self._thread.start()
        except Exception:
            self.close()
            raise

    def _stderr_reader(self) -> None:
        process = self._proc
        if process is None or process.stderr is None:
            self._stderr_done.set()
            return
        try:
            for raw_line in process.stderr:
                self._record_diagnostic(raw_line)
        except (OSError, ValueError):
            if not self._stop.is_set():
                log.exception('coreaudio-tap diagnostic pipe failed')
        finally:
            self._stderr_done.set()

    def _record_diagnostic(self, raw_line: bytes) -> None:
        line = raw_line.decode('utf-8', errors='replace').rstrip()
        if line:
            with self._stderr_lock:
                self._stderr_tail.append(line)
            log.info('coreaudio-tap: %s', line)
        if 'waiting for permission' in line:
            self._awaiting_permission = True
        elif 'permission granted' in line or line.endswith('capturing'):
            self._awaiting_permission = False

    def _reader(self) -> None:
        process = self._proc
        if process is None or process.stdout is None:
            return
        remainder = b''
        try:
            while not self._stop.is_set():
                raw = process.stdout.read(_CHUNK_FRAMES * _BYTES_PER_SAMPLE)
                if not raw:
                    break
                combined = remainder + raw
                usable = len(combined) - len(combined) % _BYTES_PER_SAMPLE
                remainder = combined[usable:]
                self._enqueue_bytes(combined[:usable])
            if remainder and not self._stop.is_set():
                self._error = RuntimeError('coreaudio-tap ended with a partial float32 sample')
        except (OSError, ValueError) as error:
            if not self._stop.is_set():
                self._error = RuntimeError(f'coreaudio-tap audio pipe failed: {error}')
        if not self._stop.is_set():
            self._record_exit(process)

    def _enqueue_bytes(self, raw: bytes) -> None:
        if not raw:
            return
        samples = np.frombuffer(raw, dtype=np.float32).copy()
        if not self._signal_seen and samples.any():
            self._signal_seen = True
        self._queue.put(samples)

    def _record_exit(self, process: subprocess.Popen) -> None:
        return_code = process.poll()
        if return_code is None:
            try:
                return_code = process.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                self._error = RuntimeError('coreaudio-tap closed audio output while still running')
        if return_code not in (None, 0):
            self._stderr_done.wait(timeout=0.2)
            with self._stderr_lock:
                diagnostic = '\n'.join(self._stderr_tail)
            message = f'coreaudio-tap exited with code {return_code}'
            self._error = RuntimeError(f'{message}: {diagnostic}' if diagnostic else message)
        self._exhausted.set()
        log.warning('coreaudio-tap audio output ended%s', f': {self._error}' if self._error else '')

    def read(self, timeout: float = 0.1) -> np.ndarray | None:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            if self._error is not None:
                raise self._error from None
            return None

    def drain(self) -> None:
        self._queue.clear()

    def close(self) -> None:
        self._stop.set()
        process = self._proc
        if process is not None:
            self._stop_process(process)
        self._join_reader(self._thread)
        self._join_reader(self._stderr_thread)
        if process is not None:
            self._close_pipes(process)
        self._proc = None
        self._thread = None
        self._stderr_thread = None
        self.drain()

    @staticmethod
    def _stop_process(process: subprocess.Popen) -> None:
        if process.stdin is not None:
            try:
                process.stdin.close()
            except (OSError, ValueError):
                log.debug('coreaudio-tap stdin already closed', exc_info=True)
        try:
            process.wait(timeout=1)
            return
        except subprocess.TimeoutExpired:
            log.warning('coreaudio-tap did not stop after stdin closed; terminating')
        try:
            process.terminate()
        except ProcessLookupError:
            log.debug('coreaudio-tap already exited')
        try:
            process.wait(timeout=1)
            return
        except subprocess.TimeoutExpired:
            log.warning('coreaudio-tap ignored termination; killing')
        try:
            process.kill()
        except ProcessLookupError:
            log.debug('coreaudio-tap exited before kill')
        process.wait(timeout=1)

    @staticmethod
    def _join_reader(reader: threading.Thread | None) -> None:
        if reader is not None and reader.ident is not None:
            reader.join(timeout=2)
            if reader.is_alive():
                raise RuntimeError('coreaudio-tap reader did not stop after helper exit')

    @staticmethod
    def _close_pipes(process: subprocess.Popen) -> None:
        for pipe in (process.stdin, process.stdout, process.stderr):
            if pipe is not None:
                pipe.close()
