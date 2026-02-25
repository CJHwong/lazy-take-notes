"""Gateway: SoundCardLoopbackSource — reads system audio via PulseAudio/WASAPI loopback."""

from __future__ import annotations

import queue
import sys
import threading

import numpy as np
import soundcard as sc  # noqa: PLC0415 -- top-level import for L3 adapter; forbidden in L1/L2 by import-linter

from lazy_take_notes.l1_entities.audio_constants import SAMPLE_RATE


def _patch_soundcard_numpy2_compat() -> None:
    """Monkey-patch soundcard 0.4.5 for numpy 2.x on Windows.

    mediafoundation.py calls numpy.fromstring(buf, dtype='float32') which was
    removed in numpy 2.0.  Upstream fix exists on master but is unreleased.
    This shim redirects the call to numpy.frombuffer.

    Remove when soundcard ships a PyPI release with the fix (>0.4.5).
    """
    if sys.platform != 'win32':
        return
    try:
        from soundcard import mediafoundation as _mf  # noqa: PLC0415 -- deferred: Windows only
    except ImportError:
        return

    _real_numpy = _mf.numpy

    class _Numpy2CompatShim:
        __slots__ = ()

        def __getattr__(self, name: str) -> object:
            if name == 'fromstring':
                return _real_numpy.frombuffer
            return getattr(_real_numpy, name)

    _mf.numpy = _Numpy2CompatShim()


_patch_soundcard_numpy2_compat()

_CHUNK_FRAMES = SAMPLE_RATE // 10  # 100ms chunks — matches CoreAudioTapSource cadence


def _win_com_init() -> None:
    """Initialize COM on the current thread for WASAPI access (Windows-only, no-op elsewhere)."""
    if sys.platform == 'win32':
        import ctypes  # noqa: PLC0415 -- Windows-only stdlib import

        ctypes.windll.ole32.CoInitializeEx(None, 0x0)  # COINIT_MULTITHREADED


def _win_com_uninit() -> None:
    """Uninitialize COM on the current thread (Windows-only, no-op elsewhere)."""
    if sys.platform == 'win32':
        import ctypes  # noqa: PLC0415 -- Windows-only stdlib import

        ctypes.windll.ole32.CoUninitialize()


class SoundCardLoopbackSource:
    """Implements AudioSource using soundcard library loopback capture (Linux/Windows)."""

    def __init__(self) -> None:
        self._queue: queue.Queue[np.ndarray] = queue.Queue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._recorder = None
        self._com_owner = False  # tracks whether we initialized COM on the caller thread

    def open(self, sample_rate: int, channels: int) -> None:
        if sys.platform == 'darwin':
            raise RuntimeError('SoundCardLoopbackSource is not supported on macOS — use CoreAudioTapSource')

        _win_com_init()
        self._com_owner = sys.platform == 'win32'

        loopback = self._find_loopback()
        self._stop.clear()
        self._recorder = loopback.recorder(samplerate=sample_rate, channels=channels)
        self._recorder.__enter__()  # noqa: PLC2801 -- manual context: open/close are separate methods
        self._thread = threading.Thread(
            target=self._reader,
            args=(self._recorder, sample_rate),
            daemon=True,
        )
        self._thread.start()

    @staticmethod
    def _find_loopback():
        """Find a loopback device matching the default speaker.

        On Windows/Linux, WASAPI/PulseAudio loopback captures from a specific
        output device.  We match the default speaker so the user hears audio
        from the same device we're capturing.  Falls back to first loopback
        if no match (e.g. PulseAudio monitor naming differs).
        """
        mics = sc.all_microphones(include_loopback=True)
        loopbacks = [m for m in mics if m.isloopback]
        if not loopbacks:
            raise RuntimeError(
                'No loopback audio device found. Ensure PulseAudio/PipeWire (Linux) or WASAPI (Windows) is running.'
            )

        default_speaker = sc.default_speaker()
        if default_speaker is not None:
            for mic in loopbacks:
                if mic.id == default_speaker.id:
                    return mic

        return loopbacks[0]

    def _reader(self, recorder, sample_rate: int) -> None:
        _win_com_init()
        try:
            chunk_frames = sample_rate // 10  # 100ms
            while not self._stop.is_set():
                data = recorder.record(numframes=chunk_frames)
                if data is None:
                    continue
                # soundcard returns (frames, channels) — flatten to mono float32
                if data.ndim > 1:
                    data = data.mean(axis=1)
                self._queue.put(data.astype(np.float32))
        finally:
            _win_com_uninit()

    def read(self, timeout: float = 0.1) -> np.ndarray | None:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def close(self) -> None:
        self._stop.set()
        # Join the reader thread BEFORE destroying the recorder — otherwise
        # the reader may still be inside recorder.record() when PulseAudio
        # tears down the stream, causing a pa_stream reference count assertion.
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None
        if self._recorder is not None:
            try:
                self._recorder.__exit__(None, None, None)  # noqa: PLC2801 -- manual context: open/close are separate methods
            except Exception:  # noqa: BLE001, S110
                pass
            self._recorder = None
        if self._com_owner:
            _win_com_uninit()
            self._com_owner = False
