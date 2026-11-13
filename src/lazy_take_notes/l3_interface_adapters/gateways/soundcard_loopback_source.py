"""Gateway: SoundCardLoopbackSource — reads system audio via PulseAudio/WASAPI loopback."""

from __future__ import annotations

import logging
import queue
import sys
import threading

import numpy as np
import soundcard as sc  # noqa: PLC0415 -- top-level import for L3 adapter; forbidden in L1/L2 by import-linter

from lazy_take_notes.l1_entities.audio_constants import SAMPLE_RATE
from lazy_take_notes.l3_interface_adapters.gateways._audio_buffer import AudioBuffer

log = logging.getLogger('ltn.audio.loopback')


def _patch_soundcard_numpy2_compat() -> None:
    """Monkey-patch soundcard 0.4.5 for numpy 2.x on Windows.

    mediafoundation.py calls numpy.fromstring(buf, dtype='float32') which was
    removed in numpy 2.0.  Upstream fix (fromstring → frombuffer + copy) exists
    on master but is unreleased.  This shim intercepts fromstring and replaces it
    with frombuffer(...).copy() — the copy is critical because frombuffer returns
    a view into the WASAPI capture buffer which is freed after _capture_release().

    Remove when soundcard ships a PyPI release with the fix (>0.4.5).
    """
    if sys.platform != 'win32':
        return
    try:
        from soundcard import mediafoundation as _mf  # noqa: PLC0415 -- deferred: Windows only
    except ImportError:
        return

    _real_numpy = _mf.numpy

    def _fromstring_compat(*args, **kwargs):
        """fromstring always copied; frombuffer returns a view — copy to match."""
        return _real_numpy.frombuffer(*args, **kwargs).copy()

    class _Numpy2CompatShim:
        __slots__ = ()

        def __getattr__(self, name: str) -> object:
            if name == 'fromstring':
                return _fromstring_compat
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

    continuous_capture = False

    def __init__(self) -> None:
        self._queue = AudioBuffer(max_frames=SAMPLE_RATE, sample_rate=SAMPLE_RATE, label='loopback')
        self._error: RuntimeError | None = None
        self._exhausted = threading.Event()
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread: threading.Thread | None = None
        self._recorder = None
        self._com_owner = False
        self.mic_muted: bool = False  # not used directly; MixedAudioSource handles muting

    @property
    def exhausted(self) -> bool:
        return self._exhausted.is_set()

    @property
    def dropped_frames(self) -> int:
        return self._queue.dropped_frames

    def open(self, sample_rate: int, channels: int) -> None:
        if sys.platform == 'darwin':
            raise RuntimeError('SoundCardLoopbackSource is not supported on macOS; use CoreAudioTapSource')

        self.close()
        self._error = None
        self._exhausted.clear()
        self._queue = AudioBuffer(max_frames=sample_rate, sample_rate=sample_rate, label='loopback')
        self._stop.clear()
        self._ready.clear()
        self._thread = threading.Thread(target=self._reader, args=(None, sample_rate, channels), daemon=True)
        try:
            self._thread.start()
        except Exception:
            self._thread = None
            raise
        if not self._ready.wait(timeout=5):
            self._stop.set()
            raise RuntimeError('Loopback capture startup timed out; worker still owns its resources')
        if self._error is not None:
            self.close()
            raise self._error from None
        if self._stop.is_set():
            raise RuntimeError('Loopback capture stopped during startup')

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

    def _reader(self, recorder, sample_rate: int, channels: int = 1) -> None:
        try:
            _win_com_init()
            self._com_owner = sys.platform == 'win32'
            if recorder is None:
                loopback = self._find_loopback()
                # PulseAudio defaults can batch seconds of samples and delay shutdown.
                blocksize = sample_rate // 10 if sys.platform == 'linux' else None
                recorder = loopback.recorder(samplerate=sample_rate, channels=channels, blocksize=blocksize)
                self._recorder = recorder
                recorder.__enter__()  # noqa: PLC2801 -- worker owns complete native lifetime
            if sys.platform != 'linux':
                self._ready.set()
            self._read_chunks(recorder, sample_rate)
        except Exception as error:
            if not self._stop.is_set():
                self._error = RuntimeError(f'Loopback capture failed: {error}')
                self._exhausted.set()
                log.exception('loopback capture stopped')
        finally:
            self._ready.set()
            self._close_recorder()
            self._release_com()

    def _read_chunks(self, recorder, sample_rate: int) -> None:
        while not self._stop.is_set():
            samples = (
                self._read_available_pulse(recorder)
                if sys.platform == 'linux'
                else recorder.record(numframes=sample_rate // 10)
            )
            if samples is None or not len(samples):
                self._stop.wait(0.01)
                continue
            if self._stop.is_set():
                continue
            if samples.ndim > 1:
                samples = samples.mean(axis=1)
            self._queue.put(samples.astype(np.float32))
            self._ready.set()

    @staticmethod
    def _read_available_pulse(recorder) -> np.ndarray | None:
        # SoundCard 0.4.5 blocks even with numframes=None when the stream is empty.
        # This worker is the only consumer, so a ready packet stays available.
        from soundcard import pulseaudio  # noqa: PLC0415 -- pinned Linux backend exposes readiness

        stream_state = pulseaudio._pulse._pa_stream_get_state(recorder.stream)
        if stream_state in (pulseaudio._pa.PA_STREAM_FAILED, pulseaudio._pa.PA_STREAM_TERMINATED):
            raise RuntimeError(f'PulseAudio stream stopped (state {stream_state})')
        readable_bytes = pulseaudio._pulse._pa_stream_readable_size(recorder.stream)
        if readable_bytes == int(pulseaudio._ffi.cast('size_t', -1)):
            raise RuntimeError('PulseAudio readable size failed')
        if readable_bytes == 0:
            return None
        return recorder.record(numframes=None)

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
        if self._thread is not None:
            self._thread.join(timeout=2)
            if self._thread.is_alive():
                raise RuntimeError('Loopback reader did not stop; recorder remains open to prevent unsafe teardown')
            self._thread = None
        self.drain()

    def _close_recorder(self) -> None:
        recorder = self._recorder
        self._recorder = None
        if recorder is not None:
            try:
                recorder.__exit__(None, None, None)  # noqa: PLC2801 -- port separates open and close
            except Exception:
                log.exception('failed to close loopback recorder')

    def _release_com(self) -> None:
        if self._com_owner:
            _win_com_uninit()
            self._com_owner = False
