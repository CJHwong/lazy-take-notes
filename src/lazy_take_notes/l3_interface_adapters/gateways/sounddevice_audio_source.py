"""Gateway: sounddevice audio source — implements AudioSource port."""

from __future__ import annotations

import logging
import queue

import numpy as np
import sounddevice as sd

from lazy_take_notes.l1_entities.audio_constants import SAMPLE_RATE
from lazy_take_notes.l3_interface_adapters.gateways._audio_buffer import AudioBuffer

log = logging.getLogger('ltn.audio.sounddevice')


class SounddeviceAudioSource:
    """Wraps sounddevice.InputStream to provide audio chunks at SAMPLE_RATE.

    Opens the device at its native rate and decimates to SAMPLE_RATE in Python
    when the native rate is an integer multiple of the target. Relying on
    PortAudio's internal resampling on macOS CoreAudio at common ratios (96 kHz
    → 16 kHz) silently under-delivers samples by ~5 % — a 3-minute recording
    consumes ~171 s of mic audio against 180 s of wall time, and the missing
    mic samples leave the MixedAudioSource's sys_buf accumulating system audio
    indefinitely (9 s of lag at 3 min, growing linearly with session length).

    Falls back to PortAudio resampling when the native rate isn't a clean
    integer multiple of SAMPLE_RATE (e.g. 44.1 kHz devices).
    """

    continuous_capture = True

    def __init__(self, device: str | None = None) -> None:
        self._device = device
        self._stream: sd.InputStream | None = None
        self._queue = AudioBuffer(max_frames=SAMPLE_RATE, sample_rate=SAMPLE_RATE, label='microphone')
        self._closing = True
        self._generation = 0
        self._error: RuntimeError | None = None
        self._exhausted = False
        self.mic_muted: bool = False  # not used directly; MixedAudioSource handles muting

    @property
    def exhausted(self) -> bool:
        return self._exhausted

    @property
    def dropped_frames(self) -> int:
        return self._queue.dropped_frames

    @staticmethod
    def list_input_devices() -> list[str]:
        """Return the names of devices with at least one input channel."""
        devices = sd.query_devices()
        names = [device['name'] for device in devices if device['max_input_channels'] > 0]
        return list(dict.fromkeys(names))

    def _input_info(self):
        try:
            return sd.query_devices(kind='input', device=self._device)
        except (ValueError, sd.PortAudioError):
            log.warning('input device %r is unavailable, using the default', self._device)
            self._device = None
            return sd.query_devices(kind='input')

    def open(self, sample_rate: int = SAMPLE_RATE, channels: int = 1) -> None:
        self.close()
        self._error = None
        self._exhausted = False
        self._queue = AudioBuffer(max_frames=sample_rate, sample_rate=sample_rate, label='microphone')
        generation = self._generation
        device_info = self._input_info()
        native_sr_raw = device_info.get('default_samplerate', float(sample_rate))
        native_sr = int(round(native_sr_raw))

        # Decimate in Python when native is an integer multiple of target rate
        # (96 kHz → 16 kHz is ratio 6; 48 kHz → 16 kHz is ratio 3). Otherwise
        # fall back to PortAudio's resampling and accept the drift.
        if native_sr >= sample_rate and native_sr % sample_rate == 0 and native_sr != sample_rate:
            stream_sr = native_sr
            ratio = native_sr // sample_rate
            # Lock the callback's frame count to a multiple of `ratio` so each
            # callback decimates cleanly with no leftover samples. ~32 ms per
            # callback at the target rate matches the previous default behaviour.
            blocksize = ratio * 512
        else:
            stream_sr = sample_rate
            ratio = 1
            blocksize = 0  # let PortAudio pick

        log.info(
            'opening sounddevice: device=%s, native_sr=%s, stream_sr=%d, decimation_ratio=%d, '
            'blocksize=%d, channels=%d',
            device_info.get('name', '?'),
            native_sr_raw,
            stream_sr,
            ratio,
            blocksize,
            channels,
        )

        def _callback(input_samples, frames, time_info, status):
            if generation != self._generation or self._closing:
                return
            try:
                self._capture(input_samples, frames, ratio, status)
            except Exception as error:
                self._error = RuntimeError(f'Microphone callback failed: {error}')
                self._exhausted = True
                raise sd.CallbackAbort from error

        def _finished():
            if generation == self._generation and not self._closing:
                if self._error is None:
                    self._error = RuntimeError('Microphone stream stopped unexpectedly')
                self._exhausted = True

        self._stream = sd.InputStream(
            device=self._device,
            samplerate=stream_sr,
            blocksize=blocksize,
            channels=channels,
            dtype='float32',
            callback=_callback,
            finished_callback=_finished,
        )
        self._closing = False
        try:
            self._stream.start()
        except Exception:
            self._closing = True
            try:
                self._stream.close()
            except Exception:
                log.exception('failed to close microphone after startup failure')
            self._stream = None
            raise

    def _capture(self, input_samples: np.ndarray, frames: int, ratio: int, status: object) -> None:
        if status:
            log.warning('PortAudio status: %s', status)
        if ratio == 1:
            self._queue.put(input_samples.copy())
            return
        output_frames = frames // ratio
        if output_frames == 0:
            return
        usable = input_samples[: output_frames * ratio]
        downsampled = usable.reshape(output_frames, ratio, -1).mean(axis=1).astype(np.float32)
        self._queue.put(downsampled)

    def read(self, timeout: float = 0.1) -> np.ndarray | None:
        try:
            return self._queue.get(timeout=timeout).flatten()
        except queue.Empty:
            if self._error is not None:
                raise self._error from None
            return None

    def drain(self) -> None:
        self._queue.clear()

    def close(self) -> None:
        self._closing = True
        self._generation += 1
        stream = self._stream
        if stream is None:
            return
        try:
            stream.stop()
        finally:
            try:
                stream.close()
            finally:
                self._stream = None
                self.drain()
