"""Gateway: sounddevice audio source — implements AudioSource port."""

from __future__ import annotations

import logging
import queue

import numpy as np
import sounddevice as sd

from lazy_take_notes.l1_entities.audio_constants import SAMPLE_RATE

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

    def __init__(self) -> None:
        self._stream: sd.InputStream | None = None
        self._queue: queue.Queue[np.ndarray] = queue.Queue()
        self.mic_muted: bool = False  # not used directly; MixedAudioSource handles muting

    def open(self, sample_rate: int = SAMPLE_RATE, channels: int = 1) -> None:
        device_info = sd.query_devices(kind='input')
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

        def _callback(indata, frames, time_info, status):
            if status:
                log.warning('PortAudio status: %s', status)
            if ratio == 1:
                self._queue.put(indata.copy())
                return
            # Box-filter decimate: average each consecutive `ratio`-sample group
            # and emit one output sample. Voice content (≤ 4 kHz) sits well
            # inside the boxcar's main lobe; high-frequency mic noise above the
            # output Nyquist (8 kHz) is attenuated by the filter notches.
            n_out = frames // ratio
            if n_out == 0:
                return
            usable = indata[: n_out * ratio]
            downsampled = usable.reshape(n_out, ratio, -1).mean(axis=1).astype(np.float32)
            self._queue.put(downsampled.copy())

        self._stream = sd.InputStream(
            samplerate=stream_sr,
            blocksize=blocksize,
            channels=channels,
            dtype='float32',
            callback=_callback,
        )
        self._stream.start()

    def read(self, timeout: float = 0.1) -> np.ndarray | None:
        try:
            return self._queue.get(timeout=timeout).flatten()
        except queue.Empty:
            return None

    def drain(self) -> None:
        """Discard all buffered audio — called on pause so resume starts fresh."""
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def close(self) -> None:
        if self._stream is not None:
            log.debug('closing sounddevice stream')
            self._stream.stop()
            self._stream.close()
            self._stream = None
