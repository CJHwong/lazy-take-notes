"""Gateway: MixedAudioSource — composites microphone and system audio into one stream."""

from __future__ import annotations

import logging
import math
import queue
import threading
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from lazy_take_notes.l2_use_cases.ports.audio_source import AudioSource

log = logging.getLogger('ltn.audio.mixed')


class MixedAudioSource:
    """Mix microphone and system audio from two AudioSource instances.

    Chunk sizes may differ between sources: sounddevice fires at device block size (~512
    samples / 32 ms) while system capture sources produce fixed 1600-sample (100 ms)
    chunks. To handle this, _sys_buf accumulates all available system chunks on every
    read() call (non-blocking drain) and dispenses exactly len(mic_chunk) samples per
    mix — equal duration, not equal chunk count. Mic chunk timing drives output cadence
    because mic data arrives more frequently; both sources contribute equally to the
    final audio.

    Output is peak-limited: when the combined signal would exceed 0.99 (float32) it is
    scaled so the peak lands at 0.99, otherwise it passes through at full amplitude.

    Auto-amplification: when the mic signal is consistently below the VAD silence
    threshold but above the noise floor (~0.003), the mixer automatically boosts the
    mic before mixing so speech clears the gate. Gain is computed from an exponential
    moving average of mic RMS and capped at max_mic_gain (default 10x). The peak
    limiter prevents the amplified signal from clipping.

    Amp is gated by a second EMA of system-audio RMS: when system audio is currently
    playing (sys RMS above noise floor) the gain is held at 1x. Without this gate,
    speaker-to-mic acoustic bleed gets boosted alongside genuine speech, and the
    remote voice appears in the mix twice — once via the system tap, once via the
    amplified mic copy — producing a doubled / phased output that degrades both
    recording quality and transcription accuracy.
    """

    def __init__(
        self,
        mic: AudioSource,
        system_audio: AudioSource,
        *,
        silence_threshold: float = 0.01,
        max_mic_gain: float = 10.0,
    ) -> None:
        self._mic = mic
        self._system = system_audio
        self._mic_q: queue.Queue[np.ndarray] = queue.Queue()
        self._sys_q: queue.Queue[np.ndarray] = queue.Queue()
        # Rolling accumulation buffer for system audio chunks awaiting consumption.
        # Filled by draining _sys_q on every read(); consumed in equal-duration slices.
        self._sys_buf: np.ndarray = np.array([], dtype=np.float32)
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        # When True, mic audio is zeroed before mixing — system audio only.
        # Written by main thread, read by audio worker thread; bool assignment is
        # atomic under the GIL so no lock is needed.
        self.mic_muted: bool = False
        # Pre-amplification mic RMS from the most recent read().
        self.last_mic_rms: float = 0.0
        # Auto-amplification state.
        self._target_mic_rms = silence_threshold * 2.5
        self._noise_floor = 0.003
        self._max_mic_gain = max_mic_gain
        self._mic_rms_ema: float = 0.0
        self._sys_rms_ema: float = 0.0
        self._mic_gain: float = 1.0
        self._gain_logged: bool = False

    def open(self, sample_rate: int, channels: int) -> None:
        log.info(
            'opening mixed source: mic=%s, system=%s',
            type(self._mic).__name__,
            type(self._system).__name__,
        )
        self._mic.open(sample_rate, channels)
        self._system.open(sample_rate, channels)
        self._stop.clear()
        self._threads = [
            threading.Thread(target=self._reader, args=(self._mic, self._mic_q), daemon=True),
            threading.Thread(target=self._reader, args=(self._system, self._sys_q), daemon=True),
        ]
        for t in self._threads:
            t.start()

    def _reader(self, src, dest: queue.Queue) -> None:
        while not self._stop.is_set():
            chunk = src.read(timeout=0.05)
            if chunk is not None:
                dest.put(chunk)

    def read(self, timeout: float = 0.1) -> np.ndarray | None:
        # Mic chunk timing drives output cadence (see class docstring).
        try:
            first = self._mic_q.get(timeout=timeout)
        except queue.Empty:
            return None

        # Catch-up: drain any additional queued mic chunks and merge them.
        # Without this, a FIFO backlog on _mic_q becomes permanent lag —
        # consumer rate matches producer rate forever but the backlog never drains.
        # By pulling all pending chunks in one call, we let the consumer catch up
        # whenever it gets behind (e.g. GC pause, LLM CPU contention, pause/resume).
        extras: list[np.ndarray] = []
        while True:
            try:
                extras.append(self._mic_q.get_nowait())
            except queue.Empty:
                break
        mic = np.concatenate([first, *extras]) if extras else first

        # Pre-mute, pre-mix mic RMS. np.dot avoids a temporary mic**2 allocation.
        self.last_mic_rms = math.sqrt(float(np.dot(mic, mic)) / mic.size) if mic.size else 0.0

        # Update EMA of mic RMS (~2s convergence at 31 Hz with alpha=0.05).
        if self.last_mic_rms > 0:
            if self._mic_rms_ema == 0.0:
                self._mic_rms_ema = self.last_mic_rms  # seed on first non-zero read
            else:
                self._mic_rms_ema += 0.05 * (self.last_mic_rms - self._mic_rms_ema)

        # Auto-amplify: when EMA indicates "speaking but below VAD gate" AND the
        # system audio EMA shows the loopback is silent, boost mic so speech clears
        # silence_threshold. The sys-quiet gate prevents amplifying acoustic bleed
        # (speaker → mic) when the user is on speakers — without it, the remote
        # voice gets doubled in the mix. Peak limiter catches overflow.
        if (
            self._mic_rms_ema > self._noise_floor
            and self._mic_rms_ema < self._target_mic_rms
            and self._sys_rms_ema <= self._noise_floor
        ):
            self._mic_gain = min(self._target_mic_rms / self._mic_rms_ema, self._max_mic_gain)
            if not self._gain_logged:
                log.info(
                    'auto-amplifying mic: ema=%.4f target=%.4f gain=%.1fx',
                    self._mic_rms_ema,
                    self._target_mic_rms,
                    self._mic_gain,
                )
                self._gain_logged = True
        else:
            self._mic_gain = 1.0

        if self._mic_gain > 1.0:
            mic = mic * self._mic_gain

        # Zero mic data when muted — reader threads keep running to preserve
        # stream state; we just silence the mic contribution.
        if self.mic_muted:
            mic *= 0

        # Drain ALL available system chunks into the rolling buffer non-blocking.
        # get_nowait() is intentional: blocking here would stall the mic path and
        # cause _mic_q to accumulate unboundedly.
        # Collect into a list first and concatenate once — the prior pattern of
        # re-concatenating onto _sys_buf inside the loop was O(K²) in the number
        # of queued chunks, which produced multi-second stalls on a single read()
        # when the consumer had fallen behind by minutes (3000 chunks → ~2.6 s).
        pending: list[np.ndarray] = []
        while True:
            try:
                pending.append(self._sys_q.get_nowait())
            except queue.Empty:
                break
        if pending:
            self._sys_buf = np.concatenate([self._sys_buf, *pending]) if self._sys_buf.size else np.concatenate(pending)

        if len(self._sys_buf) == 0:
            return mic  # no system audio yet; pass mic through at full amplitude

        # Consume exactly len(mic) samples from the system buffer so both sides
        # cover the same time window regardless of their native chunk sizes.
        n = len(mic)
        if len(self._sys_buf) >= n:
            sys = self._sys_buf[:n]
            self._sys_buf = self._sys_buf[n:]
        else:
            # System buffer is shorter than mic chunk; zero-pad the tail.
            sys = np.pad(self._sys_buf, (0, n - len(self._sys_buf)))
            self._sys_buf = np.array([], dtype=np.float32)

        # Track system audio activity for next read's amp gate (one-chunk lag is
        # well below the EMA time constant). Same alpha as mic EMA for symmetry.
        sys_rms = math.sqrt(float(np.dot(sys, sys)) / sys.size) if sys.size else 0.0
        if self._sys_rms_ema == 0.0:
            self._sys_rms_ema = sys_rms
        else:
            self._sys_rms_ema += 0.05 * (sys_rms - self._sys_rms_ema)

        # Peak limiter: scale to 0.99 only when clipping. Two scalar reductions
        # (max + min) instead of np.abs() to avoid a temp-array allocation on
        # every read (~31 Hz hot path).
        combined = mic + sys
        peak = max(float(combined.max()), float(-combined.min()))
        if peak > 0.99:
            combined *= 0.99 / peak
        return combined

    def drain(self) -> None:
        """Discard all buffered audio — called on pause so resume starts fresh.

        Drains our own mixing queues AND delegates to each underlying source so
        their internal queues don't bleed pause-period audio through on resume.
        """
        for q in (self._mic_q, self._sys_q):
            while True:
                try:
                    q.get_nowait()
                except queue.Empty:
                    break
        self._sys_buf = np.array([], dtype=np.float32)
        self.last_mic_rms = 0.0
        self._mic_rms_ema = 0.0
        self._sys_rms_ema = 0.0
        self._mic_gain = 1.0
        self._gain_logged = False
        self._mic.drain()
        self._system.drain()

    def close(self) -> None:
        log.debug('closing mixed source')
        self._stop.set()
        self._mic.close()
        self._system.close()
        for t in self._threads:
            t.join(timeout=2)
        self._threads = []
