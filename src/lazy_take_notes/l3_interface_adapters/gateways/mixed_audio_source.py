"""Gateway: MixedAudioSource — composites microphone and system audio into one stream."""

from __future__ import annotations

import logging
import math
import queue
import threading
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from lazy_take_notes.l3_interface_adapters.gateways._audio_buffer import AudioBuffer

if TYPE_CHECKING:
    from lazy_take_notes.l2_use_cases.ports.audio_source import AudioSource

log = logging.getLogger('ltn.audio.mixed')


class MixedAudioSource:
    """Mix both sources while keeping a failed source from blocking its peer.

    Silence remains valid audio. Only delivery failures trigger recovery, and a
    lost source is retried with backoff for as long as the stream is open: a
    headset that reconnects after a minute rejoins the same recording.
    Excess queued audio is discarded with a warning to bound capture delay.
    Gain, microphone mute, and the peak limiter apply after source selection.

    Health (``health``): ``lost`` while a source is being recovered. ``silent``
    when the raw microphone sends exact zeros, which a real microphone never
    does; a restart cannot fix that (input volume 0, a virtual device), so it
    is reported only. The system tap also sends exact zeros when nothing plays,
    so after a silent stretch ``test_tone`` plays an inaudible tone; a tap that
    does not return it is treated as stalled and restarted.
    """

    _restart_delays: tuple[float, ...] = (0.3, 1.0, 2.0, 5.0)
    _zero_alarm_seconds = 3.0
    _tone_after_seconds = 30.0
    _tone_wait_seconds = 1.5

    def __init__(
        self,
        mic: AudioSource,
        system_audio: AudioSource,
        *,
        silence_threshold: float = 0.01,
        max_mic_gain: float = 10.0,
        dead_audio_seconds: float = 3.0,
        test_tone: Callable[[], bool] | None = None,
    ) -> None:
        self._mic = mic
        self._system = system_audio
        self._mic_q = AudioBuffer(max_frames=4000, sample_rate=16_000, label='mixed microphone')
        self._sys_q = AudioBuffer(max_frames=4000, sample_rate=16_000, label='mixed system')
        self._sys_buf: np.ndarray = np.array([], dtype=np.float32)
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._sample_rate: int = 16_000
        self._channels: int = 1
        self._dead_audio_seconds = dead_audio_seconds
        self._ready = threading.Event()
        self._state_lock = threading.Lock()
        self._source_locks = (threading.Lock(), threading.Lock())
        self._errors: dict[bool, str] = {}
        self._silent_mic = False
        # Set when the tap swallowed the test tone; cleared only by a real sample,
        # because a restarted tap delivers zeros that look healthy.
        self._deaf_system = False
        self._last_delivery = {True: 0.0, False: 0.0}
        self._test_tone = test_tone
        self._system_generation = 0
        self._seen_system_generation = 0
        self._dropped_system_frames = 0
        self._last_system_drop_log = 0.0
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

    @property
    def system_signal_seen(self) -> bool:
        """Whether the system source delivered a non-zero sample.

        Sources that cannot report this count as seen, so the check never
        fires for a backend that has no such signal to give.
        """
        return getattr(self._system, 'system_signal_seen', True)

    @property
    def awaiting_permission(self) -> bool:
        """Whether the system source is still waiting for the user to grant access."""
        return getattr(self._system, 'awaiting_permission', False)

    @property
    def source_errors(self) -> tuple[str, ...]:
        """Current delivery failures, including recovery in progress."""
        with self._state_lock:
            return tuple(self._errors.values())

    @property
    def dropped_frames(self) -> int:
        return self._mic_q.dropped_frames + self._sys_q.dropped_frames + self._dropped_system_frames

    @property
    def exhausted(self) -> bool:
        """Never true: a lost source is retried until close()."""
        return False

    @property
    def mic_notice(self) -> str:
        """Set when the configured microphone is missing and another input is used."""
        return getattr(self._mic, 'device_notice', '')

    @property
    def health(self) -> dict[str, str]:
        with self._state_lock:
            mic_lost = True in self._errors or self._stalled(self._mic, True)
            system_lost = False in self._errors or self._deaf_system or self._stalled(self._system, False)
            mic = 'lost' if mic_lost else 'silent' if self._silent_mic else 'ok'
        return {'mic': mic, 'system': 'lost' if system_lost else 'ok'}

    def _stalled(self, source: AudioSource, is_mic: bool) -> bool:
        """A reader stuck inside the source reports nothing, so judge it by delivery time."""
        if not self._threads or not getattr(source, 'continuous_capture', False):
            return False
        if getattr(source, 'awaiting_permission', False):
            return False
        return time.monotonic() - self._last_delivery[is_mic] > self._dead_audio_seconds

    def open(self, sample_rate: int, channels: int) -> None:
        if self._threads:
            raise RuntimeError('Mixed audio source is already open')
        self._sample_rate = sample_rate
        self._channels = channels
        self._mic_q = AudioBuffer(
            max_frames=max(1, sample_rate // 4), sample_rate=sample_rate, label='mixed microphone'
        )
        self._sys_q = AudioBuffer(max_frames=max(1, sample_rate // 4), sample_rate=sample_rate, label='mixed system')
        self._dropped_system_frames = 0
        self._last_system_drop_log = 0.0
        self._errors.clear()
        self._silent_mic = False
        self._deaf_system = False
        self._last_delivery = {True: time.monotonic(), False: time.monotonic()}
        self._clear_buffers()
        # A source that fails to open is retried like a lost one. The other source
        # records from the first second; the recording must not wait for both.
        mic_error = self._open_source(self._mic, True)
        system_error = self._open_source(self._system, False)
        self._stop.clear()
        self._threads = [
            threading.Thread(target=self._reader, args=(self._mic, self._mic_q, True, mic_error), daemon=True),
            threading.Thread(target=self._reader, args=(self._system, self._sys_q, False, system_error), daemon=True),
        ]
        try:
            for thread in self._threads:
                thread.start()
        except Exception:
            self._stop.set()
            self._close_sources()
            self._join_threads()
            raise

    def _open_source(self, source: AudioSource, is_mic: bool) -> Exception | None:
        try:
            source.open(self._sample_rate, self._channels)
        except Exception as error:
            log.warning('%s audio source failed to open; retrying: %s', 'mic' if is_mic else 'system', error)
            self._set_error(is_mic, error)
            return error
        return None

    def _reader(
        self, source: AudioSource, destination: AudioBuffer, is_mic: bool, open_error: Exception | None = None
    ) -> None:
        last_delivery = time.monotonic()
        watch = _ZeroWatch()
        while not self._stop.is_set():
            try:
                if open_error is not None:
                    error, open_error = open_error, None
                    raise error
                chunk = self._read_source(source, last_delivery)
                if chunk is not None and chunk.size:
                    self._watch_signal(watch, chunk, is_mic)
            except Exception as error:
                watch = _ZeroWatch()
                chunk = self._recover(source, destination, is_mic, error)
                if chunk is None:
                    return
            if chunk is None or not chunk.size:
                self._stop.wait(0.005)
                continue
            if self._stop.is_set():
                return
            last_delivery = time.monotonic()
            self._last_delivery[is_mic] = last_delivery
            self._set_error(is_mic, None)
            destination.put(chunk)
            self._ready.set()

    def _watch_signal(self, watch: _ZeroWatch, chunk: np.ndarray, is_mic: bool) -> None:
        """Track exact-zero stretches; raise when the system tap swallowed the test tone."""
        now = time.monotonic()
        if chunk.any():
            if watch.tone_sent_at is not None:
                log.info('system audio confirmed: signal after the test tone')
            watch.reset()
            if is_mic:
                self._set_silent_mic(False)
            else:
                self._set_deaf_system(False)
            return
        if watch.zero_since is None:
            watch.zero_since = now
        silent_for = now - watch.zero_since
        if is_mic:
            if silent_for >= self._zero_alarm_seconds:
                self._set_silent_mic(True)
            return
        if self._test_tone is None:
            return
        if watch.tone_sent_at is None:
            if silent_for >= self._tone_after_seconds and self._play_tone():
                log.info('system audio silent for %.0fs; test tone played', silent_for)
                watch.tone_sent_at = time.monotonic()
            return
        if now - watch.tone_sent_at >= self._tone_wait_seconds:
            self._set_deaf_system(True)
            raise RuntimeError('system audio did not pick up the test tone')

    def _set_deaf_system(self, deaf: bool) -> None:
        with self._state_lock:
            changed = self._deaf_system != deaf
            self._deaf_system = deaf
        if changed and not deaf:
            log.info('system audio picked up signal again')

    def _play_tone(self) -> bool:
        try:
            return bool(self._test_tone and self._test_tone())
        except Exception:
            log.warning('test tone failed to play; system audio check skipped', exc_info=True)
            return False

    def _set_silent_mic(self, silent: bool) -> None:
        with self._state_lock:
            changed = self._silent_mic != silent
            self._silent_mic = silent
        if changed:
            log.warning('microphone sends exact zeros') if silent else log.info('microphone signal is back')
            self._ready.set()

    def _read_source(self, source: AudioSource, last_delivery: float) -> np.ndarray | None:
        chunk = source.read(timeout=0.05)
        if chunk is not None and chunk.size:
            return chunk
        if getattr(source, 'exhausted', False):
            raise RuntimeError('capture stream ended unexpectedly')
        if getattr(source, 'awaiting_permission', False):
            return None
        if (
            getattr(source, 'continuous_capture', False)
            and time.monotonic() - last_delivery >= self._dead_audio_seconds
        ):
            raise RuntimeError('capture stopped delivering audio buffers')
        return None

    def _set_error(self, is_mic: bool, error: Exception | None) -> None:
        label = 'Microphone' if is_mic else 'System audio'
        with self._state_lock:
            if error is None:
                self._errors.pop(is_mic, None)
            else:
                self._errors[is_mic] = f'{label}: {error}'
        self._ready.set()

    def _recover(
        self, source: AudioSource, destination: AudioBuffer, is_mic: bool, error: Exception
    ) -> np.ndarray | None:
        label = 'mic' if is_mic else 'system'
        self._set_error(is_mic, error)
        attempt = 0
        while not self._stop.wait(self._restart_delay_for(attempt)):
            attempt += 1
            log.warning('restarting %s audio source (attempt %d): %s', label, attempt, error)
            try:
                self._reopen(source, destination, is_mic)
                chunk = self._wait_for_first_buffer(source)
                if chunk is not None:
                    log.info('%s audio source recovered after %d attempt(s)', label, attempt)
                    return chunk
                error = RuntimeError('capture did not deliver a buffer after restart')
            except Exception as reopen_error:
                error = reopen_error
            self._set_error(is_mic, error)
        return None

    def _restart_delay_for(self, attempt: int) -> float:
        return self._restart_delays[min(attempt, len(self._restart_delays) - 1)]

    def _reopen(self, source: AudioSource, destination: AudioBuffer, is_mic: bool) -> None:
        with self._source_locks[int(is_mic)]:
            if self._stop.is_set():
                return
            source.close()
            self._discard(destination)
            if not is_mic:
                self._system_generation += 1
            source.open(self._sample_rate, self._channels)

    def _wait_for_first_buffer(self, source: AudioSource) -> np.ndarray | None:
        deadline = time.monotonic() + max(0.05, min(self._dead_audio_seconds, 3.0))
        while not self._stop.is_set() and time.monotonic() < deadline:
            chunk = source.read(timeout=0.05)
            if chunk is not None and chunk.size:
                return chunk
            if getattr(source, 'exhausted', False):
                raise RuntimeError('capture stream ended during restart')
            self._stop.wait(0.005)
        return None

    @staticmethod
    def _discard(destination: AudioBuffer) -> None:
        while True:
            try:
                destination.get_nowait()
            except queue.Empty:
                break

    def _take_microphone(self, timeout: float) -> np.ndarray | None:
        deadline = time.monotonic() + timeout
        while True:
            self._ready.clear()
            chunks = self._take_pending(self._mic_q)
            if chunks:
                return np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
            with self._state_lock:
                microphone_failed = True in self._errors
            system_pending = not self._sys_q.empty() or self._sys_buf.size
            if system_pending and (microphone_failed or not self._threads):
                return None
            remaining = deadline - time.monotonic()
            if remaining <= 0 or self._stop.is_set():
                return None
            self._ready.wait(remaining)

    @staticmethod
    def _take_pending(source_queue: AudioBuffer) -> list[np.ndarray]:
        chunks: list[np.ndarray] = []
        while True:
            try:
                chunks.append(source_queue.get_nowait())
            except queue.Empty:
                return chunks

    def _take_system(self) -> None:
        if self._seen_system_generation != self._system_generation:
            self._sys_buf = np.array([], dtype=np.float32)
            self._seen_system_generation = self._system_generation
        chunks = self._take_pending(self._sys_q)
        if chunks:
            self._sys_buf = np.concatenate([self._sys_buf, *chunks])
        limit = max(1, self._sample_rate // 4)
        discarded = self._sys_buf.size - limit
        if discarded > 0:
            self._dropped_system_frames += discarded
            if time.monotonic() - self._last_system_drop_log >= 1.0:
                log.warning(
                    'system audio buffer dropped %d old samples (%.3fs)',
                    self._dropped_system_frames,
                    self._dropped_system_frames / self._sample_rate,
                )
                self._last_system_drop_log = time.monotonic()
            self._sys_buf = self._sys_buf[-limit:].copy()

    def read(self, timeout: float = 0.1) -> np.ndarray | None:
        mic = self._take_microphone(timeout)
        self._take_system()
        if mic is None:
            self.last_mic_rms = 0.0
            if not self._sys_buf.size:
                return None
            system = self._sys_buf
            self._sys_buf = np.array([], dtype=np.float32)
            return self._limit_peak(system)

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

        return self._limit_peak(mic + sys)

    @staticmethod
    def _limit_peak(samples: np.ndarray) -> np.ndarray:
        peak = max(float(samples.max()), float(-samples.min()))
        if peak > 0.99:
            samples = samples * (0.99 / peak)
        return samples

    def _clear_buffers(self) -> None:
        for source_queue in (self._mic_q, self._sys_q):
            self._discard(source_queue)
        self._sys_buf = np.array([], dtype=np.float32)
        self.last_mic_rms = 0.0
        self._mic_rms_ema = 0.0
        self._sys_rms_ema = 0.0
        self._mic_gain = 1.0
        self._gain_logged = False

    def drain(self) -> None:
        self._clear_buffers()
        for is_mic, source in ((True, self._mic), (False, self._system)):
            with self._source_locks[int(is_mic)]:
                source.drain()

    def _close_sources(self) -> list[Exception]:
        errors: list[Exception] = []
        for is_mic, source in ((True, self._mic), (False, self._system)):
            try:
                with self._source_locks[int(is_mic)]:
                    source.close()
            except Exception as error:
                log.exception('failed to close %s audio source', 'mic' if is_mic else 'system')
                errors.append(error)
        return errors

    def _join_threads(self) -> list[Exception]:
        errors: list[Exception] = []
        for thread in self._threads:
            if thread.ident is not None:
                thread.join(timeout=2)
            if thread.is_alive():
                errors.append(RuntimeError('Audio reader did not stop'))
        self._threads = [thread for thread in self._threads if thread.is_alive()]
        return errors

    def close(self) -> None:
        self._stop.set()
        self._ready.set()
        errors = self._close_sources()
        errors.extend(self._join_threads())
        self._discard(self._mic_q)
        self._discard(self._sys_q)
        self._sys_buf = np.array([], dtype=np.float32)
        if errors:
            raise errors[0]


class _ZeroWatch:
    """Per-reader state for an exact-zero stretch and a pending test tone."""

    def __init__(self) -> None:
        self.zero_since: float | None = None
        self.tone_sent_at: float | None = None

    def reset(self) -> None:
        self.zero_since = None
        self.tone_sent_at = None
