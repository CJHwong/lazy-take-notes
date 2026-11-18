"""Thin thread worker shell for audio capture — connects AudioSource to TranscribeAudioUseCase."""

from __future__ import annotations

import concurrent.futures
import errno
import logging
import math
import os
import queue
import struct
import threading
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path

import numpy as np

from lazy_take_notes.l1_entities.audio_constants import SAMPLE_RATE
from lazy_take_notes.l1_entities.transcript import TranscriptSegment
from lazy_take_notes.l2_use_cases.ports.audio_source import AudioSource
from lazy_take_notes.l2_use_cases.transcribe_audio_use_case import TranscribeAudioUseCase
from lazy_take_notes.l4_frameworks_and_drivers.messages import (
    AudioHealth,
    AudioLevel,
    AudioWorkerStatus,
    TranscriptChunk,
    TranscriptionStatus,
)
from lazy_take_notes.l4_frameworks_and_drivers.workers.session_folder import SessionFolder

log = logging.getLogger('ltn.audio')

# A missing audio capture grant makes macOS hand the tap a stream of zeros with
# no error. If no system sample arrives in this window, say so once.
_NO_SIGNAL_WARN_SECONDS = 15.0

# A failing disk: retry this often, and hold at most this much audio in memory (about 19 MB).
WAV_RETRY_S = 5.0
WAV_BACKLOG_S = 600.0


def _no_signal_message() -> str:
    return (
        f'No system audio received in the first {int(_NO_SIGNAL_WARN_SECONDS)}s. '
        'If you expect it, grant "System Audio Recording Only" to this app in '
        'System Settings > Privacy & Security > Screen & System Audio Recording.'
    )


def _no_signal_hint_due(audio_source) -> bool:
    """Whether 15 s without system audio should name the permission.

    Not while the permission dialog is open. Not when the source proves the tap
    with a test tone: a quiet room is then normal, and a dead tap shows SYS LOST.
    """
    if getattr(audio_source, 'awaiting_permission', False):
        return False
    if getattr(audio_source, 'verifies_system_audio', False):
        return False
    return not getattr(audio_source, 'system_signal_seen', True)


class _StreamingWavWriter:
    """WAV writer that keeps the RIFF sizes current after every chunk.

    The diarizer reads the file while the session records. Python's wave module
    leaves both sizes at zero until close, and a reader that trusts the header
    fails with "malformed WAV: truncated RIFF payload". Rewriting 8 header bytes
    per chunk makes the file valid at any instant.
    """

    def __init__(self, path: Path, sample_rate: int) -> None:
        self.path = path
        self._file = open(path, 'wb')  # noqa: SIM115 -- kept open for the whole session
        self._bytes = 0
        self._file.write(
            b'RIFF'
            + struct.pack('<I', 36)
            + b'WAVEfmt '
            + struct.pack('<IHHIIHH', 16, 1, 1, sample_rate, sample_rate * 2, 2, 16)
            + b'data'
            + struct.pack('<I', 0)
        )

    def write(self, pcm: bytes) -> None:
        self._file.write(pcm)
        self._bytes += len(pcm)
        self._file.seek(4)
        self._file.write(struct.pack('<I', 36 + self._bytes))
        self._file.seek(40)
        self._file.write(struct.pack('<I', self._bytes))
        self._file.seek(0, os.SEEK_END)
        self._file.flush()
        if not self._still_at_path():
            # Writes to a removed or moved file (Finder's Move to Trash) still succeed.
            raise FileNotFoundError(errno.ENOENT, 'the recording file was moved or removed', str(self.path))

    def _still_at_path(self) -> bool:
        try:
            on_disk = os.stat(self.path)
        except FileNotFoundError:
            return False
        opened = os.fstat(self._file.fileno())
        return (on_disk.st_dev, on_disk.st_ino) == (opened.st_dev, opened.st_ino)

    @property
    def bytes_written(self) -> int:
        return self._bytes

    def close(self) -> None:
        self._file.close()


class _RecoveringWav:
    """The session WAV, kept going through a failing disk.

    A failed open or write (full disk, removed folder) keeps the audio in memory,
    up to WAV_BACKLOG_S, and retries every WAV_RETRY_S. Each retry writes a new
    file (recording_2.wav, ...), so a file that was valid up to the failure stays
    valid. The meeting can always be recovered from the audio, so the audio must
    not stop.
    """

    def __init__(
        self,
        output_dir: Path | SessionFolder,
        sample_rate: int,
        on_notice: Callable[[str], None] | None,
        keep_files: Callable[[], bool] | None = None,
    ) -> None:
        self._folder = output_dir if isinstance(output_dir, SessionFolder) else SessionFolder(output_dir)
        self._keep_files = keep_files
        self._names: list[str] = []
        self._sample_rate = sample_rate
        self._notice = on_notice
        self._wav: _StreamingWavWriter | None = None
        self._backlog: deque[bytes] = deque()
        self._backlog_bytes = 0
        self._backlog_limit = int(WAV_BACKLOG_S * sample_rate * 2)
        self._retry_at = 0.0
        self._failing = False
        self._dropping = False

    @property
    def _dir(self) -> Path:
        return self._folder.path

    def write(self, chunk: np.ndarray) -> None:
        self._hold((np.clip(chunk, -1.0, 1.0) * 32767).astype(np.int16).tobytes())
        if self._wav is None and time.monotonic() < self._retry_at:
            return
        self._flush()

    def close(self) -> None:
        self._retry_at = 0.0
        self._flush()
        with self._folder.lock:
            self._close_wav()
            if not self._keep_files_at_end():
                self._delete_files()
        if self._backlog:
            lost = self._backlog_bytes / (self._sample_rate * 2)
            log.error('The disk still fails at the end of the session; %.0f s of audio was not saved', lost)

    def _keep_files_at_end(self) -> bool:
        return self._keep_files is None or self._keep_files()

    def _delete_files(self) -> None:
        for name in self._names:
            path = self._dir / name
            try:
                path.unlink(missing_ok=True)
            except OSError:
                log.warning('Could not delete %s', path, exc_info=True)
        log.info('save_audio is off and nothing failed; deleted the session audio')

    def _flush(self) -> None:
        # The lock covers one chunk at a time, so a rename never waits for a long backlog.
        try:
            while True:
                with self._folder.lock:
                    if self._wav is None:
                        self._wav = self._open_next()
                    wav = self._wav
                    wav.path = self._dir / wav.path.name  # follows a rename through SessionFolder
                    if not self._backlog:
                        break
                    wav.write(self._backlog[0])
                    self._backlog_bytes -= len(self._backlog.popleft())
        except OSError as exc:
            self._fail(exc)
            return
        if self._failing:
            self._failing = False
            self._dropping = False
            log.warning('Recording continues in %s', wav.path)
            self._tell(f'Audio recording continues in {wav.path.name}.')

    def _open_next(self) -> _StreamingWavWriter:
        self._dir.mkdir(parents=True, exist_ok=True)  # the session folder may have been removed
        path = self._dir / 'recording.wav'
        number = 2
        while path.exists():
            path = self._dir / f'recording_{number}.wav'
            number += 1
        wav = _StreamingWavWriter(path, self._sample_rate)
        self._names.append(path.name)
        return wav

    def _hold(self, pcm: bytes) -> None:
        self._backlog.append(pcm)
        self._backlog_bytes += len(pcm)
        while self._backlog_bytes > self._backlog_limit and len(self._backlog) > 1:
            self._backlog_bytes -= len(self._backlog.popleft())
            if not self._dropping:
                self._dropping = True
                log.error('Audio held in memory passed %.0f s; dropping the oldest audio', WAV_BACKLOG_S)

    def _fail(self, exc: OSError) -> None:
        already_failing = self._failing
        self._failing = True
        with self._folder.lock:
            self._close_wav()
        self._retry_at = time.monotonic() + WAV_RETRY_S
        if already_failing:
            log.debug('Recording retry failed: %s', exc)
            return
        log.error('Recording to %s paused: %s', self._dir, exc, exc_info=True)
        self._tell(f'Audio recording paused: {exc}. Audio is kept in memory and saved when the disk works again.')

    def _close_wav(self) -> None:
        wav, self._wav = self._wav, None
        if wav is None:
            return
        try:
            wav.close()
            if self._failing and wav.bytes_written == 0:
                wav.path.unlink()  # a failed retry holds no audio; the next one reuses the name
        except OSError:
            log.debug('closing a failed WAV file', exc_info=True)

    def _tell(self, message: str) -> None:
        if self._notice is not None:
            self._notice(message)


def _start_processed_recorder(
    output_dir: Path | SessionFolder,
    sample_rate: int,
    on_error: Callable[[str], None] | None = None,
    keep_files: Callable[[], bool] | None = None,
) -> tuple[queue.Queue, threading.Thread]:
    """Write processed float32 chunks (system/mixed audio) to a WAV file.

    Send None into the returned queue to signal the writer to flush and close.
    The writer always drains the queue, so the unbounded queue cannot grow while
    the disk fails. *on_error* hears when the recording pauses and resumes. When
    *keep_files* returns False at the end, the writer deletes the files it wrote.
    """
    rec_q: queue.Queue[np.ndarray | None] = queue.Queue()
    recording = _RecoveringWav(output_dir, sample_rate, on_error, keep_files)

    def _writer() -> None:
        while True:
            try:
                data = rec_q.get(timeout=0.5)
            except queue.Empty:  # pragma: no cover -- timing-dependent; queue.get timeout retry
                continue
            if data is None:
                break
            recording.write(data)
        recording.close()

    writer = threading.Thread(target=_writer, daemon=True)
    writer.start()
    return rec_q, writer


# Consecutive loop failures for this long mean nothing is being recorded.
_LOOP_ALERT_AFTER_S = 1.0


class _LoopHealth:
    """Turns consecutive audio-loop failures into one persistent alert, and clears it."""

    def __init__(self, post_message: Callable) -> None:
        self._post = post_message
        self._failing_since: float | None = None
        self._failures = 0
        self._alerted = False

    def failed(self, error: Exception, health: tuple[str, str, str] | None) -> None:
        now = time.monotonic()
        self._failures += 1
        if self._failing_since is None:
            self._failing_since = now
            log.error('Audio loop error; recording continues', exc_info=error)
            self._post(
                AudioWorkerStatus(status='warning', error=f'Audio processing error: {error}. Recording continues.')
            )
        elif self._failures % 100 == 0:
            log.error('Audio loop still failing (%d in a row)', self._failures, exc_info=error)
        if not self._alerted and now - self._failing_since >= _LOOP_ALERT_AFTER_S:
            self._alerted = True
            self._post(_health_message(health, processing='error', detail=str(error)))

    def succeeded(self, health: tuple[str, str, str] | None) -> None:
        if self._failing_since is None:
            return
        log.info('Audio loop recovered after %d failure(s)', self._failures)
        self._failing_since = None
        self._failures = 0
        if self._alerted:
            self._alerted = False
            self._post(_health_message(health, processing='ok'))


def _health_message(health: tuple[str, str, str] | None, *, processing: str, detail: str = '') -> AudioHealth:
    mic, system, notice = health or ('ok', 'ok', '')
    return AudioHealth(mic=mic, system=system, detail=detail, notice=notice, processing=processing)


def _report_health(
    audio_source: AudioSource,
    source_errors: set[str],
    reported: tuple[str, str, str] | None,
    post_message: Callable,
) -> tuple[str, str, str] | None:
    """Post AudioHealth when a source's state changes. Retries alone post nothing."""
    health = getattr(audio_source, 'health', None)
    if health is None:
        return reported
    current = (health['mic'], health['system'], getattr(audio_source, 'mic_notice', ''))
    if current != reported:
        log.info('audio health: mic=%s system=%s %s', *current)
        detail = '; '.join(sorted(source_errors))
        post_message(AudioHealth(mic=current[0], system=current[1], detail=detail, notice=current[2]))
    return current


class _UnsavedAudio:
    """With save_audio off, the session audio is still written, then deleted at a clean end.

    A failure keeps it: the audio is then the only full record of the meeting.
    A crash keeps it too, because nothing runs to delete it.
    """

    def __init__(self, save_audio: bool, post_message: Callable) -> None:
        self.keep = save_audio
        self._post_message = post_message

    def keep_because(self, reason: str) -> None:
        if self.keep:
            return
        self.keep = True
        log.warning('save_audio is off, but %s; keeping the session audio', reason)
        message = (
            f'save_audio is off, but {reason}. The audio of this meeting is kept in recording.wav, '
            'so you can transcribe it later.'
        )
        self._post_message(AudioWorkerStatus(status='warning', error=message))

    def notice(self, message: str) -> None:
        """Disk notices about audio the user turned off would only confuse; the log keeps them."""
        if self.keep:
            self._post_message(AudioWorkerStatus(status='warning', error=message))


class _NoTranscription:
    """Stands in for a speech model that failed to load, so the audio still records."""

    def load_model(self, model_path: str) -> None:
        return

    def transcribe(self, audio: np.ndarray, language: str, hints: list[str] | None = None) -> list[TranscriptSegment]:
        return []

    def close(self) -> None:
        return


def run_audio_worker(
    post_message,
    is_cancelled,
    model_path: str,
    language: str,
    chunk_duration: float = 10.0,
    overlap: float = 1.0,
    silence_threshold: float = 0.01,
    pause_duration: float = 1.5,
    recognition_hints: list[str] | None = None,
    pause_event: threading.Event | None = None,
    output_dir: Path | SessionFolder | None = None,
    save_audio: bool = False,
    transcriber=None,
    audio_source: AudioSource | None = None,
) -> list[TranscriptSegment]:
    """Audio capture and transcription loop.

    Designed to run inside a Textual @work(thread=True) worker.
    """
    unsaved = _UnsavedAudio(save_audio, post_message)

    # Load model
    post_message(AudioWorkerStatus(status='loading_model'))
    try:
        if transcriber is None:  # pragma: no cover -- default wiring; transcriber always injected in tests
            from lazy_take_notes.l3_interface_adapters.gateways.subprocess_whisper_transcriber import (  # noqa: PLC0415 -- deferred: subprocess spawned only when worker starts
                SubprocessWhisperTranscriber,
            )

            transcriber = SubprocessWhisperTranscriber()
        transcriber.load_model(model_path)
    except Exception as e:
        # The audio is the record of the meeting: record it without a transcript.
        log.error('Failed to load transcription model; recording audio only: %s', e, exc_info=True)
        if transcriber is not None:
            transcriber.close()  # clean up any partially-started subprocess
        transcriber = _NoTranscription()
        message = f'Speech model failed to load: {e}. Recording audio only; run transcribe on the recording later.'
        post_message(AudioWorkerStatus(status='warning', error=message))
        unsaved.keep_because('the speech model failed to load')
    post_message(AudioWorkerStatus(status='model_ready'))

    # Resolve audio source — default to SounddeviceAudioSource when not injected
    if audio_source is None:  # pragma: no cover -- default wiring; audio_source always injected in tests
        from lazy_take_notes.l3_interface_adapters.gateways.sounddevice_audio_source import (  # noqa: PLC0415 -- deferred: sounddevice loaded only when worker starts
            SounddeviceAudioSource,
        )

        audio_source = SounddeviceAudioSource()

    use_case = TranscribeAudioUseCase(
        transcriber=transcriber,
        language=language,
        chunk_duration=chunk_duration,
        overlap=overlap,
        silence_threshold=silence_threshold,
        pause_duration=pause_duration,
        recognition_hints=recognition_hints,
    )

    all_segments: list[TranscriptSegment] = []
    reported_source_errors: set[str] = set()
    reported_health: tuple[str, str, str] | None = None
    total_samples_fed: int = 0
    _last_level_post: float = 0.0
    _level_accum: list[np.ndarray] = []

    # Periodic stats accumulators (30s intervals)
    _stats_interval: float = 30.0
    _stats_last_time: float = 0.0  # set after audio_source.open()
    _stats_rms_sum: float = 0.0
    _stats_rms_count: int = 0
    _stats_zero_samples: int = 0
    _stats_total_samples: int = 0
    _stats_transcriptions: int = 0

    # Off-thread transcription: audio reading continues while subprocess infers.
    _executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    _transcript_future: concurrent.futures.Future | None = None
    _pending_meta: tuple[float, bool] | None = None  # (buffer_wall_start, is_first_chunk)

    def _collect_future() -> None:
        nonlocal _transcript_future, _pending_meta
        if _transcript_future is None or _pending_meta is None or not _transcript_future.done():
            return
        try:
            raw_segs = _transcript_future.result()
            buf_start, was_first = _pending_meta
            new_segs = use_case.apply_result(raw_segs, buf_start, was_first)
            log.debug('transcription complete: %d segments', len(new_segs))
            if new_segs:
                all_segments.extend(new_segs)
                post_message(TranscriptChunk(segments=new_segs))
        except Exception as e:
            log.error('Transcription failed: %s', e, exc_info=True)
            unsaved.keep_because('a transcription failed')
            post_message(AudioWorkerStatus(status='error', error=str(e)))
        finally:
            _transcript_future = None
            _pending_meta = None
            post_message(TranscriptionStatus(active=False))

    proc_rec_q: queue.Queue | None = None
    proc_rec_writer: threading.Thread | None = None
    if output_dir:  # with save_audio off too, see _UnsavedAudio
        try:
            proc_rec_q, proc_rec_writer = _start_processed_recorder(
                output_dir,
                SAMPLE_RATE,
                on_error=unsaved.notice,
                keep_files=lambda: unsaved.keep,
            )
        except Exception:  # noqa: S110 -- best-effort; recording continues without save  # pragma: no cover
            pass

    # Start recording
    post_message(AudioWorkerStatus(status='recording'))
    _worker_start: float = 0.0
    try:
        audio_source.open(SAMPLE_RATE, 1)
        log.info('Audio source opened: %s', type(audio_source).__name__)
        _worker_start = time.monotonic()
        _stats_last_time = _worker_start
        _last_data_time = _worker_start
        _silence_logged_at: float = 0.0
        _was_paused = False
        _no_signal_warned = False
        try:
            loop_health = _LoopHealth(post_message)
            while not is_cancelled():
                iteration_failed = False
                try:
                    if pause_event is not None and pause_event.is_set():
                        if not _was_paused:
                            # Pause-entry: commit buffered audio so the user's last
                            # sentence isn't silently lost by reset_buffer().

                            # Wait for in-flight transcription (single-use subprocess).
                            if _transcript_future is not None:
                                try:
                                    _transcript_future.result(timeout=120)
                                    _collect_future()
                                except Exception as e:
                                    log.error(
                                        'Flushing in-flight transcription on pause failed: %s',
                                        e,
                                        exc_info=True,
                                    )
                                    unsaved.keep_because('a transcription failed')

                            # Flush remaining buffer through the transcriber.
                            prepared = use_case.prepare_buffer()
                            if prepared is not None and len(prepared[0]) > 0:
                                flush_buf, flush_hints, flush_wall_start, flush_is_first = prepared
                                log.debug(
                                    'pause-flush transcribing %d samples',
                                    len(flush_buf),
                                )
                                post_message(TranscriptionStatus(active=True))
                                try:
                                    flush_segs = transcriber.transcribe(flush_buf, language, flush_hints)
                                    new_segs = use_case.apply_result(
                                        flush_segs, flush_wall_start, flush_is_first, final=True
                                    )
                                    if new_segs:
                                        all_segments.extend(new_segs)
                                        post_message(TranscriptChunk(segments=new_segs))
                                except Exception as e:
                                    log.error('Pause flush transcription failed: %s', e, exc_info=True)
                                    unsaved.keep_because('a transcription failed')
                                finally:
                                    post_message(TranscriptionStatus(active=False))

                            # Discard audio that arrived during the flush.
                            audio_source.drain()
                            use_case.reset_buffer()
                            _was_paused = True

                        # During pause: keep draining so producer queues don't grow.
                        audio_source.drain()
                        _collect_future()
                        time.sleep(0.1)
                        _last_data_time = time.monotonic()  # don't count pause as silence
                        continue

                    _was_paused = False
                    _collect_future()

                    data = audio_source.read(timeout=0.1)
                    source_errors: set[str] = set(getattr(audio_source, 'source_errors', ()))
                    for source_error in sorted(source_errors - reported_source_errors):
                        log.warning('%s', source_error)
                        if not hasattr(audio_source, 'health'):
                            post_message(AudioWorkerStatus(status='warning', error=source_error))
                    reported_source_errors = source_errors
                    reported_health = _report_health(audio_source, source_errors, reported_health, post_message)
                    if data is None:
                        gap = time.monotonic() - _last_data_time
                        # Check if the audio source signals permanent exhaustion
                        source_exhausted = getattr(audio_source, 'exhausted', False)
                        if source_exhausted:
                            elapsed = time.monotonic() - _worker_start
                            log.error(
                                'Audio stream exhausted after %.1fs (%d samples captured, %d segments transcribed)',
                                elapsed,
                                total_samples_fed,
                                len(all_segments),
                            )
                            post_message(
                                AudioWorkerStatus(
                                    status='error',
                                    error=f'Audio stream stopped unexpectedly after {elapsed:.0f}s',
                                )
                            )
                            break
                        if gap >= 5.0 and time.monotonic() - _silence_logged_at >= 5.0:
                            log.warning(
                                'No audio data for %.1fs (total_samples=%d, segments=%d)',
                                gap,
                                total_samples_fed,
                                len(all_segments),
                            )
                            _silence_logged_at = time.monotonic()
                        continue

                    _last_data_time = time.monotonic()

                    if proc_rec_q is not None:
                        proc_rec_q.put(data)

                    total_samples_fed += len(data)
                    use_case.set_session_offset(total_samples_fed / SAMPLE_RATE)
                    use_case.feed_audio(data)

                    # np.dot avoids a temp data**2 allocation on the ~31 Hz hot path.
                    chunk_rms = math.sqrt(float(np.dot(data, data)) / len(data))
                    _stats_rms_sum += chunk_rms
                    _stats_rms_count += 1
                    _stats_zero_samples += int(np.count_nonzero(data == 0.0))
                    _stats_total_samples += len(data)

                    # An all-zero chunk is silence, not a fault. The Core Audio tap
                    # delivers silence at the full rate, so zeros no longer mean the
                    # capture died. Capture failures surface as the source reporting
                    # `exhausted` after the subprocess exits.

                    _level_accum.append(data)
                    now_abs = time.monotonic()
                    if now_abs - _last_level_post >= 0.1:
                        window = np.concatenate(_level_accum)
                        rms = float(np.sqrt(np.mean(window**2)))
                        if not np.isfinite(rms):
                            rms = 0.0
                        _level_accum.clear()
                        post_message(AudioLevel(rms=rms))
                        _last_level_post = now_abs

                    if not _no_signal_warned and now_abs - _worker_start >= _NO_SIGNAL_WARN_SECONDS:
                        # Hold the warning while the permission dialog is open: the
                        # user is already being asked to grant it.
                        if not getattr(audio_source, 'awaiting_permission', False):
                            _no_signal_warned = True
                            if _no_signal_hint_due(audio_source):
                                message = _no_signal_message()
                                log.warning(message)
                                post_message(AudioWorkerStatus(status='warning', error=message))

                    # Periodic stats every 30s
                    if now_abs - _stats_last_time >= _stats_interval:
                        elapsed = now_abs - _worker_start
                        rms_avg = _stats_rms_sum / _stats_rms_count if _stats_rms_count else 0.0
                        zero_pct = (_stats_zero_samples / _stats_total_samples * 100) if _stats_total_samples else 0.0
                        minutes, secs = divmod(int(elapsed), 60)
                        # Consumer drift: samples the worker has consumed vs samples
                        # that should have been produced if real-time were the only
                        # constraint. Positive drift => the audio_worker is behind
                        # real time (queues somewhere have backlog); zero drift =>
                        # consumer keeping up and any perceived lag is downstream
                        # of the worker (UI, rendering, terminal).
                        expected_samples = int(elapsed * SAMPLE_RATE)
                        drift_samples = expected_samples - total_samples_fed
                        drift_s = drift_samples / SAMPLE_RATE
                        # Queue sizes — only meaningful for MixedAudioSource; gracefully
                        # ignore for any source that doesn't expose these.
                        mic_q = getattr(audio_source, '_mic_q', None)
                        sys_q = getattr(audio_source, '_sys_q', None)
                        sys_buf = getattr(audio_source, '_sys_buf', None)
                        qmic = mic_q.qsize() if mic_q is not None else -1
                        qsys = sys_q.qsize() if sys_q is not None else -1
                        qbuf = len(sys_buf) if sys_buf is not None else -1
                        log.info(
                            'Audio stats [%d:%02d]: %d samples, rms=%.3f, zero=%.1f%%, '
                            'transcriptions=%d, drift=%+.2fs, mic_q=%d, sys_q=%d, sys_buf=%d',
                            minutes,
                            secs,
                            total_samples_fed,
                            rms_avg,
                            zero_pct,
                            _stats_transcriptions,
                            drift_s,
                            qmic,
                            qsys,
                            qbuf,
                        )

                        _stats_last_time = now_abs

                    if use_case.should_trigger() and _transcript_future is None:
                        prepared = use_case.prepare_buffer()
                        if prepared is not None:
                            buf, hints, buf_wall_start, is_first = prepared
                            log.debug('transcription triggered: buffer=%d samples, first_chunk=%s', len(buf), is_first)
                            _stats_transcriptions += 1
                            _pending_meta = (buf_wall_start, is_first)
                            post_message(TranscriptionStatus(active=True))
                            _transcript_future = _executor.submit(
                                transcriber.transcribe,
                                buf,
                                language,
                                hints,
                            )
                except Exception as loop_error:
                    # One bad chunk must not end the meeting recording.
                    iteration_failed = True
                    loop_health.failed(loop_error, reported_health)
                    unsaved.keep_because('the audio loop hit an error')
                    time.sleep(0.05)
                finally:
                    if not iteration_failed:
                        loop_health.succeeded(reported_health)

            # Wait for any in-flight transcription before draining
            if _transcript_future is not None:
                try:
                    _transcript_future.result(timeout=120)
                    _collect_future()
                except Exception as e:
                    log.error('In-flight transcription at shutdown: %s', e, exc_info=True)
                    unsaved.keep_because('a transcription failed')

            # Shutdown drain: read remaining for up to 500ms
            deadline = time.monotonic() + 0.5
            while time.monotonic() < deadline:
                data = audio_source.read(timeout=0.1)
                if data is None:
                    break
                if proc_rec_q is not None:
                    proc_rec_q.put(data)
                total_samples_fed += len(data)
                use_case.feed_audio(data)

            use_case.set_session_offset(total_samples_fed / SAMPLE_RATE)
            post_message(TranscriptionStatus(active=True))
            try:
                flushed = use_case.flush()
            finally:
                post_message(TranscriptionStatus(active=False))
            if flushed:
                all_segments.extend(flushed)
                post_message(TranscriptChunk(segments=flushed))

        finally:
            audio_source.close()
            log.info(
                'Audio source closed: %d total samples, %d segments',
                total_samples_fed,
                len(all_segments),
            )

    except Exception as e:
        log.error('Audio source error: %s', e, exc_info=True)
        unsaved.keep_because('the audio loop failed')
        post_message(AudioWorkerStatus(status='error', error=str(e)))

    # Stop processed audio recorder
    if proc_rec_q is not None:
        proc_rec_q.put(None)
    if proc_rec_writer is not None:
        proc_rec_writer.join(timeout=5)

    _executor.shutdown(wait=False, cancel_futures=True)

    # Release transcriber resources (suppresses C-level teardown noise)
    transcriber.close()

    post_message(AudioWorkerStatus(status='stopped'))
    return all_segments
