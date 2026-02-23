"""Thin thread worker shell for audio capture — connects AudioSource to TranscribeAudioUseCase."""

from __future__ import annotations

import concurrent.futures
import logging
import queue
import threading
import time
import wave
from pathlib import Path
from typing import Any

import numpy as np

from lazy_take_notes.l1_entities.audio_constants import SAMPLE_RATE
from lazy_take_notes.l1_entities.transcript import TranscriptSegment
from lazy_take_notes.l2_use_cases.ports.audio_source import AudioSource
from lazy_take_notes.l2_use_cases.transcribe_audio_use_case import TranscribeAudioUseCase
from lazy_take_notes.l4_frameworks_and_drivers.messages import (
    AudioLevel,
    AudioWorkerStatus,
    TranscriptChunk,
)

log = logging.getLogger('ltn.audio')


def _start_processed_recorder(
    output_dir: Path,
    sample_rate: int,
) -> tuple[queue.Queue, threading.Thread]:
    """Write processed float32 chunks (system/mixed audio) to a WAV file.

    Used for non-mic sources where _start_raw_recorder is not applicable.
    Send None into the returned queue to signal the writer to flush and close.
    """
    wav_path = output_dir / 'recording.wav'
    rec_q: queue.Queue[np.ndarray | None] = queue.Queue()

    def _writer() -> None:
        wf = wave.open(str(wav_path), 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(2)  # int16
        wf.setframerate(sample_rate)
        try:
            while True:
                try:
                    data = rec_q.get(timeout=0.5)
                    if data is None:
                        break
                    pcm = np.clip(data, -1.0, 1.0)
                    wf.writeframes((pcm * 32767).astype(np.int16).tobytes())
                except queue.Empty:  # pragma: no cover -- timing-dependent; queue.get timeout retry
                    pass
        finally:
            wf.close()

    writer = threading.Thread(target=_writer, daemon=True)
    writer.start()
    return rec_q, writer


def _start_raw_recorder(  # pragma: no cover -- requires sounddevice hardware
    output_dir: Path,
    is_cancelled,
) -> tuple[Any, threading.Thread, queue.Queue]:
    """Start a parallel high-quality audio stream that writes to a WAV file."""
    import sounddevice as sd  # noqa: PLC0415 -- deferred: only used for raw WAV recording (mic mode)

    device_info = sd.query_devices(kind='input')
    native_sr = int(device_info['default_samplerate'])

    wav_path = output_dir / 'recording.wav'
    raw_q: queue.Queue[np.ndarray | None] = queue.Queue()

    def _raw_callback(indata, frames, time_info, status):
        raw_q.put(indata.copy())

    stream = sd.InputStream(
        samplerate=native_sr,
        channels=1,
        dtype='int16',
        callback=_raw_callback,
    )

    def _wav_writer():
        wf = wave.open(str(wav_path), 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(native_sr)
        try:
            while True:
                try:
                    data = raw_q.get(timeout=0.5)
                    if data is None:
                        break
                    wf.writeframes(data.tobytes())
                except queue.Empty:
                    if is_cancelled():
                        break
        finally:
            wf.close()

    writer = threading.Thread(target=_wav_writer, daemon=True)
    writer.start()
    stream.start()

    return stream, writer, raw_q


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
    output_dir: Path | None = None,
    save_audio: bool = False,
    transcriber=None,
    audio_source: AudioSource | None = None,
) -> list[TranscriptSegment]:
    """Audio capture and transcription loop.

    Designed to run inside a Textual @work(thread=True) worker.
    """
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
        log.error('Failed to load transcription model: %s', e, exc_info=True)
        post_message(AudioWorkerStatus(status='error', error=f'Failed to load model: {e}'))
        if transcriber is not None:
            transcriber.close()  # clean up any partially-started subprocess
        return []
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
    total_samples_fed: int = 0
    _last_level_post: float = 0.0
    _level_accum: list[np.ndarray] = []

    # Off-thread transcription: audio reading continues while subprocess infers.
    _executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    _transcript_future: concurrent.futures.Future | None = None
    _pending_meta: tuple[float, bool] | None = None  # (buffer_wall_start, is_first_chunk)

    def _collect_future() -> None:
        nonlocal _transcript_future, _pending_meta
        if _transcript_future is None or not _transcript_future.done():
            return
        try:
            raw_segs = _transcript_future.result()
            buf_start, was_first = _pending_meta  # type: ignore[misc]
            new_segs = use_case.apply_result(raw_segs, buf_start, was_first)
            if new_segs:
                all_segments.extend(new_segs)
                post_message(TranscriptChunk(segments=new_segs))
        except Exception as e:
            log.error('Transcription failed: %s', e, exc_info=True)
            post_message(AudioWorkerStatus(status='error', error=str(e)))
        finally:
            _transcript_future = None
            _pending_meta = None

    # Raw WAV recorder only makes sense for mic-based sources
    from lazy_take_notes.l3_interface_adapters.gateways.sounddevice_audio_source import (  # noqa: PLC0415 -- deferred: sounddevice loaded only when worker starts
        SounddeviceAudioSource,
    )

    raw_stream: Any = None
    raw_writer: threading.Thread | None = None
    raw_q: queue.Queue | None = None
    proc_rec_q: queue.Queue | None = None
    proc_rec_writer: threading.Thread | None = None
    if save_audio and output_dir:
        if isinstance(audio_source, SounddeviceAudioSource):  # pragma: no cover -- only fires for real mic source
            # Mic-only: capture a separate high-quality stream at native sample rate.
            try:
                raw_stream, raw_writer, raw_q = _start_raw_recorder(output_dir, is_cancelled)
            except Exception:  # noqa: S110 — best-effort; recording continues without raw save
                pass
        else:
            # Mixed or system-only: save the processed SAMPLE_RATE audio from read().
            try:
                proc_rec_q, proc_rec_writer = _start_processed_recorder(output_dir, SAMPLE_RATE)
            except Exception:  # noqa: S110 -- best-effort; recording continues without raw save  # pragma: no cover
                pass

    # Start recording
    post_message(AudioWorkerStatus(status='recording'))
    try:
        audio_source.open(SAMPLE_RATE, 1)
        try:
            while not is_cancelled():
                if pause_event is not None and pause_event.is_set():
                    use_case.reset_buffer()
                    _collect_future()
                    time.sleep(0.1)
                    continue

                _collect_future()

                data = audio_source.read(timeout=0.1)
                if data is None:
                    continue

                if proc_rec_q is not None:
                    proc_rec_q.put(data)

                total_samples_fed += len(data)
                use_case.set_session_offset(total_samples_fed / SAMPLE_RATE)
                use_case.feed_audio(data)

                _level_accum.append(data)
                now_abs = time.monotonic()
                if now_abs - _last_level_post >= 0.1:
                    window = np.concatenate(_level_accum)
                    rms = float(np.sqrt(np.mean(window**2)))
                    _level_accum.clear()
                    post_message(AudioLevel(rms=rms))
                    _last_level_post = now_abs

                if use_case.should_trigger() and _transcript_future is None:
                    prepared = use_case.prepare_buffer()
                    if prepared is not None:
                        buf, hints, buf_wall_start, is_first = prepared
                        _pending_meta = (buf_wall_start, is_first)
                        _transcript_future = _executor.submit(
                            transcriber.transcribe,
                            buf,
                            language,
                            hints,
                        )

            # Wait for any in-flight transcription before draining
            if _transcript_future is not None:
                try:
                    _transcript_future.result(timeout=120)
                    _collect_future()
                except Exception as e:
                    log.error('In-flight transcription at shutdown: %s', e, exc_info=True)

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
            flushed = use_case.flush()
            if flushed:
                all_segments.extend(flushed)
                post_message(TranscriptChunk(segments=flushed))

        finally:
            audio_source.close()

    except Exception as e:
        log.error('Audio source error: %s', e, exc_info=True)
        post_message(AudioWorkerStatus(status='error', error=str(e)))

    # Stop raw audio recorder (mic mode)
    if raw_stream is not None:  # pragma: no cover -- only fires for real mic source
        raw_stream.stop()
        raw_stream.close()
    if raw_q is not None:  # pragma: no cover -- only fires for real mic source
        raw_q.put(None)
    if raw_writer is not None:  # pragma: no cover -- only fires for real mic source
        raw_writer.join(timeout=5)

    # Stop processed audio recorder (mixed / system mode)
    if proc_rec_q is not None:
        proc_rec_q.put(None)
    if proc_rec_writer is not None:
        proc_rec_writer.join(timeout=5)

    _executor.shutdown(wait=True)

    # Release transcriber resources (suppresses C-level teardown noise)
    transcriber.close()

    post_message(AudioWorkerStatus(status='stopped'))
    return all_segments
