"""Thin thread worker shell for audio capture — connects AudioSource to TranscribeAudioUseCase."""

from __future__ import annotations

import queue
import threading
import time
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd

from lazy_take_notes.l1_entities.transcript import TranscriptSegment
from lazy_take_notes.l2_use_cases.transcribe_audio_use_case import (
    SAMPLE_RATE,
    TranscribeAudioUseCase,
)
from lazy_take_notes.l3_interface_adapters.presenters.messages import (
    AudioWorkerStatus,
    TranscriptChunk,
)


def _start_raw_recorder(
    output_dir: Path,
    is_cancelled,
) -> tuple[sd.InputStream, threading.Thread, queue.Queue]:
    """Start a parallel high-quality audio stream that writes to a WAV file."""
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
    whisper_prompt: str = '',
    pause_event: threading.Event | None = None,
    output_dir: Path | None = None,
    save_audio: bool = False,
    transcriber=None,
) -> list[TranscriptSegment]:
    """Audio capture and transcription loop.

    Designed to run inside a Textual @work(thread=True) worker.
    """
    # Load model
    post_message(AudioWorkerStatus(status='loading_model'))
    try:
        if transcriber is None:
            from lazy_take_notes.l3_interface_adapters.gateways.whisper_transcriber import (
                WhisperTranscriber,
            )

            transcriber = WhisperTranscriber()
        transcriber.load_model(model_path)
    except Exception as e:
        post_message(AudioWorkerStatus(status='error', error=f'Failed to load model: {e}'))
        return []
    post_message(AudioWorkerStatus(status='model_ready'))

    use_case = TranscribeAudioUseCase(
        transcriber=transcriber,
        language=language,
        chunk_duration=chunk_duration,
        overlap=overlap,
        silence_threshold=silence_threshold,
        pause_duration=pause_duration,
        whisper_prompt=whisper_prompt,
    )

    audio_q: queue.Queue[np.ndarray] = queue.Queue()
    all_segments: list[TranscriptSegment] = []
    session_start = time.monotonic()

    # Start raw audio recorder if requested
    raw_stream = None
    raw_writer = None
    raw_q = None
    if save_audio and output_dir:
        try:
            raw_stream, raw_writer, raw_q = _start_raw_recorder(output_dir, is_cancelled)
        except Exception:  # noqa: S110 — best-effort; recording continues without raw save
            pass

    # Start recording
    post_message(AudioWorkerStatus(status='recording'))
    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype='float32',
            callback=lambda indata, frames, time_info, status: audio_q.put(indata.copy()),
        ):
            while not is_cancelled():
                # Check pause state
                if pause_event is not None and pause_event.is_set():
                    # Drain queue but discard
                    while not audio_q.empty():
                        try:
                            audio_q.get_nowait()
                        except queue.Empty:
                            break
                    use_case.reset_buffer()
                    time.sleep(0.1)
                    continue

                try:
                    data = audio_q.get(timeout=0.1)
                except queue.Empty:
                    continue

                now = time.monotonic() - session_start
                use_case.set_session_offset(now)
                use_case.feed_audio(data)

                # Drain remaining
                while not audio_q.empty():
                    try:
                        extra = audio_q.get_nowait()
                        use_case.feed_audio(extra)
                    except queue.Empty:
                        break

                if use_case.should_trigger():
                    new_segments = use_case.process_buffer()
                    if new_segments:
                        all_segments.extend(new_segments)
                        post_message(TranscriptChunk(segments=new_segments))

            # Shutdown: drain and flush
            while not audio_q.empty():
                try:
                    extra = audio_q.get_nowait()
                    use_case.feed_audio(extra)
                except queue.Empty:
                    break

            now = time.monotonic() - session_start
            use_case.set_session_offset(now)
            flushed = use_case.flush()
            if flushed:
                all_segments.extend(flushed)
                post_message(TranscriptChunk(segments=flushed))

    except Exception as e:
        post_message(AudioWorkerStatus(status='error', error=str(e)))

    # Stop raw audio recorder
    if raw_stream is not None:
        raw_stream.stop()
        raw_stream.close()
    if raw_q is not None:
        raw_q.put(None)
    if raw_writer is not None:
        raw_writer.join(timeout=5)

    # Release transcriber resources (suppresses C-level teardown noise)
    transcriber.close()

    post_message(AudioWorkerStatus(status='stopped'))
    return all_segments
