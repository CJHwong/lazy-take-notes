"""Thin thread worker shell for audio capture — connects AudioSource to TranscribeAudioUseCase."""

from __future__ import annotations

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


def _start_raw_recorder(
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
    whisper_prompt: str = '',
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
        if transcriber is None:
            from lazy_take_notes.l3_interface_adapters.gateways.whisper_transcriber import (  # noqa: PLC0415 -- deferred: whisper.cpp loaded only when worker starts
                WhisperTranscriber,
            )

            transcriber = WhisperTranscriber()
        transcriber.load_model(model_path)
    except Exception as e:
        post_message(AudioWorkerStatus(status='error', error=f'Failed to load model: {e}'))
        return []
    post_message(AudioWorkerStatus(status='model_ready'))

    # Resolve audio source — default to SounddeviceAudioSource when not injected
    if audio_source is None:
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
        whisper_prompt=whisper_prompt,
    )

    all_segments: list[TranscriptSegment] = []
    total_samples_fed: int = 0
    _last_level_post: float = 0.0

    # Raw WAV recorder only makes sense for mic-based sources
    from lazy_take_notes.l3_interface_adapters.gateways.sounddevice_audio_source import (  # noqa: PLC0415 -- deferred: sounddevice loaded only when worker starts
        SounddeviceAudioSource,
    )

    raw_stream: Any = None
    raw_writer: threading.Thread | None = None
    raw_q: queue.Queue | None = None
    if save_audio and output_dir and isinstance(audio_source, SounddeviceAudioSource):
        try:
            raw_stream, raw_writer, raw_q = _start_raw_recorder(output_dir, is_cancelled)
        except Exception:  # noqa: S110 — best-effort; recording continues without raw save
            pass

    # Start recording
    post_message(AudioWorkerStatus(status='recording'))
    try:
        audio_source.open(SAMPLE_RATE, 1)
        try:
            while not is_cancelled():
                if pause_event is not None and pause_event.is_set():
                    use_case.reset_buffer()
                    time.sleep(0.1)
                    continue

                data = audio_source.read(timeout=0.1)
                if data is None:
                    continue

                total_samples_fed += len(data)
                use_case.set_session_offset(total_samples_fed / SAMPLE_RATE)
                use_case.feed_audio(data)

                now_abs = time.monotonic()
                if now_abs - _last_level_post >= 0.1:
                    rms = float(np.sqrt(np.mean(data**2)))
                    post_message(AudioLevel(rms=rms))
                    _last_level_post = now_abs

                if use_case.should_trigger():
                    new_segments = use_case.process_buffer()
                    if new_segments:
                        all_segments.extend(new_segments)
                        post_message(TranscriptChunk(segments=new_segments))

            # Shutdown drain: read remaining for up to 500ms
            deadline = time.monotonic() + 0.5
            while time.monotonic() < deadline:
                data = audio_source.read(timeout=0.1)
                if data is None:
                    break
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
