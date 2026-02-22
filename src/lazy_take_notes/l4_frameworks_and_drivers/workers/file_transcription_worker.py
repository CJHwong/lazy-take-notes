"""File transcription worker — thread body for processing an audio file."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from lazy_take_notes.l1_entities.audio_constants import SAMPLE_RATE
from lazy_take_notes.l1_entities.transcript import TranscriptSegment
from lazy_take_notes.l2_use_cases.ports.transcriber import Transcriber
from lazy_take_notes.l2_use_cases.transcribe_audio_use_case import TranscribeAudioUseCase
from lazy_take_notes.l4_frameworks_and_drivers.messages import (
    AudioWorkerStatus,
    ModelDownloadProgress,
    TranscriptChunk,
)

log = logging.getLogger('ltn.file_worker')

_FEED_CHUNK = SAMPLE_RATE  # 1 second of audio per feed call


def run_file_transcription(
    post_message: Callable,
    is_cancelled: Callable[[], bool],
    audio_path: Path,
    model_name: str,
    language: str,
    chunk_duration: float,
    overlap: float,
    silence_threshold: float,
    pause_duration: float,
    recognition_hints: list[str] | None = None,
    transcriber: Transcriber | None = None,
) -> list[TranscriptSegment]:
    """Transcribe an audio file in chunks, posting messages for TUI updates.

    Designed to run inside a Textual @work(thread=True) worker.
    """
    from lazy_take_notes.l3_interface_adapters.gateways.audio_file_loader import (  # noqa: PLC0415 -- deferred: only loaded when worker starts
        load_audio_file,
    )
    from lazy_take_notes.l3_interface_adapters.gateways.hf_model_resolver import (  # noqa: PLC0415 -- deferred: only loaded when worker starts
        HfModelResolver,
    )

    # Load audio file
    try:
        audio = load_audio_file(audio_path)
    except (FileNotFoundError, RuntimeError) as exc:
        log.error('Failed to load audio file: %s', exc, exc_info=True)
        post_message(AudioWorkerStatus(status='error', error=str(exc)))
        return []

    # Resolve whisper model (posts download progress)
    def _on_progress(percent: int) -> None:
        post_message(ModelDownloadProgress(percent=percent, model_name=model_name))

    post_message(AudioWorkerStatus(status='loading_model'))
    try:
        resolver = HfModelResolver(on_progress=_on_progress)
        model_path = resolver.resolve(model_name)
    except Exception as exc:
        log.error('Failed to resolve model: %s', exc, exc_info=True)
        post_message(AudioWorkerStatus(status='error', error=str(exc)))
        return []

    # Load model into transcriber
    if transcriber is None:  # pragma: no cover -- default wiring; transcriber always injected in tests
        from lazy_take_notes.l3_interface_adapters.gateways.subprocess_whisper_transcriber import (  # noqa: PLC0415 -- deferred: subprocess spawned only when worker starts
            SubprocessWhisperTranscriber,
        )

        transcriber = SubprocessWhisperTranscriber()

    try:
        transcriber.load_model(model_path)
    except Exception as exc:
        log.error('Failed to load model: %s', exc, exc_info=True)
        post_message(AudioWorkerStatus(status='error', error=str(exc)))
        transcriber.close()
        return []
    post_message(AudioWorkerStatus(status='model_ready'))

    # Transcribe
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
    post_message(AudioWorkerStatus(status='recording'))

    offset = 0
    while offset < len(audio) and not is_cancelled():
        chunk = audio[offset : offset + _FEED_CHUNK]
        offset += len(chunk)
        use_case.feed_audio(chunk)
        use_case.set_session_offset(offset / SAMPLE_RATE)

        if use_case.should_trigger():
            segments = use_case.process_buffer()
            if segments:
                all_segments.extend(segments)
                post_message(TranscriptChunk(segments=segments))

    # Flush remaining audio
    if not is_cancelled():
        tail = use_case.flush()
        if tail:
            all_segments.extend(tail)
            post_message(TranscriptChunk(segments=tail))

    transcriber.close()
    post_message(AudioWorkerStatus(status='stopped'))
    return all_segments
