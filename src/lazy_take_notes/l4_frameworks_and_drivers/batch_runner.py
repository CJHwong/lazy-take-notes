"""Batch runner — headless transcribe-from-file and single final digest."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from lazy_take_notes.l1_entities.audio_constants import SAMPLE_RATE
from lazy_take_notes.l1_entities.config import AppConfig
from lazy_take_notes.l1_entities.digest_state import DigestState
from lazy_take_notes.l1_entities.template import SessionTemplate
from lazy_take_notes.l2_use_cases.digest_use_case import RunDigestUseCase
from lazy_take_notes.l2_use_cases.transcribe_audio_use_case import TranscribeAudioUseCase
from lazy_take_notes.l3_interface_adapters.gateways.audio_file_loader import load_audio_file
from lazy_take_notes.l3_interface_adapters.gateways.hf_model_resolver import HfModelResolver
from lazy_take_notes.l4_frameworks_and_drivers.container import DependencyContainer
from lazy_take_notes.l4_frameworks_and_drivers.infra_config import InfraConfig

_FEED_CHUNK = SAMPLE_RATE  # 1 second of audio per feed call


def _fmt(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f'{h:02d}:{m:02d}:{s:02d}'


def _err(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def run_batch(
    audio_path: Path,
    config: AppConfig,
    template: SessionTemplate,
    out_dir: Path,
    infra: InfraConfig,
) -> None:
    """Load *audio_path*, transcribe in chunks, run one final digest. Blocks until done."""

    # -- Load audio --
    _err(f'Loading audio: {audio_path}')
    try:
        audio = load_audio_file(audio_path)
    except (FileNotFoundError, RuntimeError) as exc:
        _err(f'Error: {exc}')
        raise SystemExit(1) from exc

    duration = len(audio) / SAMPLE_RATE
    _err(f'Duration: {_fmt(duration)}  ({len(audio):,} samples @ {SAMPLE_RATE} Hz)')

    # -- Build container (no audio source needed for batch) --
    container = DependencyContainer(config, template, out_dir, infra, audio_mode=None)

    # -- Resolve and load whisper model --
    locale = template.metadata.locale
    language = locale.split('-')[0].lower()
    model_name = config.transcription.model_for_locale(locale)
    _err(f'Whisper model: {model_name}')

    def _on_progress(percent: int) -> None:
        _err(f'  Downloading {model_name}: {percent}%')

    try:
        model_path = HfModelResolver(on_progress=_on_progress).resolve(model_name)
    except Exception as exc:
        _err(f'Error resolving model: {exc}')
        raise SystemExit(1) from exc

    try:
        container.transcriber.load_model(model_path)
    except Exception as exc:
        _err(f'Error loading model: {exc}')
        raise SystemExit(1) from exc

    # -- Transcribe in chunks --
    tc = config.transcription
    use_case = TranscribeAudioUseCase(
        transcriber=container.transcriber,
        language=language,
        chunk_duration=tc.chunk_duration,
        overlap=tc.overlap,
        silence_threshold=tc.silence_threshold,
        pause_duration=tc.pause_duration,
        whisper_prompt=template.whisper_prompt,
    )
    all_segments = []

    _err('Transcribing...')
    offset = 0
    while offset < len(audio):
        chunk = audio[offset : offset + _FEED_CHUNK]
        offset += len(chunk)
        use_case.feed_audio(chunk)
        use_case.set_session_offset(offset / SAMPLE_RATE)

        if use_case.should_trigger():
            segments = use_case.process_buffer()
            if segments:
                all_segments.extend(segments)
                container.persistence.save_transcript_lines(segments)
                for seg in segments:
                    _err(f'  [{_fmt(seg.wall_start)}] {seg.text}')

    tail = use_case.flush()
    if tail:
        all_segments.extend(tail)
        container.persistence.save_transcript_lines(tail)
        for seg in tail:
            _err(f'  [{_fmt(seg.wall_start)}] {seg.text}')

    container.transcriber.close()

    if not all_segments:
        _err('No speech detected in the audio file.')
        raise SystemExit(0)

    _err(f'\nTranscription complete — {len(all_segments)} segments.')

    # -- Single final digest --
    _err('Running digest...')
    full_transcript = '\n'.join(seg.text for seg in all_segments)
    state = DigestState()
    state.init_messages(template.system_prompt)
    state.buffer.extend(full_transcript.splitlines())

    result = asyncio.run(
        RunDigestUseCase(container.llm_client).execute(
            state=state,
            model=config.digest.model,
            template=template,
            is_final=True,
            full_transcript=full_transcript,
        )
    )

    digest_markdown = result.data
    if digest_markdown is None:
        _err(f'Digest failed: {result.error}')
        raise SystemExit(1)

    digest_path = container.persistence.save_digest_md(digest_markdown, state.digest_count)
    container.persistence.save_history(digest_markdown, state.digest_count, is_final=True)
    transcript_path = out_dir / 'transcript_raw.txt'
    _err(f'\nSaved:\n  Transcript: {transcript_path}\n  Digest:     {digest_path}\n')
    print(digest_markdown)
