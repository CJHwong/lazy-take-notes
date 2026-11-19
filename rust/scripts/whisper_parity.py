"""Check that the Rust whisper child transcribes like the Python one.

Transcribes the start of one audio file the way the file worker does (one
second per feed, a chunk when the use case triggers, a flush at the end), with
a template's recognition hints, in both versions. Compares every segment's text
and times, and prints the time each spent and the time `close()` took.

Run from the repo root after `cargo build --release --bin lazy-take-notes --example cross_check`:
  uv run python rust/scripts/whisper_parity.py MODEL AUDIO LANGUAGE SECONDS TEMPLATE
"""

from __future__ import annotations

import json
import subprocess  # noqa: S404 -- fixed argument list, no shell
import sys
import time
from pathlib import Path

import yaml

from lazy_take_notes.l1_entities.audio_constants import SAMPLE_RATE
from lazy_take_notes.l2_use_cases.transcribe_audio_use_case import TranscribeAudioUseCase
from lazy_take_notes.l3_interface_adapters.gateways.audio_file_loader import load_audio_file
from lazy_take_notes.l3_interface_adapters.gateways.subprocess_whisper_transcriber import SubprocessWhisperTranscriber

RELEASE = Path(__file__).resolve().parent.parent / 'target' / 'release'


def python_run(model: str, audio_path: str, language: str, seconds: int, hints: list[str]) -> dict:
    audio = load_audio_file(Path(audio_path))[: seconds * SAMPLE_RATE]
    whisper = SubprocessWhisperTranscriber()
    whisper.load_model(model)
    use_case = TranscribeAudioUseCase(transcriber=whisper, language=language, recognition_hints=hints)
    segments = []
    started = time.monotonic()
    offset = 0
    while offset < len(audio):
        chunk = audio[offset : offset + SAMPLE_RATE]
        offset += len(chunk)
        use_case.feed_audio(chunk)
        use_case.set_session_offset(offset / SAMPLE_RATE)
        if use_case.should_trigger():
            segments.extend(use_case.process_buffer())
    segments.extend(use_case.flush())
    total = time.monotonic() - started
    started = time.monotonic()
    whisper.close()
    return {
        'segments': [s.model_dump() for s in segments],
        'total_s': total,
        'close_s': time.monotonic() - started,
    }


def rust_run(model: str, audio_path: str, language: str, seconds: int, template: str) -> dict:
    command = [str(RELEASE / 'examples' / 'cross_check'), 'transcribe-file', str(RELEASE / 'lazy-take-notes')]
    result = subprocess.run(  # noqa: S603 -- fixed argument list
        [*command, model, audio_path, language, str(seconds), template],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def main() -> None:
    model, audio_path, language, seconds, template = sys.argv[1:6]
    hints = (yaml.safe_load(Path(template).read_text(encoding='utf-8')) or {}).get('recognition_hints') or []
    python = python_run(model, audio_path, language, int(seconds), hints)
    rust = rust_run(model, audio_path, language, int(seconds), template)
    key = ('text', 'wall_start', 'wall_end')
    pairs = list(zip(python['segments'], rust['segments'], strict=False))
    same = sum(1 for a, b in pairs if [a[k] for k in key] == [b[k] for k in key])
    count = (len(python['segments']), len(rust['segments']))
    print(f'segments: python {count[0]}, rust {count[1]}, identical {same}')
    for name in ('python', 'rust'):
        run = python if name == 'python' else rust
        print(f'{name}: transcribe {run["total_s"]:.1f}s, close {run["close_s"]:.2f}s')
    for a, b in pairs:
        if [a[k] for k in key] != [b[k] for k in key]:
            print(f'  python: {a}\n  rust:   {b}')
    raise SystemExit(0 if same == count[0] == count[1] else 1)


if __name__ == '__main__':
    main()
