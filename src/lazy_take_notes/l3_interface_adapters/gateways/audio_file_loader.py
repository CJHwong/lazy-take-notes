"""Gateway: audio file loader — reads any audio format via ffmpeg subprocess."""

from __future__ import annotations

import shutil
import subprocess  # noqa: S404 -- intentional: shells out to ffmpeg with a fixed arg list, not shell=True
from pathlib import Path

import numpy as np

from lazy_take_notes.l1_entities.audio_constants import SAMPLE_RATE

_FFMPEG_TIMEOUT = 300  # seconds


def load_audio_file(path: Path) -> np.ndarray:
    """Load *path* using ffmpeg, returning float32 mono PCM at 16 kHz.

    Supports any format ffmpeg can decode: WAV, FLAC, MP3, M4A, OGG, MP4, etc.

    Raises:
        FileNotFoundError: audio file does not exist.
        RuntimeError: ffmpeg is missing, conversion failed, timed out, or
                      the file contains no decodable audio.
    """
    if not path.exists():
        raise FileNotFoundError(f'Audio file not found: {path}')

    if shutil.which('ffmpeg') is None:
        raise RuntimeError(
            'ffmpeg is required but not found on PATH.\n  macOS:  brew install ffmpeg\n  Debian: apt install ffmpeg'
        )

    cmd = [
        'ffmpeg',
        '-i',
        str(path),
        '-ar',
        str(SAMPLE_RATE),
        '-ac',
        '1',
        '-f',
        'f32le',
        '-v',
        'quiet',
        'pipe:1',
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, timeout=_FFMPEG_TIMEOUT)  # noqa: S603
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f'ffmpeg timed out after {_FFMPEG_TIMEOUT}s processing: {path}') from exc
    except OSError as exc:
        raise RuntimeError(f'Failed to launch ffmpeg: {exc}') from exc

    if result.returncode != 0:
        stderr = result.stderr.decode('utf-8', errors='replace').strip()
        raise RuntimeError(f'ffmpeg exited with code {result.returncode} for: {path}\n{stderr}')

    if not result.stdout:
        raise RuntimeError(f'ffmpeg produced no audio output for: {path}')

    audio = np.frombuffer(result.stdout, dtype=np.float32)
    if len(audio) == 0:
        raise RuntimeError(f'Audio file appears to be empty: {path}')

    return audio
