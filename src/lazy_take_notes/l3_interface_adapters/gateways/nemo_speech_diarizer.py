"""Gateway: speaker diarization through the native nemo-speech CLI.

The runtime and the model are external files, not Python packages: a 7 MB
native binary and a 107 MB Q8_0 GGUF. Point LTN_NEMO_BINARY and
LTN_NEMO_MODEL at them.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess  # noqa: S404 -- intentional: fixed arg list, not shell=True
import tempfile
from pathlib import Path

from lazy_take_notes.l1_entities.speaker_turn import SpeakerTurn

log = logging.getLogger('ltn.diarize')

DEFAULT_PRESET = 'v3-offline'


class NemoSpeechDiarizer:
    """Implements Diarizer by running `nemo-speech diarize` on a WAV file."""

    def __init__(
        self,
        binary: str | Path | None = None,
        model: str | Path | None = None,
        *,
        device: str = 'auto',
        preset: str = DEFAULT_PRESET,
    ) -> None:
        self._binary = binary if binary is not None else os.environ.get('LTN_NEMO_BINARY', '')
        self._model = model if model is not None else os.environ.get('LTN_NEMO_MODEL', '')
        self._device = device
        self._preset = preset

    def _resolve(self) -> tuple[Path, Path]:
        binary = Path(str(self._binary)).expanduser()
        model = Path(str(self._model)).expanduser()
        if not self._binary or not binary.is_file():
            raise RuntimeError('nemo-speech binary not found. Set LTN_NEMO_BINARY to the executable path.')
        if not self._model or not model.is_file():
            raise RuntimeError('diarization model not found. Set LTN_NEMO_MODEL to the GGUF path.')
        return binary, model

    def diarize(self, audio_path: Path, timeout: float = 300.0) -> list[SpeakerTurn]:
        binary, model = self._resolve()
        if not audio_path.is_file():
            raise RuntimeError(f'audio file not found: {audio_path}')

        with tempfile.TemporaryDirectory() as workdir:
            output = Path(workdir) / 'turns.json'
            command = [
                str(binary),
                'diarize',
                str(audio_path),
                '--model',
                str(model),
                '--device',
                self._device,
                '--preset',
                self._preset,
                '--format',
                'json',
                '--output',
                str(output),
            ]
            log.info('diarizing %s (device=%s, preset=%s)', audio_path, self._device, self._preset)
            try:
                completed = subprocess.run(  # noqa: S603 -- fixed arg list, not shell=True
                    command,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(f'diarization timed out after {timeout:.0f}s') from exc

            if completed.returncode != 0:
                detail = (completed.stderr or completed.stdout).strip()
                raise RuntimeError(f'nemo-speech failed with code {completed.returncode}: {detail}')
            if not output.is_file():
                raise RuntimeError('nemo-speech produced no output file')
            payload = json.loads(output.read_text(encoding='utf-8'))

        return [
            SpeakerTurn(start=float(item['start']), end=float(item['end']), speaker=int(item['speaker']))
            for item in payload.get('segments', [])
        ]
