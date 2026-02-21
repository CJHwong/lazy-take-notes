"""Gateway: whisper.cpp transcriber — implements Transcriber port."""

from __future__ import annotations

import contextlib
import os

import numpy as np
from pywhispercpp.model import Model

from lazy_take_notes.l1_entities.transcript import TranscriptSegment


@contextlib.contextmanager
def _suppress_c_stdout():
    """Redirect C-level stdout and stderr to /dev/null.

    whisper.cpp prints init/progress messages directly via C fprintf,
    bypassing Python's sys.stdout. This corrupts the TUI.
    """
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stdout = os.dup(1)
    old_stderr = os.dup(2)
    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(old_stdout, 1)
        os.dup2(old_stderr, 2)
        os.close(devnull)
        os.close(old_stdout)
        os.close(old_stderr)


class WhisperTranscriber:
    """pywhispercpp adapter. Handles model loading, C stdout suppression,
    and centisecond-to-seconds conversion."""

    def __init__(self) -> None:
        self._model: Model | None = None

    def close(self) -> None:
        """Explicitly release the model, suppressing C-level teardown noise."""
        if self._model is not None:
            with _suppress_c_stdout():
                del self._model
                self._model = None

    def load_model(self, model_path: str) -> None:
        with _suppress_c_stdout():
            self._model = Model(model_path, print_progress=False, print_realtime=False)

    def transcribe(
        self,
        audio: np.ndarray,
        language: str,
        hints: list[str] | None = None,
    ) -> list[TranscriptSegment]:
        if self._model is None:
            raise RuntimeError('Model not loaded. Call load_model() first.')

        kwargs: dict = {'language': language}
        if hints:
            kwargs['initial_prompt'] = ' '.join(hints)

        with _suppress_c_stdout():
            raw_segments = self._model.transcribe(audio, **kwargs)

        result: list[TranscriptSegment] = []
        for seg in raw_segments:
            text = seg.text.strip()
            if text:
                result.append(
                    TranscriptSegment(
                        text=text,
                        wall_start=seg.t0 / 100.0,
                        wall_end=seg.t1 / 100.0,
                    )
                )
        return result
