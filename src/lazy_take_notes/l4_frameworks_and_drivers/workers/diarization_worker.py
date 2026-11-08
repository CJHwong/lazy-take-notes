"""Background worker: re-run diarization over the recording as it grows."""

from __future__ import annotations

import logging
import time
from pathlib import Path

from lazy_take_notes.l2_use_cases.ports.diarizer import Diarizer
from lazy_take_notes.l4_frameworks_and_drivers.messages import SpeakerTurns

log = logging.getLogger('ltn.diarize')


def run_diarization_worker(
    post_message,
    is_cancelled,
    audio_path: Path,
    diarizer: Diarizer,
    interval_s: float = 60.0,
    timeout_s: float = 300.0,
) -> None:
    """One full diarization pass per interval, over the audio recorded so far.

    Speaker numbers follow arrival order, so labels already on screen stay
    valid when a new voice joins. A pass is skipped while no new audio arrived.
    """
    processed_size = -1
    while not is_cancelled():
        if audio_path.is_file():
            size = audio_path.stat().st_size
            if size > processed_size + 1024:
                try:
                    turns = diarizer.diarize(audio_path, timeout=timeout_s)
                except RuntimeError as exc:
                    log.warning('diarization pass failed: %s', exc)
                else:
                    if turns:
                        speakers = len({turn.speaker for turn in turns})
                        log.info('diarization pass: %d turns, %d speakers', len(turns), speakers)
                        post_message(SpeakerTurns(turns=turns))
                processed_size = size

        deadline = time.monotonic() + interval_s
        while not is_cancelled() and time.monotonic() < deadline:
            time.sleep(0.2)
