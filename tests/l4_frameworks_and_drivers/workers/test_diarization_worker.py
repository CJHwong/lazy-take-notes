"""Tests for the live diarization worker."""

from __future__ import annotations

from pathlib import Path

from lazy_take_notes.l1_entities.speaker_turn import SpeakerTurn
from lazy_take_notes.l4_frameworks_and_drivers.messages import SpeakerTurns
from lazy_take_notes.l4_frameworks_and_drivers.workers.diarization_worker import run_diarization_worker


class _FakeDiarizer:
    def __init__(self, turns: list[SpeakerTurn]) -> None:
        self.turns = turns
        self.calls = 0

    def diarize(self, audio_path: Path, timeout: float = 300.0) -> list[SpeakerTurn]:
        self.calls += 1
        return self.turns


class _FailingDiarizer:
    def __init__(self) -> None:
        self.calls = 0

    def diarize(self, audio_path: Path, timeout: float = 300.0) -> list[SpeakerTurn]:
        self.calls += 1
        raise RuntimeError('model missing')


def _canceller(limit: int):
    calls = 0

    def is_cancelled() -> bool:
        nonlocal calls
        calls += 1
        return calls > limit

    return is_cancelled


class TestDiarizationWorker:
    def test_posts_turns_for_growing_audio(self, tmp_path):
        audio = tmp_path / 'recording.wav'
        audio.write_bytes(b'\0' * 4096)
        messages: list = []

        run_diarization_worker(
            messages.append,
            _canceller(3),
            audio,
            _FakeDiarizer([SpeakerTurn(0.0, 2.0, 1)]),
            interval_s=0.0,
        )

        posted = [m for m in messages if isinstance(m, SpeakerTurns)]
        assert len(posted) == 1
        assert posted[0].turns == [SpeakerTurn(0.0, 2.0, 1)]

    def test_skips_a_pass_when_no_new_audio_arrived(self, tmp_path):
        audio = tmp_path / 'recording.wav'
        audio.write_bytes(b'\0' * 4096)
        diarizer = _FakeDiarizer([SpeakerTurn(0.0, 2.0, 1)])
        messages: list = []

        # Five loop turns at the same file size must produce one pass.
        run_diarization_worker(messages.append, _canceller(5), audio, diarizer, interval_s=0.0)

        assert diarizer.calls == 1

    def test_a_failed_pass_does_not_stop_the_worker(self, tmp_path):
        audio = tmp_path / 'recording.wav'
        audio.write_bytes(b'\0' * 4096)
        diarizer = _FailingDiarizer()
        messages: list = []

        run_diarization_worker(messages.append, _canceller(3), audio, diarizer, interval_s=0.0)

        assert diarizer.calls == 1
        assert [m for m in messages if isinstance(m, SpeakerTurns)] == []

    def test_waits_for_the_recording_to_appear(self, tmp_path):
        diarizer = _FakeDiarizer([])
        messages: list = []

        run_diarization_worker(
            messages.append,
            _canceller(1),
            tmp_path / 'missing.wav',
            diarizer,
            interval_s=0.0,
        )

        assert diarizer.calls == 0
