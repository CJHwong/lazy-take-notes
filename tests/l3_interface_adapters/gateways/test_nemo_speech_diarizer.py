"""Tests for the nemo-speech diarization gateway."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from lazy_take_notes.l1_entities.speaker_turn import SpeakerTurn
from lazy_take_notes.l3_interface_adapters.gateways.nemo_speech_diarizer import NemoSpeechDiarizer


def _fake_release(tmp_path: Path) -> tuple[Path, Path, Path]:
    binary = tmp_path / 'nemo-speech'
    binary.write_bytes(b'')
    model = tmp_path / 'model.gguf'
    model.write_bytes(b'')
    audio = tmp_path / 'recording.wav'
    audio.write_bytes(b'')
    return binary, model, audio


def _write_output(command: list[str]) -> None:
    """Emulate the CLI: write a two-turn JSON to the --output path."""
    output = Path(command[command.index('--output') + 1])
    output.write_text(
        json.dumps(
            {
                'segments': [
                    {'start': 0.0, 'end': 2.5, 'speaker': 1},
                    {'start': 3.0, 'end': 4.0, 'speaker': 2},
                ]
            }
        ),
        encoding='utf-8',
    )


class TestNemoSpeechDiarizer:
    def test_parses_turns(self, tmp_path):
        binary, model, audio = _fake_release(tmp_path)

        def fake_run(command, **kwargs):
            _write_output(command)
            return subprocess.CompletedProcess(command, 0, '', '')

        with patch('subprocess.run', side_effect=fake_run):
            turns = NemoSpeechDiarizer(binary, model).diarize(audio)

        assert turns == [SpeakerTurn(start=0.0, end=2.5, speaker=1), SpeakerTurn(start=3.0, end=4.0, speaker=2)]

    def test_builds_the_expected_command(self, tmp_path):
        binary, model, audio = _fake_release(tmp_path)

        def fake_run(command, **kwargs):
            _write_output(command)
            return subprocess.CompletedProcess(command, 0, '', '')

        with patch('subprocess.run', side_effect=fake_run) as mock_run:
            NemoSpeechDiarizer(binary, model, device='cpu').diarize(audio)

        command = mock_run.call_args.args[0]
        assert command[0] == str(binary)
        assert command[1] == 'diarize'
        assert command[2] == str(audio)
        assert command[command.index('--model') + 1] == str(model)
        assert command[command.index('--device') + 1] == 'cpu'
        assert command[command.index('--preset') + 1] == 'v3-offline'
        assert mock_run.call_args.kwargs['timeout'] == 300.0

    def test_missing_binary_is_reported(self, tmp_path, monkeypatch):
        monkeypatch.delenv('LTN_NEMO_BINARY', raising=False)
        _, model, audio = _fake_release(tmp_path)

        with pytest.raises(RuntimeError, match='LTN_NEMO_BINARY'):
            NemoSpeechDiarizer('', model).diarize(audio)

    def test_missing_model_is_reported(self, tmp_path, monkeypatch):
        monkeypatch.delenv('LTN_NEMO_MODEL', raising=False)
        binary, _, audio = _fake_release(tmp_path)

        with pytest.raises(RuntimeError, match='LTN_NEMO_MODEL'):
            NemoSpeechDiarizer(binary, '').diarize(audio)

    def test_reads_paths_from_the_environment(self, tmp_path, monkeypatch):
        binary, model, audio = _fake_release(tmp_path)
        monkeypatch.setenv('LTN_NEMO_BINARY', str(binary))
        monkeypatch.setenv('LTN_NEMO_MODEL', str(model))

        def fake_run(command, **kwargs):
            _write_output(command)
            return subprocess.CompletedProcess(command, 0, '', '')

        with patch('subprocess.run', side_effect=fake_run):
            turns = NemoSpeechDiarizer().diarize(audio)

        assert len(turns) == 2

    def test_failure_reports_the_stderr(self, tmp_path):
        binary, model, audio = _fake_release(tmp_path)

        with patch(
            'subprocess.run',
            return_value=subprocess.CompletedProcess([], 2, '', 'unsupported runtime version'),
        ):
            with pytest.raises(RuntimeError, match='unsupported runtime version'):
                NemoSpeechDiarizer(binary, model).diarize(audio)

    def test_timeout_is_reported(self, tmp_path):
        binary, model, audio = _fake_release(tmp_path)

        with patch('subprocess.run', side_effect=subprocess.TimeoutExpired([], 5)):
            with pytest.raises(RuntimeError, match='timed out'):
                NemoSpeechDiarizer(binary, model).diarize(audio, timeout=5)
