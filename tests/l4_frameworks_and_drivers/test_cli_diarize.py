"""Tests for the diarize CLI command."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from lazy_take_notes.l1_entities.speaker_turn import SpeakerTurn
from lazy_take_notes.l4_frameworks_and_drivers.cli import cli

_DIARIZER = 'lazy_take_notes.l3_interface_adapters.gateways.nemo_speech_diarizer.NemoSpeechDiarizer'


def _make_session(tmp_path: Path, transcript: str) -> Path:
    session = tmp_path / '2026-11-08_100000_test'
    session.mkdir()
    (session / 'recording.wav').write_bytes(b'')
    (session / 'transcript.txt').write_text(transcript, encoding='utf-8')
    return session


class TestDiarizeCommand:
    def test_labels_the_transcript(self, tmp_path):
        session = _make_session(tmp_path, '[00:00:01] first line\n[00:00:03] second line\n[00:00:09] silence\n')

        with patch(_DIARIZER) as diarizer_class:
            diarizer_class.return_value.diarize.return_value = [
                SpeakerTurn(0.0, 2.0, 1),
                SpeakerTurn(2.5, 5.0, 2),
            ]
            result = CliRunner().invoke(cli, ['diarize', str(session)])

        assert result.exit_code == 0, result.output
        labeled = (session / 'transcript_speakers.txt').read_text(encoding='utf-8')
        assert '[00:00:01] S1: first line' in labeled
        assert '[00:00:03] S2: second line' in labeled
        assert '[00:00:09] silence' in labeled
        assert '2 turns, 2 speakers' in result.output

    def test_writes_the_turns_json(self, tmp_path):
        session = _make_session(tmp_path, '[00:00:01] hello\n')

        with patch(_DIARIZER) as diarizer_class:
            diarizer_class.return_value.diarize.return_value = [SpeakerTurn(0.0, 2.0, 1)]
            CliRunner().invoke(cli, ['diarize', str(session)])

        payload = json.loads((session / 'speakers.json').read_text(encoding='utf-8'))
        assert payload == [{'start': 0.0, 'end': 2.0, 'speaker': 1}]

    def test_missing_recording_is_reported(self, tmp_path):
        session = tmp_path / 'empty'
        session.mkdir()
        (session / 'transcript.txt').write_text('', encoding='utf-8')

        result = CliRunner().invoke(cli, ['diarize', str(session)])

        assert result.exit_code != 0
        assert 'recording.wav' in result.output

    def test_backend_failure_is_reported(self, tmp_path):
        session = _make_session(tmp_path, '[00:00:01] hello\n')

        with patch(_DIARIZER) as diarizer_class:
            diarizer_class.return_value.diarize.side_effect = RuntimeError('LTN_NEMO_MODEL is not set')
            result = CliRunner().invoke(cli, ['diarize', str(session)])

        assert result.exit_code != 0
        assert 'LTN_NEMO_MODEL' in result.output
