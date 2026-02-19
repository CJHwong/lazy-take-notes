"""Tests for audio file loader gateway."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestLoadAudioFile:
    def test_raises_file_not_found_when_missing(self, tmp_path: Path) -> None:
        from lazy_take_notes.l3_interface_adapters.gateways.audio_file_loader import load_audio_file

        with pytest.raises(FileNotFoundError, match='not found'):
            load_audio_file(tmp_path / 'missing.wav')

    @patch('lazy_take_notes.l3_interface_adapters.gateways.audio_file_loader.shutil.which', return_value=None)
    def test_raises_when_ffmpeg_not_on_path(self, _mock_which, tmp_path: Path) -> None:
        from lazy_take_notes.l3_interface_adapters.gateways.audio_file_loader import load_audio_file

        audio = tmp_path / 'audio.wav'
        audio.touch()

        with pytest.raises(RuntimeError, match='ffmpeg is required'):
            load_audio_file(audio)

    @patch(
        'lazy_take_notes.l3_interface_adapters.gateways.audio_file_loader.shutil.which', return_value='/usr/bin/ffmpeg'
    )
    @patch('lazy_take_notes.l3_interface_adapters.gateways.audio_file_loader.subprocess.run')
    def test_returns_float32_array_on_success(self, mock_run: MagicMock, _mock_which, tmp_path: Path) -> None:
        from lazy_take_notes.l3_interface_adapters.gateways.audio_file_loader import load_audio_file

        audio = tmp_path / 'audio.wav'
        audio.touch()
        samples = np.ones(16000, dtype=np.float32) * 0.5
        mock_run.return_value = MagicMock(returncode=0, stdout=samples.tobytes(), stderr=b'')

        result = load_audio_file(audio)

        assert result.dtype == np.float32
        assert len(result) == 16000
        assert np.allclose(result, 0.5)

    @patch(
        'lazy_take_notes.l3_interface_adapters.gateways.audio_file_loader.shutil.which', return_value='/usr/bin/ffmpeg'
    )
    @patch('lazy_take_notes.l3_interface_adapters.gateways.audio_file_loader.subprocess.run')
    def test_raises_on_nonzero_exit_code(self, mock_run: MagicMock, _mock_which, tmp_path: Path) -> None:
        from lazy_take_notes.l3_interface_adapters.gateways.audio_file_loader import load_audio_file

        audio = tmp_path / 'audio.mp3'
        audio.touch()
        mock_run.return_value = MagicMock(returncode=1, stdout=b'', stderr=b'Invalid data found when processing input')

        with pytest.raises(RuntimeError, match='exited with code 1'):
            load_audio_file(audio)

    @patch(
        'lazy_take_notes.l3_interface_adapters.gateways.audio_file_loader.shutil.which', return_value='/usr/bin/ffmpeg'
    )
    @patch('lazy_take_notes.l3_interface_adapters.gateways.audio_file_loader.subprocess.run')
    def test_raises_on_empty_stdout(self, mock_run: MagicMock, _mock_which, tmp_path: Path) -> None:
        from lazy_take_notes.l3_interface_adapters.gateways.audio_file_loader import load_audio_file

        audio = tmp_path / 'audio.wav'
        audio.touch()
        mock_run.return_value = MagicMock(returncode=0, stdout=b'', stderr=b'')

        with pytest.raises(RuntimeError, match='no audio output'):
            load_audio_file(audio)

    @patch(
        'lazy_take_notes.l3_interface_adapters.gateways.audio_file_loader.shutil.which', return_value='/usr/bin/ffmpeg'
    )
    @patch('lazy_take_notes.l3_interface_adapters.gateways.audio_file_loader.subprocess.run')
    def test_raises_on_timeout(self, mock_run: MagicMock, _mock_which, tmp_path: Path) -> None:
        from lazy_take_notes.l3_interface_adapters.gateways.audio_file_loader import load_audio_file

        audio = tmp_path / 'audio.wav'
        audio.touch()
        mock_run.side_effect = subprocess.TimeoutExpired(cmd=['ffmpeg'], timeout=300)

        with pytest.raises(RuntimeError, match='timed out'):
            load_audio_file(audio)

    @patch(
        'lazy_take_notes.l3_interface_adapters.gateways.audio_file_loader.shutil.which', return_value='/usr/bin/ffmpeg'
    )
    @patch('lazy_take_notes.l3_interface_adapters.gateways.audio_file_loader.subprocess.run')
    def test_raises_on_os_error(self, mock_run: MagicMock, _mock_which, tmp_path: Path) -> None:
        from lazy_take_notes.l3_interface_adapters.gateways.audio_file_loader import load_audio_file

        audio = tmp_path / 'audio.wav'
        audio.touch()
        mock_run.side_effect = OSError('ffmpeg: No such file or directory')

        with pytest.raises(RuntimeError, match='Failed to launch ffmpeg'):
            load_audio_file(audio)

    @patch(
        'lazy_take_notes.l3_interface_adapters.gateways.audio_file_loader.shutil.which', return_value='/usr/bin/ffmpeg'
    )
    @patch('lazy_take_notes.l3_interface_adapters.gateways.audio_file_loader.subprocess.run')
    def test_ffmpeg_command_uses_correct_flags(self, mock_run: MagicMock, _mock_which, tmp_path: Path) -> None:
        """Verify ffmpeg is invoked with 16 kHz mono float32 pipe output."""
        from lazy_take_notes.l3_interface_adapters.gateways.audio_file_loader import (
            SAMPLE_RATE,
            load_audio_file,
        )

        audio = tmp_path / 'recording.m4a'
        audio.touch()
        samples = np.zeros(SAMPLE_RATE, dtype=np.float32)
        mock_run.return_value = MagicMock(returncode=0, stdout=samples.tobytes(), stderr=b'')

        load_audio_file(audio)

        cmd = mock_run.call_args[0][0]
        assert cmd[0] == 'ffmpeg'
        assert '-ar' in cmd
        assert str(SAMPLE_RATE) in cmd
        assert '-ac' in cmd
        assert '1' in cmd
        assert '-f' in cmd
        assert 'f32le' in cmd
        assert 'pipe:1' in cmd
