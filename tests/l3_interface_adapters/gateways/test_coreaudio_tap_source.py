"""Tests for CoreAudioTapSource gateway."""

from __future__ import annotations

import io
import struct
import sys
import threading
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import lazy_take_notes.l3_interface_adapters.gateways.coreaudio_tap_source as coreaudio_mod
from lazy_take_notes.l3_interface_adapters.gateways.coreaudio_tap_source import CoreAudioTapSource


def _float32_bytes(values: list[float]) -> bytes:
    return struct.pack(f'{len(values)}f', *values)


class TestCoreAudioTapSource:
    def test_non_macos_raises(self, monkeypatch):
        monkeypatch.setattr(sys, 'platform', 'linux')
        src = CoreAudioTapSource()
        with pytest.raises(RuntimeError, match='macOS only'):
            src.open(16000, 1)

    def test_missing_binary_raises(self, monkeypatch, tmp_path):
        monkeypatch.setattr(sys, 'platform', 'darwin')
        monkeypatch.setattr(coreaudio_mod, '_BINARY', tmp_path / 'nonexistent-binary')
        src = CoreAudioTapSource()
        with pytest.raises(RuntimeError, match='Native binary not found'):
            src.open(16000, 1)

    def test_read_returns_float32_array(self, monkeypatch, tmp_path):
        monkeypatch.setattr(sys, 'platform', 'darwin')

        # Create a dummy executable file so the exists() check passes
        fake_binary = tmp_path / 'coreaudio-tap'
        fake_binary.write_bytes(b'')
        monkeypatch.setattr(coreaudio_mod, '_BINARY', fake_binary)

        expected_values = [0.1, 0.2, 0.3, 0.4]
        raw_bytes = _float32_bytes(expected_values)

        mock_proc = MagicMock()
        mock_proc.stdout.read.return_value = raw_bytes
        mock_proc.stdin = MagicMock()

        with patch('subprocess.Popen', return_value=mock_proc):
            src = CoreAudioTapSource()
            src.open(16000, 1)
            time.sleep(0.05)  # let reader thread put data in queue
            result = src.read(timeout=0.5)
            src.close()

        assert result is not None
        np.testing.assert_allclose(result[: len(expected_values)], expected_values, atol=1e-6)

    def test_read_returns_none_on_timeout(self, monkeypatch, tmp_path):
        monkeypatch.setattr(sys, 'platform', 'darwin')

        fake_binary = tmp_path / 'coreaudio-tap'
        fake_binary.write_bytes(b'')
        monkeypatch.setattr(coreaudio_mod, '_BINARY', fake_binary)

        mock_proc = MagicMock()
        mock_proc.stdout.read.return_value = b''  # EOF immediately
        mock_proc.poll.return_value = 0  # clean exit — not an error
        mock_proc.stdin = MagicMock()

        with patch('subprocess.Popen', return_value=mock_proc):
            src = CoreAudioTapSource()
            src.open(16000, 1)
            result = src.read(timeout=0.05)
            src.close()

        assert result is None

    def test_process_nonzero_exit_raises_on_read(self, monkeypatch, tmp_path):
        monkeypatch.setattr(sys, 'platform', 'darwin')

        fake_binary = tmp_path / 'coreaudio-tap'
        fake_binary.write_bytes(b'')
        monkeypatch.setattr(coreaudio_mod, '_BINARY', fake_binary)

        mock_proc = MagicMock()
        mock_proc.stdout.read.return_value = b''  # immediate EOF (process crashed)
        mock_proc.poll.return_value = 1  # non-zero = error exit
        mock_proc.wait.return_value = 1
        mock_proc.stderr.read.return_value = b'AudioHardwareCreateAggregateDevice failed'
        mock_proc.stdin = MagicMock()

        with patch('subprocess.Popen', return_value=mock_proc):
            src = CoreAudioTapSource()
            src.open(16000, 1)
            time.sleep(0.05)  # let reader thread detect the exit
            with pytest.raises(RuntimeError, match='coreaudio-tap exited with code 1'):
                src.read(timeout=0.1)
            src.close()

    def test_close_terminates_process(self, monkeypatch, tmp_path):
        monkeypatch.setattr(sys, 'platform', 'darwin')

        fake_binary = tmp_path / 'coreaudio-tap'
        fake_binary.write_bytes(b'')
        monkeypatch.setattr(coreaudio_mod, '_BINARY', fake_binary)

        mock_proc = MagicMock()
        mock_proc.stdout.read.return_value = b''
        mock_proc.poll.return_value = 0  # clean exit — not an error
        mock_proc.stdin = MagicMock()

        with patch('subprocess.Popen', return_value=mock_proc):
            src = CoreAudioTapSource()
            src.open(16000, 1)
            src.close()

        mock_proc.terminate.assert_called_once()

    def test_stderr_read_exception_still_sets_error(self, monkeypatch, tmp_path):
        monkeypatch.setattr(sys, 'platform', 'darwin')

        fake_binary = tmp_path / 'coreaudio-tap'
        fake_binary.write_bytes(b'')
        monkeypatch.setattr(coreaudio_mod, '_BINARY', fake_binary)

        mock_proc = MagicMock()
        mock_proc.stdout.read.return_value = b''  # immediate EOF (crash)
        mock_proc.poll.return_value = 42
        mock_proc.wait.return_value = 42
        mock_proc.stderr.read.side_effect = OSError('broken pipe')
        mock_proc.stdin = MagicMock()

        with patch('subprocess.Popen', return_value=mock_proc):
            src = CoreAudioTapSource()
            src.open(16000, 1)
            time.sleep(0.05)
            with pytest.raises(RuntimeError, match='exited with code 42'):
                src.read(timeout=0.1)
            src.close()

    def test_stderr_reader_logs_lines(self, monkeypatch, tmp_path):
        """_stderr_reader thread should consume stderr lines from the Swift binary."""
        monkeypatch.setattr(sys, 'platform', 'darwin')

        fake_binary = tmp_path / 'coreaudio-tap'
        fake_binary.write_bytes(b'')
        monkeypatch.setattr(coreaudio_mod, '_BINARY', fake_binary)

        mock_proc = MagicMock()
        mock_proc.stdout.read.return_value = b''  # EOF immediately
        mock_proc.poll.return_value = 0  # clean exit
        mock_proc.stdin = MagicMock()
        mock_proc.pid = 12345
        # BytesIO is iterable (yields lines) AND has .read() — matches real stderr
        mock_proc.stderr = io.BytesIO(b'48000 Hz 2ch -> 16000 Hz 1ch\nready\n')

        with patch('subprocess.Popen', return_value=mock_proc):
            src = CoreAudioTapSource()
            src.open(16000, 1)
            time.sleep(0.1)  # let stderr_reader thread consume lines
            src.close()

    def test_exhausted_false_initially(self):
        src = CoreAudioTapSource()
        assert src.exhausted is False

    def test_exhausted_true_after_clean_eof(self, monkeypatch, tmp_path):
        """When stdout hits EOF with clean exit (code 0), exhausted should be True."""
        monkeypatch.setattr(sys, 'platform', 'darwin')

        fake_binary = tmp_path / 'coreaudio-tap'
        fake_binary.write_bytes(b'')
        monkeypatch.setattr(coreaudio_mod, '_BINARY', fake_binary)

        mock_proc = MagicMock()
        mock_proc.stdout.read.return_value = b''  # EOF immediately
        mock_proc.poll.return_value = 0  # clean exit
        mock_proc.stdin = MagicMock()

        with patch('subprocess.Popen', return_value=mock_proc):
            src = CoreAudioTapSource()
            src.open(16000, 1)
            time.sleep(0.05)  # let reader thread hit EOF
            assert src.exhausted is True
            src.close()

    def test_exhausted_cleared_on_reopen(self, monkeypatch, tmp_path):
        """open() should clear the exhausted flag from a prior session."""
        monkeypatch.setattr(sys, 'platform', 'darwin')

        fake_binary = tmp_path / 'coreaudio-tap'
        fake_binary.write_bytes(b'')
        monkeypatch.setattr(coreaudio_mod, '_BINARY', fake_binary)

        # First session: EOF immediately → exhausted
        eof_proc = MagicMock()
        eof_proc.stdout.read.return_value = b''
        eof_proc.poll.return_value = 0
        eof_proc.stdin = MagicMock()

        with patch('subprocess.Popen', return_value=eof_proc):
            src = CoreAudioTapSource()
            src.open(16000, 1)
            time.sleep(0.05)
            assert src.exhausted is True
            src.close()

        # Second session: reader blocks (stream alive) → exhausted cleared
        block_event = threading.Event()
        hang_proc = MagicMock()
        hang_proc.stdout.read.side_effect = lambda size: (block_event.wait(), b'')[1]
        hang_proc.poll.return_value = None
        hang_proc.stdin = MagicMock()

        with patch('subprocess.Popen', return_value=hang_proc):
            src.open(16000, 1)
            assert src.exhausted is False  # reader is blocked, flag stays clear
            block_event.set()
            src.close()

    def test_close_kills_on_timeout(self, monkeypatch, tmp_path):
        monkeypatch.setattr(sys, 'platform', 'darwin')

        fake_binary = tmp_path / 'coreaudio-tap'
        fake_binary.write_bytes(b'')
        monkeypatch.setattr(coreaudio_mod, '_BINARY', fake_binary)

        import subprocess as subprocess_mod

        mock_proc = MagicMock()
        mock_proc.stdout.read.side_effect = lambda size: time.sleep(10)
        mock_proc.stdin = MagicMock()
        mock_proc.wait.side_effect = subprocess_mod.TimeoutExpired(cmd='coreaudio-tap', timeout=3)

        with patch('subprocess.Popen', return_value=mock_proc):
            src = CoreAudioTapSource()
            src.open(16000, 1)
            src.close()

        mock_proc.terminate.assert_called_once()
        mock_proc.kill.assert_called_once()
