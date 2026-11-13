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

    def test_awaiting_permission_while_the_tap_asks(self, monkeypatch, tmp_path):
        monkeypatch.setattr(sys, 'platform', 'darwin')
        fake_binary = tmp_path / 'coreaudio-tap'
        fake_binary.write_bytes(b'')
        monkeypatch.setattr(coreaudio_mod, '_BINARY', fake_binary)

        mock_proc = MagicMock()
        mock_proc.stdout.read.return_value = b''
        mock_proc.poll.return_value = 0  # clean exit: the reader must not drain stderr
        mock_proc.stderr = io.BytesIO(b'coreaudio-tap: waiting for permission: enable this app in System Settings\n')

        with patch('subprocess.Popen', return_value=mock_proc):
            src = CoreAudioTapSource()
            src.open(16000, 1)
            time.sleep(0.05)
            awaiting = src.awaiting_permission
            src.close()

        assert awaiting is True

    def test_awaiting_permission_clears_when_granted(self, monkeypatch, tmp_path):
        monkeypatch.setattr(sys, 'platform', 'darwin')
        fake_binary = tmp_path / 'coreaudio-tap'
        fake_binary.write_bytes(b'')
        monkeypatch.setattr(coreaudio_mod, '_BINARY', fake_binary)

        mock_proc = MagicMock()
        mock_proc.stdout.read.return_value = b''
        mock_proc.poll.return_value = 0  # clean exit: the reader must not drain stderr
        mock_proc.stderr = io.BytesIO(
            b'coreaudio-tap: waiting for permission: enable this app\n'
            b'coreaudio-tap: permission granted\n'
            b'coreaudio-tap: capturing\n'
        )

        with patch('subprocess.Popen', return_value=mock_proc):
            src = CoreAudioTapSource()
            src.open(16000, 1)
            time.sleep(0.05)
            awaiting = src.awaiting_permission
            src.close()

        assert awaiting is False

    def test_system_signal_seen_after_nonzero_sample(self, monkeypatch, tmp_path):
        monkeypatch.setattr(sys, 'platform', 'darwin')
        fake_binary = tmp_path / 'coreaudio-tap'
        fake_binary.write_bytes(b'')
        monkeypatch.setattr(coreaudio_mod, '_BINARY', fake_binary)

        mock_proc = MagicMock()
        mock_proc.stdout.read.return_value = _float32_bytes([0.0, 0.5, 0.0])
        mock_proc.stdin = MagicMock()

        with patch('subprocess.Popen', return_value=mock_proc):
            src = CoreAudioTapSource()
            src.open(16000, 1)
            time.sleep(0.05)
            seen = src.system_signal_seen
            src.close()

        assert seen is True

    def test_system_signal_seen_stays_false_for_all_zero_stream(self, monkeypatch, tmp_path):
        # A missing capture grant makes macOS deliver zeros with no error.
        monkeypatch.setattr(sys, 'platform', 'darwin')
        fake_binary = tmp_path / 'coreaudio-tap'
        fake_binary.write_bytes(b'')
        monkeypatch.setattr(coreaudio_mod, '_BINARY', fake_binary)

        mock_proc = MagicMock()
        mock_proc.stdout.read.return_value = _float32_bytes([0.0, 0.0, 0.0])
        mock_proc.stdin = MagicMock()

        with patch('subprocess.Popen', return_value=mock_proc):
            src = CoreAudioTapSource()
            src.open(16000, 1)
            time.sleep(0.05)
            seen = src.system_signal_seen
            src.close()

        assert seen is False

    def test_spawns_in_own_session(self, monkeypatch, tmp_path):
        # Ctrl-C must not reach the tap subprocess in headless mode, so it runs
        # in its own session (process group).
        monkeypatch.setattr(sys, 'platform', 'darwin')
        fake_binary = tmp_path / 'coreaudio-tap'
        fake_binary.write_bytes(b'')
        monkeypatch.setattr(coreaudio_mod, '_BINARY', fake_binary)

        mock_proc = MagicMock()
        mock_proc.stdout.read.return_value = b''
        with patch('subprocess.Popen', return_value=mock_proc) as mock_popen:
            src = CoreAudioTapSource()
            src.open(16000, 1)
            src.close()

        assert mock_popen.call_args.kwargs['start_new_session'] is True

    def test_asks_for_permission_in_desktop_bundle(self, monkeypatch, tmp_path):
        # The bundle has no terminal to grant the permission, so the tap must
        # raise the dialog itself on first run.
        monkeypatch.setattr(sys, 'platform', 'darwin')
        fake_binary = tmp_path / 'coreaudio-tap'
        fake_binary.write_bytes(b'')
        monkeypatch.setattr(coreaudio_mod, '_BINARY', fake_binary)
        monkeypatch.setenv('LTN_KIOSK', '1')

        mock_proc = MagicMock()
        mock_proc.stdout.read.return_value = b''
        with patch('subprocess.Popen', return_value=mock_proc) as mock_popen:
            src = CoreAudioTapSource()
            src.open(16000, 1)
            src.close()

        assert mock_popen.call_args.args[0] == [str(fake_binary), '--ensure-permission']

    def test_does_not_ask_for_permission_outside_desktop_bundle(self, monkeypatch, tmp_path):
        monkeypatch.setattr(sys, 'platform', 'darwin')
        fake_binary = tmp_path / 'coreaudio-tap'
        fake_binary.write_bytes(b'')
        monkeypatch.setattr(coreaudio_mod, '_BINARY', fake_binary)
        monkeypatch.delenv('LTN_KIOSK', raising=False)

        mock_proc = MagicMock()
        mock_proc.stdout.read.return_value = b''
        with patch('subprocess.Popen', return_value=mock_proc) as mock_popen:
            src = CoreAudioTapSource()
            src.open(16000, 1)
            src.close()

        assert mock_popen.call_args.args[0] == [str(fake_binary)]

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

    def test_close_prefers_stdin_eof(self, monkeypatch, tmp_path):
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

        mock_proc.stdin.close.assert_called()
        mock_proc.terminate.assert_not_called()
        mock_proc.wait.assert_called()

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

    def test_drain_discards_buffered_chunks(self, monkeypatch, tmp_path):
        monkeypatch.setattr(sys, 'platform', 'darwin')

        fake_binary = tmp_path / 'coreaudio-tap'
        fake_binary.write_bytes(b'')
        monkeypatch.setattr(coreaudio_mod, '_BINARY', fake_binary)

        # Reader pushes two chunks then EOF
        chunks = [_float32_bytes([0.1, 0.2]), _float32_bytes([0.3, 0.4]), b'']
        mock_proc = MagicMock()
        mock_proc.stdout.read.side_effect = chunks
        mock_proc.poll.return_value = 0
        mock_proc.stdin = MagicMock()

        with patch('subprocess.Popen', return_value=mock_proc):
            src = CoreAudioTapSource()
            src.open(16000, 1)
            time.sleep(0.05)  # let reader thread enqueue chunks
            src.drain()
            # After drain, no data remains.
            assert src.read(timeout=0.01) is None
            src.close()

    def test_close_kills_on_timeout(self, monkeypatch, tmp_path):
        monkeypatch.setattr(sys, 'platform', 'darwin')

        fake_binary = tmp_path / 'coreaudio-tap'
        fake_binary.write_bytes(b'')
        monkeypatch.setattr(coreaudio_mod, '_BINARY', fake_binary)

        import subprocess as subprocess_mod

        mock_proc = MagicMock()
        mock_proc.stdout.read.side_effect = lambda size: (time.sleep(0.02), b'')[1]
        mock_proc.stdin = MagicMock()
        mock_proc.wait.side_effect = [
            subprocess_mod.TimeoutExpired(cmd='coreaudio-tap', timeout=1),
            subprocess_mod.TimeoutExpired(cmd='coreaudio-tap', timeout=1),
            0,
        ]

        with patch('subprocess.Popen', return_value=mock_proc):
            src = CoreAudioTapSource()
            src.open(16000, 1)
            src.close()

        mock_proc.terminate.assert_called_once()
        mock_proc.kill.assert_called_once()
        assert mock_proc.wait.call_count == 3
        mock_proc.stdout.close.assert_called_once()
        mock_proc.stderr.close.assert_called_once()


def test_reader_reassembles_partial_float_bytes():
    class SplitPipe:
        def __init__(self):
            self.parts = iter([_float32_bytes([0.25])[:3], _float32_bytes([0.25])[3:], b''])

        def read(self, size):
            return next(self.parts)

    source = CoreAudioTapSource()
    process = MagicMock()
    process.stdout = SplitPipe()
    process.poll.return_value = 0
    source._proc = process
    source._reader()

    np.testing.assert_array_equal(source.read(0.01), np.array([0.25], dtype=np.float32))
    assert source.exhausted


def test_reader_reports_truncated_float_at_eof():
    source = CoreAudioTapSource()
    process = MagicMock()
    process.stdout = io.BytesIO(b'abc')
    process.poll.return_value = 0
    source._proc = process
    source._reader()

    with pytest.raises(RuntimeError, match='partial float32'):
        source.read(0.01)


def test_stderr_has_one_reader_and_keeps_exit_diagnostic():
    source = CoreAudioTapSource()
    process = MagicMock()
    process.stdout = io.BytesIO(b'')
    process.stderr = io.BytesIO(b'coreaudio-tap: aggregate failed\n')
    process.poll.return_value = 7
    process.wait.return_value = 7
    source._proc = process
    source._stderr_reader()
    source._reader()

    with pytest.raises(RuntimeError, match='aggregate failed'):
        source.read(0.01)


def test_open_rejects_non_native_output_format(monkeypatch):
    monkeypatch.setattr(sys, 'platform', 'darwin')
    with pytest.raises(ValueError, match='16000 Hz mono'):
        CoreAudioTapSource().open(48000, 2)


@pytest.mark.skipif(sys.platform == 'win32', reason='Helper requires POSIX executable scripts')
def test_real_helper_partial_writes_and_exit_diagnostic(monkeypatch, tmp_path):
    helper = tmp_path / 'helper'
    helper.write_text(
        f'#!{sys.executable}\n'
        'import os, struct, sys, time\n'
        'samples = struct.pack("2f", 0.25, -0.5)\n'
        'os.write(1, samples[:3])\n'
        'time.sleep(0.02)\n'
        'os.write(1, samples[3:])\n'
        'sys.stderr.write("capture device disappeared\\n")\n'
        'sys.exit(7)\n'
    )
    helper.chmod(0o755)
    monkeypatch.setattr(coreaudio_mod, '_BINARY', helper)
    monkeypatch.setattr(sys, 'platform', 'darwin')
    source = CoreAudioTapSource()
    source.open(16000, 1)
    process = source._proc
    try:
        np.testing.assert_array_equal(source.read(5), [0.25, -0.5])
        with pytest.raises(RuntimeError, match='capture device disappeared'):
            source.read(1)
        assert source.exhausted
    finally:
        source.close()
    assert process is not None
    assert process.returncode == 7
    assert process.stdout is not None
    assert process.stderr is not None
    assert process.stdin is not None
    assert process.stdout.closed
    assert process.stderr.closed
    assert process.stdin.closed


@pytest.mark.skipif(sys.platform == 'win32', reason='Helper requires POSIX executable scripts')
def test_real_helper_stops_on_stdin_eof(monkeypatch, tmp_path):
    helper = tmp_path / 'helper'
    helper.write_text(f'#!{sys.executable}\nimport sys\nsys.stdin.buffer.read()\n')
    helper.chmod(0o755)
    monkeypatch.setattr(coreaudio_mod, '_BINARY', helper)
    monkeypatch.setattr(sys, 'platform', 'darwin')
    source = CoreAudioTapSource()
    source.open(16000, 1)
    process = source._proc
    reader = source._thread
    diagnostic_reader = source._stderr_thread
    source.close()
    source.close()
    assert process is not None
    assert reader is not None
    assert diagnostic_reader is not None
    assert process.returncode == 0
    assert not reader.is_alive()
    assert not diagnostic_reader.is_alive()
