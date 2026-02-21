"""Tests for SoundCardLoopbackSource gateway."""

from __future__ import annotations

import sys
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import lazy_take_notes.l3_interface_adapters.gateways.soundcard_loopback_source as loopback_mod
from lazy_take_notes.l3_interface_adapters.gateways.soundcard_loopback_source import SoundCardLoopbackSource


class TestSoundCardLoopbackSource:
    def test_darwin_raises(self, monkeypatch):
        monkeypatch.setattr(sys, 'platform', 'darwin')
        src = SoundCardLoopbackSource()
        with pytest.raises(RuntimeError, match='not supported on macOS'):
            src.open(16000, 1)

    def test_no_loopback_device_raises(self, monkeypatch):
        monkeypatch.setattr(sys, 'platform', 'linux')
        # Return mics with no loopback flag
        non_loopback = MagicMock()
        non_loopback.isloopback = False
        with patch.object(loopback_mod.sc, 'all_microphones', return_value=[non_loopback]):
            src = SoundCardLoopbackSource()
            with pytest.raises(RuntimeError, match='No loopback audio device found'):
                src.open(16000, 1)

    def test_read_returns_float32_array(self, monkeypatch):
        monkeypatch.setattr(sys, 'platform', 'linux')

        loopback_device = MagicMock()
        loopback_device.isloopback = True

        expected = np.array([[0.1], [0.2], [0.3], [0.4]], dtype=np.float32)
        mock_recorder = MagicMock()
        mock_recorder.record.return_value = expected
        mock_recorder.__enter__ = MagicMock(return_value=mock_recorder)
        mock_recorder.__exit__ = MagicMock(return_value=False)
        loopback_device.recorder.return_value = mock_recorder

        with patch.object(loopback_mod.sc, 'all_microphones', return_value=[loopback_device]):
            src = SoundCardLoopbackSource()
            src.open(16000, 1)
            time.sleep(0.05)  # let reader thread put data in queue
            result = src.read(timeout=0.5)
            src.close()

        assert result is not None
        assert result.dtype == np.float32
        # (frames, 1) should be flattened to mono
        np.testing.assert_allclose(result, expected.mean(axis=1), atol=1e-6)

    def test_read_returns_none_on_timeout(self, monkeypatch):
        monkeypatch.setattr(sys, 'platform', 'linux')

        loopback_device = MagicMock()
        loopback_device.isloopback = True

        mock_recorder = MagicMock()
        # Block forever — simulating no audio available
        mock_recorder.record.side_effect = lambda numframes: time.sleep(10)
        mock_recorder.__enter__ = MagicMock(return_value=mock_recorder)
        mock_recorder.__exit__ = MagicMock(return_value=False)
        loopback_device.recorder.return_value = mock_recorder

        with patch.object(loopback_mod.sc, 'all_microphones', return_value=[loopback_device]):
            src = SoundCardLoopbackSource()
            src.open(16000, 1)
            result = src.read(timeout=0.05)
            src.close()

        assert result is None

    def test_close_stops_recorder(self, monkeypatch):
        monkeypatch.setattr(sys, 'platform', 'linux')

        loopback_device = MagicMock()
        loopback_device.isloopback = True

        mock_recorder = MagicMock()
        mock_recorder.record.side_effect = lambda numframes: time.sleep(10)
        mock_recorder.__enter__ = MagicMock(return_value=mock_recorder)
        mock_recorder.__exit__ = MagicMock(return_value=False)
        loopback_device.recorder.return_value = mock_recorder

        with patch.object(loopback_mod.sc, 'all_microphones', return_value=[loopback_device]):
            src = SoundCardLoopbackSource()
            src.open(16000, 1)
            src.close()

        mock_recorder.__exit__.assert_called_once()
