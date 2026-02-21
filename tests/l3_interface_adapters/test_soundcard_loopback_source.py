"""Tests for SoundCardLoopbackSource gateway."""

from __future__ import annotations

import sys
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import lazy_take_notes.l3_interface_adapters.gateways.soundcard_loopback_source as loopback_mod
from lazy_take_notes.l3_interface_adapters.gateways.soundcard_loopback_source import SoundCardLoopbackSource


def _make_loopback(device_id='dev-1'):
    """Create a mock loopback microphone."""
    mic = MagicMock()
    mic.isloopback = True
    mic.id = device_id
    return mic


def _make_recorder(*, blocking=False):
    """Create a mock recorder with context manager support."""
    recorder = MagicMock()
    if blocking:
        recorder.record.side_effect = lambda numframes: time.sleep(10)
    else:
        recorder.record.return_value = np.array([[0.1], [0.2], [0.3], [0.4]], dtype=np.float32)
    recorder.__enter__ = MagicMock(return_value=recorder)
    recorder.__exit__ = MagicMock(return_value=False)
    return recorder


def _patch_sc(loopback_devices, default_speaker=None):
    """Patch soundcard's all_microphones and default_speaker."""
    return (
        patch.object(loopback_mod.sc, 'all_microphones', return_value=loopback_devices),
        patch.object(loopback_mod.sc, 'default_speaker', return_value=default_speaker),
    )


class TestFindLoopback:
    def test_matches_default_speaker_by_id(self):
        speaker = MagicMock()
        speaker.id = 'realtek-out'
        wrong = _make_loopback('headset-out')
        correct = _make_loopback('realtek-out')

        patches = _patch_sc([wrong, correct], default_speaker=speaker)
        with patches[0], patches[1]:
            result = SoundCardLoopbackSource._find_loopback()

        assert result is correct

    def test_falls_back_to_first_when_no_id_match(self):
        speaker = MagicMock()
        speaker.id = 'nonexistent-device'
        first = _make_loopback('dev-a')
        second = _make_loopback('dev-b')

        patches = _patch_sc([first, second], default_speaker=speaker)
        with patches[0], patches[1]:
            result = SoundCardLoopbackSource._find_loopback()

        assert result is first

    def test_falls_back_to_first_when_no_default_speaker(self):
        first = _make_loopback('dev-a')
        second = _make_loopback('dev-b')

        patches = _patch_sc([first, second], default_speaker=None)
        with patches[0], patches[1]:
            result = SoundCardLoopbackSource._find_loopback()

        assert result is first

    def test_no_loopback_device_raises(self):
        non_loopback = MagicMock()
        non_loopback.isloopback = False

        patches = _patch_sc([non_loopback], default_speaker=None)
        with patches[0], patches[1]:
            with pytest.raises(RuntimeError, match='No loopback audio device found'):
                SoundCardLoopbackSource._find_loopback()


class TestSoundCardLoopbackSource:
    def test_darwin_raises(self, monkeypatch):
        monkeypatch.setattr(sys, 'platform', 'darwin')
        src = SoundCardLoopbackSource()
        with pytest.raises(RuntimeError, match='not supported on macOS'):
            src.open(16000, 1)

    def test_no_loopback_device_raises(self, monkeypatch):
        monkeypatch.setattr(sys, 'platform', 'linux')
        non_loopback = MagicMock()
        non_loopback.isloopback = False
        patches = _patch_sc([non_loopback])
        with patches[0], patches[1]:
            src = SoundCardLoopbackSource()
            with pytest.raises(RuntimeError, match='No loopback audio device found'):
                src.open(16000, 1)

    def test_read_returns_float32_array(self, monkeypatch):
        monkeypatch.setattr(sys, 'platform', 'linux')

        loopback_device = _make_loopback()
        mock_recorder = _make_recorder()
        loopback_device.recorder.return_value = mock_recorder

        patches = _patch_sc([loopback_device])
        with patches[0], patches[1]:
            src = SoundCardLoopbackSource()
            src.open(16000, 1)
            time.sleep(0.05)  # let reader thread put data in queue
            result = src.read(timeout=0.5)
            src.close()

        assert result is not None
        assert result.dtype == np.float32
        expected = np.array([[0.1], [0.2], [0.3], [0.4]], dtype=np.float32)
        np.testing.assert_allclose(result, expected.mean(axis=1), atol=1e-6)

    def test_read_returns_none_on_timeout(self, monkeypatch):
        monkeypatch.setattr(sys, 'platform', 'linux')

        loopback_device = _make_loopback()
        mock_recorder = _make_recorder(blocking=True)
        loopback_device.recorder.return_value = mock_recorder

        patches = _patch_sc([loopback_device])
        with patches[0], patches[1]:
            src = SoundCardLoopbackSource()
            src.open(16000, 1)
            result = src.read(timeout=0.05)
            src.close()

        assert result is None

    def test_close_stops_recorder(self, monkeypatch):
        monkeypatch.setattr(sys, 'platform', 'linux')

        loopback_device = _make_loopback()
        mock_recorder = _make_recorder(blocking=True)
        loopback_device.recorder.return_value = mock_recorder

        patches = _patch_sc([loopback_device])
        with patches[0], patches[1]:
            src = SoundCardLoopbackSource()
            src.open(16000, 1)
            src.close()

        mock_recorder.__exit__.assert_called_once()

    def test_close_survives_exit_error(self, monkeypatch):
        monkeypatch.setattr(sys, 'platform', 'linux')

        loopback_device = _make_loopback()
        mock_recorder = _make_recorder(blocking=True)
        mock_recorder.__exit__ = MagicMock(side_effect=RuntimeError('PulseAudio teardown fail'))
        loopback_device.recorder.return_value = mock_recorder

        patches = _patch_sc([loopback_device])
        with patches[0], patches[1]:
            src = SoundCardLoopbackSource()
            src.open(16000, 1)
            # close() must not raise even when __exit__ blows up
            src.close()

        assert src._recorder is None
