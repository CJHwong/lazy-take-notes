"""Tests for MixedAudioSource compositor."""

from __future__ import annotations

import time

import numpy as np

from lazy_take_notes.l3_interface_adapters.gateways.mixed_audio_source import MixedAudioSource
from tests.conftest import FakeAudioSource


class TestMixedAudioSource:
    def test_both_sources_mixed_and_attenuated(self):
        mic = FakeAudioSource(chunks=[np.array([0.6, 0.7], dtype=np.float32)])
        sys_audio = FakeAudioSource(chunks=[np.array([0.6, 0.7], dtype=np.float32)])
        src = MixedAudioSource(mic, sys_audio)
        src.open(16000, 1)

        time.sleep(0.1)  # let reader threads enqueue
        result = src.read(timeout=0.5)
        src.close()

        assert result is not None
        # (0.6 + 0.6) * 0.5 = 0.6 — no clipping, attenuated
        np.testing.assert_allclose(result, [0.6, 0.7], atol=1e-6)

    def test_only_mic_has_data(self):
        mic = FakeAudioSource(chunks=[np.array([0.3, 0.4], dtype=np.float32)])
        sys_audio = FakeAudioSource(chunks=[])  # no data
        src = MixedAudioSource(mic, sys_audio)
        src.open(16000, 1)

        time.sleep(0.1)
        result = src.read(timeout=0.5)
        src.close()

        assert result is not None
        np.testing.assert_allclose(result, [0.3, 0.4], atol=1e-6)

    def test_no_data_returns_none(self):
        mic = FakeAudioSource(chunks=[])
        sys_audio = FakeAudioSource(chunks=[])
        src = MixedAudioSource(mic, sys_audio)
        src.open(16000, 1)
        result = src.read(timeout=0.05)
        src.close()
        assert result is None

    def test_close_calls_both_sources(self):
        mic = FakeAudioSource()
        sys_audio = FakeAudioSource()
        src = MixedAudioSource(mic, sys_audio)
        src.open(16000, 1)
        src.close()
        assert mic.close_calls == 1
        assert sys_audio.close_calls == 1

    def test_mic_muted_returns_only_system_audio(self):
        mic = FakeAudioSource(chunks=[np.array([0.6, 0.7], dtype=np.float32)])
        sys_audio = FakeAudioSource(chunks=[np.array([0.4, 0.5], dtype=np.float32)])
        src = MixedAudioSource(mic, sys_audio)
        src.mic_muted = True
        src.open(16000, 1)

        time.sleep(0.1)
        result = src.read(timeout=0.5)
        src.close()

        assert result is not None
        # mic zeroed: (0.0 + 0.4) * 0.5 = 0.2, (0.0 + 0.5) * 0.5 = 0.25
        np.testing.assert_allclose(result, [0.2, 0.25], atol=1e-6)

    def test_mic_muted_toggle_mid_stream(self):
        mic = FakeAudioSource(
            chunks=[
                np.array([0.6, 0.7], dtype=np.float32),
                np.array([0.6, 0.7], dtype=np.float32),
            ]
        )
        sys_audio = FakeAudioSource(
            chunks=[
                np.array([0.4, 0.5], dtype=np.float32),
                np.array([0.4, 0.5], dtype=np.float32),
            ]
        )
        src = MixedAudioSource(mic, sys_audio)
        src.open(16000, 1)

        time.sleep(0.1)
        # First read — unmuted
        r1 = src.read(timeout=0.5)
        assert r1 is not None
        np.testing.assert_allclose(r1, [0.5, 0.6], atol=1e-6)

        # Mute and read again
        src.mic_muted = True
        time.sleep(0.05)
        r2 = src.read(timeout=0.5)
        src.close()

        assert r2 is not None
        # mic zeroed: (0.0 + 0.4) * 0.5 = 0.2, (0.0 + 0.5) * 0.5 = 0.25
        np.testing.assert_allclose(r2, [0.2, 0.25], atol=1e-6)

    def test_size_mismatch_pads_shorter_to_mic_length(self):
        mic = FakeAudioSource(chunks=[np.array([0.1, 0.2, 0.3], dtype=np.float32)])
        sys_audio = FakeAudioSource(chunks=[np.array([0.1, 0.2], dtype=np.float32)])
        src = MixedAudioSource(mic, sys_audio)
        src.open(16000, 1)

        time.sleep(0.1)
        result = src.read(timeout=0.5)
        src.close()

        assert result is not None
        # System audio (len=2) is zero-padded to mic length (3).
        # (0.1+0.1)*0.5=0.1, (0.2+0.2)*0.5=0.2, (0.3+0.0)*0.5=0.15
        assert len(result) == 3
        np.testing.assert_allclose(result, [0.1, 0.2, 0.15], atol=1e-5)
