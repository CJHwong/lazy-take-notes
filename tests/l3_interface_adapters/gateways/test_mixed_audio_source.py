"""Tests for MixedAudioSource compositor."""

from __future__ import annotations

import time

import numpy as np
import pytest

from lazy_take_notes.l3_interface_adapters.gateways.mixed_audio_source import MixedAudioSource
from tests.conftest import FakeAudioSource


class TestMixedAudioSource:
    def test_both_sources_mixed_with_peak_limit(self):
        mic = FakeAudioSource(chunks=[np.array([0.6, 0.7], dtype=np.float32)])
        sys_audio = FakeAudioSource(chunks=[np.array([0.6, 0.7], dtype=np.float32)])
        src = MixedAudioSource(mic, sys_audio)
        src.open(16000, 1)

        time.sleep(0.1)  # let reader threads enqueue
        result = src.read(timeout=0.5)
        src.close()

        assert result is not None
        # Sum = [1.2, 1.4], peak = 1.4 → scale by 0.99/1.4 → [0.8486, 0.99]
        np.testing.assert_allclose(result, [1.2 * (0.99 / 1.4), 0.99], atol=1e-5)

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
        # mic zeroed → combined = sys = [0.4, 0.5], peak 0.5 < 0.99, no scale.
        np.testing.assert_allclose(result, [0.4, 0.5], atol=1e-6)

    def test_mic_muted_toggle_mid_stream(self):
        """Muting takes effect between read() calls. White-box: drive _mic_q/_sys_q
        directly so a second read sees a fresh chunk (the drain-all read path would
        otherwise consume pre-queued chunks together and the mute toggle wouldn't
        apply to the second one)."""
        mic = FakeAudioSource()
        sys_audio = FakeAudioSource()
        src = MixedAudioSource(mic, sys_audio)
        src.open(16000, 1)

        # First read — unmuted: combined = [1.0, 1.2], peak 1.2 → scale by 0.99/1.2.
        src._mic_q.put(np.array([0.6, 0.7], dtype=np.float32))  # noqa: SLF001
        src._sys_q.put(np.array([0.4, 0.5], dtype=np.float32))  # noqa: SLF001
        r1 = src.read(timeout=0.5)
        assert r1 is not None
        np.testing.assert_allclose(r1, [1.0 * (0.99 / 1.2), 0.99], atol=1e-5)

        # Mute, push fresh chunk, read again — mic zeroed → combined = sys, peak 0.5, no scale.
        src.mic_muted = True
        src._mic_q.put(np.array([0.6, 0.7], dtype=np.float32))  # noqa: SLF001
        src._sys_q.put(np.array([0.4, 0.5], dtype=np.float32))  # noqa: SLF001
        r2 = src.read(timeout=0.5)
        src.close()

        assert r2 is not None
        np.testing.assert_allclose(r2, [0.4, 0.5], atol=1e-6)

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
        # combined = [0.2, 0.4, 0.3], peak 0.4 < 0.99, no scale.
        assert len(result) == 3
        np.testing.assert_allclose(result, [0.2, 0.4, 0.3], atol=1e-5)

    def test_read_drains_backlog_into_single_chunk(self):
        """Catch-up path: when mic_q has backlog, one read() returns the merged chunks.

        Prevents permanent lag — a FIFO backlog would otherwise stay queued forever
        because consumer rate matches producer rate in steady state.
        """
        mic = FakeAudioSource(
            chunks=[
                np.array([0.1, 0.2], dtype=np.float32),
                np.array([0.3, 0.4], dtype=np.float32),
                np.array([0.5, 0.6], dtype=np.float32),
            ]
        )
        # System audio covering the same total span (6 samples = 3 mic chunks × 2).
        sys_audio = FakeAudioSource(
            chunks=[np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2], dtype=np.float32)],
        )
        src = MixedAudioSource(mic, sys_audio)
        src.open(16000, 1)

        time.sleep(0.1)  # let reader threads enqueue all three mic chunks
        result = src.read(timeout=0.5)
        src.close()

        assert result is not None
        # All 6 mic samples come through in one call. mic = [0.1..0.6], sys = [0.2]*6.
        # combined = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8], peak 0.8 < 0.99, no scale.
        assert len(result) == 6
        np.testing.assert_allclose(result, [0.3, 0.4, 0.5, 0.6, 0.7, 0.8], atol=1e-5)

    def test_peak_limit_scales_when_combined_would_clip(self):
        """When mic + sys exceeds 0.99, output is scaled so the peak lands at 0.99."""
        mic = FakeAudioSource(chunks=[np.array([0.8, 0.9], dtype=np.float32)])
        sys_audio = FakeAudioSource(chunks=[np.array([0.8, 0.9], dtype=np.float32)])
        src = MixedAudioSource(mic, sys_audio)
        src.open(16000, 1)

        time.sleep(0.1)
        result = src.read(timeout=0.5)
        src.close()

        assert result is not None
        # combined = [1.6, 1.8], peak = 1.8 → scale by 0.99/1.8 → [0.88, 0.99]
        np.testing.assert_allclose(result, [1.6 * (0.99 / 1.8), 0.99], atol=1e-5)

    def test_last_mic_rms_tracks_pre_mix_mic_signal(self):
        """last_mic_rms must reflect the mic's own signal, not the mixed output."""
        mic_chunk = np.array([0.02, 0.03, 0.04], dtype=np.float32)
        sys_chunk = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        mic = FakeAudioSource(chunks=[mic_chunk])
        sys_audio = FakeAudioSource(chunks=[sys_chunk])
        src = MixedAudioSource(mic, sys_audio)
        src.open(16000, 1)

        time.sleep(0.1)
        _ = src.read(timeout=0.5)
        src.close()

        # Expected RMS of the mic chunk alone — system audio must not inflate it.
        expected = float(np.sqrt(np.mean(mic_chunk**2)))
        assert src.last_mic_rms == pytest.approx(expected, rel=1e-4)

    def test_last_mic_rms_reflects_pre_mute_signal(self):
        """Muting must not zero out last_mic_rms — the diagnostic should reflect
        what the mic is actually picking up even when output is muted."""
        mic_chunk = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        # Compute expected BEFORE read(): real sources (Sounddevice, CoreAudioTap)
        # copy each chunk before publishing to the queue, so in-place mic*=0 in
        # read() is safe. FakeAudioSource hands back the same reference, so we
        # snapshot the expected RMS before it's mutated.
        expected = float(np.sqrt(np.mean(mic_chunk**2)))
        mic = FakeAudioSource(chunks=[mic_chunk.copy()])
        sys_audio = FakeAudioSource(chunks=[])
        src = MixedAudioSource(mic, sys_audio)
        src.mic_muted = True
        src.open(16000, 1)

        time.sleep(0.1)
        _ = src.read(timeout=0.5)
        src.close()

        assert src.last_mic_rms == pytest.approx(expected, rel=1e-4)

    def test_silent_sys_preserves_full_mic_amplitude(self):
        """Regression: sys producing zero-filled chunks (CoreAudioTap's idle behaviour)
        must not cause mic attenuation — that was the bug that dropped quiet speakers
        below the transcription VAD gate.
        """
        mic = FakeAudioSource(chunks=[np.array([0.02, 0.03, 0.015], dtype=np.float32)])
        sys_audio = FakeAudioSource(chunks=[np.zeros(3, dtype=np.float32)])
        src = MixedAudioSource(mic, sys_audio)
        src.open(16000, 1)

        time.sleep(0.1)
        result = src.read(timeout=0.5)
        src.close()

        assert result is not None
        # sys is all zeros → combined = mic, peak well under 0.99, no scale.
        # Old * 0.5 behaviour would have halved these quiet mic samples to 0.01/0.015/0.0075.
        np.testing.assert_allclose(result, [0.02, 0.03, 0.015], atol=1e-5)

    def test_drain_clears_queues_and_source_buffers(self):
        """drain() must empty our queues AND delegate to the underlying sources."""
        mic = FakeAudioSource(chunks=[np.array([0.1, 0.2], dtype=np.float32)])
        sys_audio = FakeAudioSource(chunks=[np.array([0.3, 0.4], dtype=np.float32)])
        src = MixedAudioSource(mic, sys_audio)
        src.open(16000, 1)

        time.sleep(0.1)  # let reader threads enqueue chunks into _mic_q/_sys_q
        src.drain()

        # After drain, the next read must not surface the pre-drain chunks.
        result = src.read(timeout=0.05)
        src.close()

        assert result is None
        # drain() must delegate to both underlying sources.
        assert mic.drain_calls == 1
        assert sys_audio.drain_calls == 1
