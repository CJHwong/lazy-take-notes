"""Tests for MixedAudioSource compositor."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from lazy_take_notes.l3_interface_adapters.gateways.mixed_audio_source import MixedAudioSource
from tests.conftest import FakeAudioSource


class RestartableSource:
    """Fake source that plays a scripted batch per open() call and counts restarts."""

    def __init__(self, batches: list[list[np.ndarray]]) -> None:
        self._batches = batches
        self._current: list[np.ndarray] = []
        self.open_calls: list[tuple[int, int]] = []
        self.close_calls = 0
        self.drain_calls = 0
        self.mic_muted = False

    def open(self, sample_rate: int, channels: int) -> None:
        self.open_calls.append((sample_rate, channels))
        index = min(len(self.open_calls) - 1, len(self._batches) - 1)
        self._current = list(self._batches[index])

    def read(self, timeout: float = 0.1) -> np.ndarray | None:
        if not self._current:
            time.sleep(0.005)  # keep the reader thread from spinning hot
            return None
        return self._current.pop(0)

    def close(self) -> None:
        self.close_calls += 1

    def drain(self) -> None:
        self.drain_calls += 1
        self._current = []


class TestMixedAudioSource:
    def test_partial_thread_start_failure_closes_sources_and_started_reader(self, monkeypatch):
        original_start = threading.Thread.start
        starts = 0

        def fail_second_start(thread):
            nonlocal starts
            starts += 1
            if starts == 2:
                raise RuntimeError('cannot create reader thread')
            original_start(thread)

        monkeypatch.setattr(threading.Thread, 'start', fail_second_start)
        microphone, system = FakeAudioSource(), FakeAudioSource()
        source = MixedAudioSource(microphone, system)

        with pytest.raises(RuntimeError, match='cannot create reader thread'):
            source.open(16000, 1)

        assert microphone.close_calls == 1
        assert system.close_calls == 1
        assert not source._threads

    def test_restart_discards_system_samples_from_the_previous_capture(self):
        fail_now = threading.Event()

        class RecoveringSource(FakeAudioSource):
            delivered = False

            def read(self, timeout=0.1):
                if len(self.open_calls) == 1:
                    if fail_now.wait(timeout=0.005):
                        raise RuntimeError('device changed')
                    return None
                if not self.delivered:
                    self.delivered = True
                    return np.full(160, 0.4, dtype=np.float32)
                return None

        system = RecoveringSource()
        source = MixedAudioSource(FakeAudioSource(), system)
        source._restart_delay = 0.001
        source.open(16000, 1)
        source._sys_buf = np.full(160, 0.1, dtype=np.float32)
        fail_now.set()
        deadline = time.monotonic() + 1
        try:
            while source._sys_q.empty() and time.monotonic() < deadline:
                time.sleep(0.005)
            source._mic_q.put(np.zeros(160, dtype=np.float32))
            result = source.read(timeout=0.1)
            assert result is not None
            np.testing.assert_allclose(result, 0.4)
        finally:
            source.close()

    def test_reopen_without_buffers_eventually_reports_failure(self):
        class SilentAfterReopen(FakeAudioSource):
            def read(self, timeout=0.1):
                if len(self.open_calls) == 1:
                    raise RuntimeError('device disconnected')
                return None

        microphone = SilentAfterReopen()
        source = MixedAudioSource(microphone, FakeAudioSource(), dead_audio_seconds=0.01)
        source._restart_delay = 0.001
        source.open(16000, 1)
        deadline = time.monotonic() + 1
        try:
            while len(microphone.open_calls) < 4 and time.monotonic() < deadline:
                time.sleep(0.005)
            assert len(microphone.open_calls) == 4
            assert any('did not deliver a buffer' in error for error in source.source_errors)
            assert not source.exhausted
        finally:
            source.close()

    def test_startup_rollback_preserves_error_when_cleanup_also_fails(self):
        class BrokenMicrophone(FakeAudioSource):
            def close(self):
                raise RuntimeError('microphone cleanup failed')

        class BrokenSystem(FakeAudioSource):
            def open(self, sample_rate, channels):
                raise RuntimeError('system startup failed')

        source = MixedAudioSource(BrokenMicrophone(), BrokenSystem())

        with pytest.raises(RuntimeError, match='system startup failed'):
            source.open(16000, 1)

    def test_reopen_resets_dropped_frame_count(self):
        source = MixedAudioSource(FakeAudioSource(), FakeAudioSource())
        source._sys_q.put(np.ones(8000, dtype=np.float32))
        assert source.dropped_frames == 4000

        source.open(16000, 1)
        try:
            assert source.dropped_frames == 0
        finally:
            source.close()

    def test_interleaved_sources_do_not_count_the_same_time_twice(self):
        source = MixedAudioSource(FakeAudioSource(), FakeAudioSource())
        source.open(16000, 1)
        source._sys_q.put(np.full(1600, 0.2, dtype=np.float32))
        timer = threading.Timer(0.02, lambda: source._mic_q.put(np.full(512, 0.1, dtype=np.float32)))
        timer.start()
        try:
            result = source.read(timeout=0.1)
            assert result is not None
            assert result.size == 512
            np.testing.assert_allclose(result, 0.3)
            assert source._sys_buf.size == 1088
        finally:
            timer.join()
            source.close()

    def test_retry_count_is_bounded_and_both_failed_sources_exhaust(self):
        class BrokenSource(FakeAudioSource):
            def read(self, timeout=0.1):
                raise RuntimeError('disconnected')

        microphone, system = BrokenSource(), BrokenSource()
        source = MixedAudioSource(microphone, system)
        source._restart_delay = 0.001
        source.open(16000, 1)
        deadline = time.monotonic() + 1
        try:
            while not source.exhausted and time.monotonic() < deadline:
                time.sleep(0.005)
            assert source.exhausted
            assert len(source.source_errors) == 2
            assert len(microphone.open_calls) == 4
            assert len(system.open_calls) == 4
        finally:
            source.close()

    def test_recovery_confirms_a_fresh_buffer_before_clearing_the_error(self):
        release_buffer = threading.Event()

        class RecoveringSource(FakeAudioSource):
            def read(self, timeout=0.1):
                if len(self.open_calls) == 1:
                    raise RuntimeError('disconnected')
                if release_buffer.wait(timeout=0.005):
                    return np.full(160, 0.2, dtype=np.float32)
                return None

        microphone = RecoveringSource()
        source = MixedAudioSource(microphone, FakeAudioSource())
        source._restart_delay = 0.001
        source.open(16000, 1)
        deadline = time.monotonic() + 1
        try:
            while len(microphone.open_calls) < 2 and time.monotonic() < deadline:
                time.sleep(0.005)
            assert len(microphone.open_calls) == 2
            assert source.source_errors
            release_buffer.set()
            assert source.read(timeout=0.5) is not None
            assert not source.source_errors
        finally:
            source.close()

    def test_stop_during_recovery_cannot_reopen_a_source(self):
        class BrokenSource(FakeAudioSource):
            def read(self, timeout=0.1):
                raise RuntimeError('disconnected')

        microphone = BrokenSource()
        source = MixedAudioSource(microphone, FakeAudioSource())
        source.open(16000, 1)
        deadline = time.monotonic() + 1
        while not source.source_errors and time.monotonic() < deadline:
            time.sleep(0.005)
        source.close()

        assert len(microphone.open_calls) == 1
        assert not source._threads

    def test_continuous_source_restarts_when_buffers_stop(self):
        class StalledSource(RestartableSource):
            continuous_capture = True

        microphone = StalledSource([[], [np.full(160, 0.1, dtype=np.float32)] * 10])
        source = MixedAudioSource(microphone, FakeAudioSource(), dead_audio_seconds=0.03)
        source._restart_delay = 0.001
        source.open(16000, 1)
        try:
            result = source.read(timeout=0.5)
            assert result is not None
            assert len(microphone.open_calls) == 2
        finally:
            source.close()

    def test_permission_wait_does_not_count_as_a_stalled_stream(self):
        class PermissionSource(FakeAudioSource):
            continuous_capture = True
            awaiting_permission = True

        microphone = PermissionSource()
        source = MixedAudioSource(microphone, FakeAudioSource(), dead_audio_seconds=0.01)
        source.open(16000, 1)
        try:
            assert source.read(timeout=0.05) is None
            assert not source.source_errors
            assert len(microphone.open_calls) == 1
        finally:
            source.close()

    def test_system_audio_continues_without_microphone_frames(self):
        source = MixedAudioSource(FakeAudioSource(), FakeAudioSource())
        source._sys_q.put(np.full(1600, 0.2, dtype=np.float32))

        result = source.read(timeout=0.01)

        assert result is not None
        np.testing.assert_allclose(result, 0.2)
        assert source.last_mic_rms == 0.0

    def test_source_errors_reach_caller_while_other_audio_continues(self):
        class BrokenSource(FakeAudioSource):
            def read(self, timeout=0.1):
                raise RuntimeError('test device disconnected')

        source = MixedAudioSource(BrokenSource(), FakeAudioSource())
        source._restart_delay = 0.001
        source.open(16000, 1)
        deadline = time.monotonic() + 1
        while not source.source_errors and time.monotonic() < deadline:
            time.sleep(0.005)
        source._sys_q.put(np.full(160, 0.2, dtype=np.float32))
        try:
            result = source.read(timeout=0.1)
            assert result is not None
            np.testing.assert_allclose(result, 0.2)
            assert any('test device disconnected' in error for error in source.source_errors)
            assert not source.exhausted
        finally:
            source.close()

    def test_failed_startup_closes_already_open_microphone(self):
        class BrokenSource(FakeAudioSource):
            def open(self, sample_rate, channels):
                raise RuntimeError('system startup failed')

        microphone = FakeAudioSource()
        source = MixedAudioSource(microphone, BrokenSource())

        with pytest.raises(RuntimeError, match='system startup failed'):
            source.open(16000, 1)

        assert microphone.close_calls == 1

    def test_close_still_closes_system_when_microphone_close_fails(self):
        class BrokenSource(FakeAudioSource):
            def close(self):
                raise RuntimeError('microphone close failed')

        system = FakeAudioSource()
        source = MixedAudioSource(BrokenSource(), system)
        source.open(16000, 1)

        with pytest.raises(RuntimeError, match='microphone close failed'):
            source.close()

        assert system.close_calls == 1
        assert not source._threads

    def test_slow_consumer_keeps_recent_audio_with_bounded_duration(self, caplog):
        source = MixedAudioSource(FakeAudioSource(), FakeAudioSource())
        for index in range(100):
            source._sys_q.put(np.full(1600, index / 1000, dtype=np.float32))

        result = source.read(timeout=0.01)

        assert result is not None
        assert result.size <= 4000
        assert result[-1] == pytest.approx(0.099)
        assert 'dropped' in caplog.text

    def test_frame_mismatch_cannot_accumulate_unbounded_system_delay(self, caplog):
        source = MixedAudioSource(FakeAudioSource(), FakeAudioSource())
        for _ in range(1000):
            source._mic_q.put(np.zeros(1590, dtype=np.float32))
            source._sys_q.put(np.zeros(1600, dtype=np.float32))
            source.read(timeout=0.001)

        assert source._sys_buf.size <= 4000
        assert 'dropped' in caplog.text

    def test_system_signal_seen_delegates_to_system_source(self):
        class ReportingSource(FakeAudioSource):
            system_signal_seen = False

        src = MixedAudioSource(FakeAudioSource(chunks=[]), ReportingSource(chunks=[]))

        assert src.system_signal_seen is False

    def test_system_signal_seen_defaults_true_for_sources_without_the_flag(self):
        src = MixedAudioSource(FakeAudioSource(chunks=[]), FakeAudioSource(chunks=[]))

        assert src.system_signal_seen is True

    def test_awaiting_permission_delegates_to_system_source(self):
        class ReportingSource(FakeAudioSource):
            awaiting_permission = True

        src = MixedAudioSource(FakeAudioSource(chunks=[]), ReportingSource(chunks=[]))

        assert src.awaiting_permission is True

    def test_awaiting_permission_defaults_false_for_sources_without_the_flag(self):
        src = MixedAudioSource(FakeAudioSource(chunks=[]), FakeAudioSource(chunks=[]))

        assert src.awaiting_permission is False

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
        below the transcription VAD gate. Mic values set above the amplification
        target so this isolates peak-limit behaviour from auto-amplify.
        """
        mic = FakeAudioSource(chunks=[np.array([0.04, 0.05, 0.03], dtype=np.float32)])
        sys_audio = FakeAudioSource(chunks=[np.zeros(3, dtype=np.float32)])
        src = MixedAudioSource(mic, sys_audio)
        src.open(16000, 1)

        time.sleep(0.1)
        result = src.read(timeout=0.5)
        src.close()

        assert result is not None
        # sys is all zeros → combined = mic, peak 0.05 < 0.99, no scale.
        np.testing.assert_allclose(result, [0.04, 0.05, 0.03], atol=1e-5)

    def test_auto_amplify_quiet_mic(self):
        """Mic below silence_threshold target is amplified so speech clears VAD gate.
        White-box: push chunks into _mic_q directly to control timing per-read."""
        mic = FakeAudioSource()
        sys_audio = FakeAudioSource()
        src = MixedAudioSource(mic, sys_audio, silence_threshold=0.01)
        src.open(16000, 1)

        quiet = np.full(100, 0.005, dtype=np.float32)

        # Feed several reads to let EMA converge to ~0.005.
        for _ in range(10):
            src._mic_q.put(quiet.copy())  # noqa: SLF001
            src.read(timeout=0.5)

        # Next read with converged EMA: gain ≈ 0.025 / 0.005 = 5x.
        src._mic_q.put(quiet.copy())  # noqa: SLF001
        result = src.read(timeout=0.5)
        src.close()

        assert result is not None
        result_rms = float(np.sqrt(np.mean(result**2)))
        assert result_rms > 0.02, f'Expected amplified RMS > 0.02, got {result_rms}'

    def test_no_amplify_normal_mic(self):
        """Mic above target RMS should not be amplified."""
        mic = FakeAudioSource()
        sys_audio = FakeAudioSource()
        src = MixedAudioSource(mic, sys_audio)
        src.open(16000, 1)

        normal = np.full(100, 0.05, dtype=np.float32)
        src._mic_q.put(normal.copy())  # noqa: SLF001
        result = src.read(timeout=0.5)
        src.close()

        assert result is not None
        np.testing.assert_allclose(result, 0.05, atol=1e-5)

    def test_no_amplify_dead_mic(self):
        """Mic below noise floor (dead/disconnected) should not be amplified."""
        mic = FakeAudioSource()
        sys_audio = FakeAudioSource()
        src = MixedAudioSource(mic, sys_audio)
        src.open(16000, 1)

        dead = np.full(100, 0.0001, dtype=np.float32)
        for _ in range(5):
            src._mic_q.put(dead.copy())  # noqa: SLF001
            src.read(timeout=0.5)

        src._mic_q.put(dead.copy())  # noqa: SLF001
        result = src.read(timeout=0.5)
        src.close()

        assert result is not None
        np.testing.assert_allclose(result, 0.0001, atol=1e-6)

    def test_amplify_gain_capped(self):
        """Gain must not exceed max_mic_gain even when uncapped gain would be higher."""
        mic = FakeAudioSource()
        sys_audio = FakeAudioSource()
        # mic at 0.005, target 0.025 → uncapped gain = 5x, but cap at 3x.
        src = MixedAudioSource(mic, sys_audio, silence_threshold=0.01, max_mic_gain=3.0)
        src.open(16000, 1)

        quiet = np.full(100, 0.005, dtype=np.float32)
        for _ in range(10):
            src._mic_q.put(quiet.copy())  # noqa: SLF001
            src.read(timeout=0.5)

        src._mic_q.put(quiet.copy())  # noqa: SLF001
        result = src.read(timeout=0.5)
        src.close()

        assert result is not None
        result_rms = float(np.sqrt(np.mean(result**2)))
        # Capped at 3x: output ~0.015, not 5x = ~0.025.
        assert result_rms < 0.018, f'Expected capped RMS < 0.018, got {result_rms}'
        assert result_rms > 0.012, f'Expected amplified RMS > 0.012, got {result_rms}'

    def test_no_amplify_when_system_audio_is_active(self):
        """When system audio is playing, mic must NOT be auto-amplified — otherwise
        speaker-to-mic acoustic bleed gets boosted and the remote voice ends up in
        the mix twice (once via the loopback tap, once via the amplified mic copy).
        That doubling/phasing was the original 'voice tracks mix together' bug."""
        mic = FakeAudioSource()
        sys_audio = FakeAudioSource()
        src = MixedAudioSource(mic, sys_audio, silence_threshold=0.01)
        src.open(16000, 1)

        # mic carries quiet acoustic bleed (in the [noise_floor, target] band that
        # would normally trigger amp), sys carries the actual playing voice.
        bleed = np.full(100, 0.005, dtype=np.float32)
        sys_active = np.full(100, 0.05, dtype=np.float32)

        for _ in range(20):  # let both EMAs converge
            src._mic_q.put(bleed.copy())  # noqa: SLF001
            src._sys_q.put(sys_active.copy())  # noqa: SLF001
            src.read(timeout=0.5)

        src._mic_q.put(bleed.copy())  # noqa: SLF001
        src._sys_q.put(sys_active.copy())  # noqa: SLF001
        src.read(timeout=0.5)
        src.close()

        # Without the sys-active gate the gain would be ~5x. The gate must hold it at 1x.
        assert src._mic_gain == 1.0, f'Expected gain held at 1.0x, got {src._mic_gain}x'  # noqa: SLF001

    def test_system_backlog_retains_recent_sample_order(self):
        source = MixedAudioSource(FakeAudioSource(), FakeAudioSource())
        for index in range(200):
            source._sys_q.put(np.full(100, (index + 1) / 1000, dtype=np.float32))
        source._mic_q.put(np.zeros(10, dtype=np.float32))

        result = source.read(timeout=0.1)

        assert result is not None
        np.testing.assert_allclose(result, 0.161)
        assert source._sys_buf.size == 3990
        np.testing.assert_allclose(source._sys_buf[:90], 0.161)
        np.testing.assert_allclose(source._sys_buf[90:190], 0.162)
        np.testing.assert_allclose(source._sys_buf[-100:], 0.2)
        assert source.dropped_frames == 16000

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

    def test_valid_system_silence_never_restarts_while_mic_is_active(self):
        """Local speech does not imply computer playback."""
        mic = RestartableSource([[np.full(16, 0.001, dtype=np.float32)] * 500])
        sys_audio = RestartableSource(
            [
                [np.full(16, 0.5, dtype=np.float32), *[np.zeros(16, dtype=np.float32)] * 4],
                [np.full(16, 0.5, dtype=np.float32)],
            ]
        )
        src = MixedAudioSource(mic, sys_audio, dead_audio_seconds=0.001)
        src.open(16000, 1)
        time.sleep(0.2)
        src.close()

        assert len(sys_audio.open_calls) == 1

    def test_dead_system_source_waits_for_mic_activity(self):
        """Genuine system silence is all zeros. Without mic signal to prove the
        room is active, the tap must not be restarted."""
        mic = RestartableSource([[np.zeros(16, dtype=np.float32)] * 500])
        sys_audio = RestartableSource([[np.full(16, 0.5, dtype=np.float32), *[np.zeros(16, dtype=np.float32)] * 50]])
        src = MixedAudioSource(mic, sys_audio, dead_audio_seconds=0.001)
        src.open(16000, 1)
        time.sleep(0.2)
        src.close()

        assert len(sys_audio.open_calls) == 1

    def test_valid_microphone_silence_never_restarts_after_signal(self):
        """A muted device may deliver exact zeros after normal speech."""
        mic = RestartableSource(
            [
                [np.full(16, 0.001, dtype=np.float32), *[np.zeros(16, dtype=np.float32)] * 4],
                [np.full(16, 0.001, dtype=np.float32)],
            ]
        )
        sys_audio = RestartableSource([[np.full(16, 0.5, dtype=np.float32)] * 500])
        src = MixedAudioSource(mic, sys_audio, dead_audio_seconds=0.001)
        src.open(16000, 1)
        time.sleep(0.2)
        src.close()

        assert len(mic.open_calls) == 1

    def test_zero_only_source_never_restarts(self):
        """A source that has never delivered signal is not dead; it is quiet.
        Leave it alone."""
        mic = RestartableSource([[np.zeros(16, dtype=np.float32)] * 500])
        sys_audio = RestartableSource([[np.zeros(16, dtype=np.float32)] * 500])
        src = MixedAudioSource(mic, sys_audio, dead_audio_seconds=0.001)
        src.open(16000, 1)
        time.sleep(0.1)
        src.close()

        assert len(mic.open_calls) == 1
        assert len(sys_audio.open_calls) == 1

    def test_extended_valid_silence_does_not_restart_the_system_source(self):
        """Fresh buffers confirm delivery even when every sample is zero."""
        mic = RestartableSource([[np.full(16, 0.001, dtype=np.float32)] * 1000])
        sys_audio = RestartableSource(
            [
                [np.full(16, 0.5, dtype=np.float32), *[np.zeros(16, dtype=np.float32)] * 100],
                [np.zeros(16, dtype=np.float32)] * 100,
            ]
        )
        src = MixedAudioSource(mic, sys_audio, dead_audio_seconds=0.001)
        src.open(16000, 1)
        time.sleep(0.3)
        src.close()

        assert len(sys_audio.open_calls) == 1
