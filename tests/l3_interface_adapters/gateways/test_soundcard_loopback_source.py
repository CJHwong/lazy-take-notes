"""Tests for SoundCardLoopbackSource gateway."""

from __future__ import annotations

import sys
import time
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# soundcard eagerly connects to PulseAudio at import time (module-level singleton).
# Pre-seed sys.modules so collection works on Linux CI without a running daemon.
# setdefault is a no-op on macOS/Windows where the real import already succeeded.
_sc_stub = MagicMock()
sys.modules.setdefault('soundcard', _sc_stub)
sys.modules.setdefault('soundcard.pulseaudio', _sc_stub)

import lazy_take_notes.l3_interface_adapters.gateways.soundcard_loopback_source as loopback_mod  # noqa: E402 -- must follow soundcard stub
from lazy_take_notes.l3_interface_adapters.gateways.soundcard_loopback_source import (  # noqa: E402 -- must follow soundcard stub
    SoundCardLoopbackSource,
    _patch_soundcard_numpy2_compat,  # noqa: PLC2701 -- testing private patch function
)


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
        first_buffer = True

        def record(numframes):
            nonlocal first_buffer
            if first_buffer:
                first_buffer = False
                return np.zeros((32, 1), dtype=np.float32)
            time.sleep(0.02)
            return None

        recorder.record.side_effect = record
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
            src.drain()
            result = src.read(timeout=0.05)
            src.close()

        assert result is None

    def test_drain_discards_buffered_chunks(self, monkeypatch):
        monkeypatch.setattr(sys, 'platform', 'linux')

        loopback_device = _make_loopback()
        mock_recorder = _make_recorder()  # non-blocking, keeps producing chunks
        loopback_device.recorder.return_value = mock_recorder

        patches = _patch_sc([loopback_device])
        with patches[0], patches[1]:
            src = SoundCardLoopbackSource()
            src.open(16000, 1)
            time.sleep(0.05)  # let reader thread enqueue a few chunks
            src.drain()
            # Stop the reader producing before asserting so we can see the drain
            # actually emptied the queue (next read must time out).
            src._stop.set()  # noqa: SLF001 -- white-box: halt reader to isolate drain effect
            time.sleep(0.02)
            # Drain once more to catch anything the reader snuck in between set() and halt.
            src.drain()
            assert src.read(timeout=0.01) is None
            src.close()

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

    def test_reader_skips_none_record(self, monkeypatch):
        """Line 92: recorder.record() returns None once, then real data."""
        monkeypatch.setattr(sys, 'platform', 'linux')

        loopback_device = _make_loopback()
        mock_recorder = _make_recorder(blocking=True)  # default: block forever
        real_data = np.array([[0.5], [0.6]], dtype=np.float32)

        # None → real data → block (keeps thread alive until close)
        def _record_sequence(numframes, _calls=[0]):  # noqa: B006 -- mutable default is intentional
            _calls[0] += 1
            if _calls[0] == 1:
                return None
            if _calls[0] == 2:
                return real_data
            time.sleep(0.02)

        mock_recorder.record.side_effect = _record_sequence
        loopback_device.recorder.return_value = mock_recorder

        patches = _patch_sc([loopback_device])
        with patches[0], patches[1]:
            src = SoundCardLoopbackSource()
            src.open(16000, 1)
            time.sleep(0.05)
            result = src.read(timeout=0.5)
            src.close()

        assert result is not None
        np.testing.assert_allclose(result, real_data.mean(axis=1), atol=1e-6)

    def test_win32_com_init_and_uninit(self, monkeypatch):
        """Lines 20-22, 28-30, 121-122: COM init/uninit on win32."""
        monkeypatch.setattr(sys, 'platform', 'win32')

        mock_ole32 = MagicMock()
        mock_windll = MagicMock()
        mock_windll.ole32 = mock_ole32

        # ctypes is imported inside the function; mock it at module level
        mock_ctypes = MagicMock()
        mock_ctypes.windll = mock_windll
        monkeypatch.setitem(sys.modules, 'ctypes', mock_ctypes)

        loopback_device = _make_loopback()
        mock_recorder = _make_recorder()
        loopback_device.recorder.return_value = mock_recorder

        patches = _patch_sc([loopback_device])
        with patches[0], patches[1]:
            src = SoundCardLoopbackSource()
            src.open(16000, 1)
            time.sleep(0.05)
            src.close()

        # The worker owns COM setup and cleanup.
        assert mock_ole32.CoInitializeEx.call_count == 1
        assert mock_ole32.CoUninitialize.call_count == 1
        assert src._com_owner is False


class TestPatchSoundcardNumpy2Compat:
    """Tests for the numpy 2.x monkey-patch on soundcard's mediafoundation backend."""

    def test_shim_redirects_fromstring_with_copy_on_win32(self, monkeypatch):
        monkeypatch.setattr(sys, 'platform', 'win32')

        fake_mf = MagicMock()
        fake_mf.numpy = np  # real numpy — the patch wraps it
        # `from soundcard import mediafoundation` resolves via getattr on the
        # parent MagicMock, NOT sys.modules — so we must set both.
        monkeypatch.setattr(_sc_stub, 'mediafoundation', fake_mf)
        monkeypatch.setitem(sys.modules, 'soundcard.mediafoundation', fake_mf)

        _patch_soundcard_numpy2_compat()

        shim = fake_mf.numpy
        # other attributes pass through unchanged
        assert shim.array is np.array
        assert shim.float32 is np.float32

        # fromstring must return a COPY (not a view) — frombuffer returns a view
        # into the original buffer, but fromstring always copied.  Without .copy(),
        # WASAPI capture buffers become dangling pointers after _capture_release().
        raw = b'\x00\x00\x80\x3f\x00\x00\x00\x40'  # [1.0, 2.0] as float32
        compat_fromstring = shim.fromstring
        result = compat_fromstring(raw, dtype='float32')  # type: ignore[no-matching-overload]  -- shim, not real numpy
        np.testing.assert_array_equal(result, np.array([1.0, 2.0], dtype=np.float32))
        assert result.flags.owndata  # must own its data (copy, not view)

    def test_noop_on_non_windows(self, monkeypatch):
        monkeypatch.setattr(sys, 'platform', 'linux')

        fake_mf = MagicMock()
        original_numpy = fake_mf.numpy
        monkeypatch.setattr(_sc_stub, 'mediafoundation', fake_mf)
        monkeypatch.setitem(sys.modules, 'soundcard.mediafoundation', fake_mf)

        _patch_soundcard_numpy2_compat()

        # numpy attribute must not be replaced
        assert fake_mf.numpy is original_numpy

    def test_survives_missing_mediafoundation(self, monkeypatch):
        monkeypatch.setattr(sys, 'platform', 'win32')

        # Replace MagicMock stub with a bare module that genuinely lacks mediafoundation
        bare_sc = types.ModuleType('soundcard')
        monkeypatch.setitem(sys.modules, 'soundcard', bare_sc)

        # Must not raise — graceful no-op when mediafoundation isn't installed
        _patch_soundcard_numpy2_compat()


def test_loopback_reader_error_reaches_read():
    source = SoundCardLoopbackSource()
    recorder = MagicMock()
    recorder.record.side_effect = RuntimeError('device removed')
    source._reader(recorder, 16000)
    assert source.exhausted
    with pytest.raises(RuntimeError, match='device removed'):
        source.read(0.01)


def test_windows_setup_failure_releases_com(monkeypatch):
    monkeypatch.setattr(sys, 'platform', 'win32')
    with (
        patch.object(loopback_mod, '_win_com_init'),
        patch.object(loopback_mod, '_win_com_uninit') as uninitialize,
        patch.object(SoundCardLoopbackSource, '_find_loopback', side_effect=RuntimeError('missing device')),
    ):
        source = SoundCardLoopbackSource()
        with pytest.raises(RuntimeError, match='missing device'):
            source.open(16000, 1)
    uninitialize.assert_called_once()
    assert source._com_owner is False


def test_blocked_reader_keeps_recorder_until_reader_stops(monkeypatch):
    import threading

    monkeypatch.setattr(sys, 'platform', 'linux')
    source = SoundCardLoopbackSource()
    recorder = MagicMock()
    release = threading.Event()
    entered = threading.Event()

    first_buffer = True

    def record(numframes):
        nonlocal first_buffer
        if first_buffer:
            first_buffer = False
            return np.zeros((32, 1), dtype=np.float32)
        entered.set()
        release.wait(5)
        return None

    recorder.record.side_effect = record
    loopback = _make_loopback()
    loopback.recorder.return_value = recorder
    with patch.object(SoundCardLoopbackSource, '_find_loopback', return_value=loopback):
        source.open(16000, 1)
    assert entered.wait(1)
    try:
        with pytest.raises(RuntimeError, match='prevent unsafe teardown'):
            source.close()
        recorder.__exit__.assert_not_called()
        assert source._recorder is recorder
    finally:
        release.set()
        source.close()
    recorder.__exit__.assert_called_once()
    assert source._recorder is None


def test_recorder_and_com_lifetime_stay_on_one_worker(monkeypatch):
    import threading

    monkeypatch.setattr(sys, 'platform', 'win32')
    owner_threads = []
    loopback = _make_loopback()
    recorder = _make_recorder(blocking=True)
    loopback.recorder.side_effect = lambda **kwargs: (owner_threads.append(threading.get_ident()), recorder)[1]
    recorder.__enter__.side_effect = lambda: owner_threads.append(threading.get_ident())
    recorder.__exit__.side_effect = lambda *arguments: owner_threads.append(threading.get_ident())
    with (
        patch.object(loopback_mod, '_win_com_init', side_effect=lambda: owner_threads.append(threading.get_ident())),
        patch.object(loopback_mod, '_win_com_uninit', side_effect=lambda: owner_threads.append(threading.get_ident())),
        patch.object(SoundCardLoopbackSource, '_find_loopback', return_value=loopback),
    ):
        source = SoundCardLoopbackSource()
        source.open(16000, 1)
        source.close()
    assert len(owner_threads) == 5
    assert len(set(owner_threads)) == 1
    assert owner_threads[0] != threading.get_ident()


def test_loopback_requests_bounded_capture_fragments(monkeypatch):
    monkeypatch.setattr(sys, 'platform', 'linux')
    loopback = _make_loopback()
    loopback.recorder.return_value = _make_recorder(blocking=True)
    with patch.object(SoundCardLoopbackSource, '_find_loopback', return_value=loopback):
        source = SoundCardLoopbackSource()
        source.open(16000, 1)
        source.close()
    assert loopback.recorder.call_args.kwargs['blocksize'] == 1600


def test_server_fragment_pacing_allows_shutdown_after_first_read(monkeypatch):
    import threading

    monkeypatch.setattr(sys, 'platform', 'linux')
    release = threading.Event()
    second_read = threading.Event()
    loopback = _make_loopback()
    recorder = MagicMock()

    def create_recorder(*, samplerate, channels, blocksize=None):
        fragment_seconds = blocksize / samplerate if blocksize else 2.1
        calls = 0

        def record(numframes):
            nonlocal calls
            calls += 1
            if calls == 2:
                second_read.set()
            release.wait(fragment_seconds)
            frame_count = numframes if numframes is not None else blocksize
            assert frame_count is not None
            return np.zeros((frame_count, channels), dtype=np.float32)

        recorder.record.side_effect = record
        return recorder

    loopback.recorder.side_effect = create_recorder
    source = SoundCardLoopbackSource()
    try:
        with patch.object(SoundCardLoopbackSource, '_find_loopback', return_value=loopback):
            source.open(16000, 1)
        assert source.read(3) is not None
        assert second_read.wait(1)
        closing = time.monotonic()
        source.close()
        assert time.monotonic() - closing < 0.5
    finally:
        release.set()
        source.close()
    recorder.__exit__.assert_called_once()


@pytest.fixture(autouse=True)
def pulse_backend(monkeypatch):
    backend = MagicMock()
    backend._pa.PA_STREAM_READY = 2
    backend._pa.PA_STREAM_FAILED = 3
    backend._pa.PA_STREAM_TERMINATED = 4
    backend._pulse._pa_stream_get_state.return_value = 2
    backend._pulse._pa_stream_readable_size.return_value = 6400
    backend._ffi.cast.return_value = 2**64 - 1
    monkeypatch.setattr(loopback_mod.sc, 'pulseaudio', backend, raising=False)
    return backend


def test_no_packet_read_stays_cancellable(monkeypatch, pulse_backend):
    import threading

    monkeypatch.setattr(sys, 'platform', 'linux')
    pulse_backend._pulse._pa_stream_readable_size.side_effect = iter([6400] + [0] * 10000)
    entered = threading.Event()
    release = threading.Event()
    recorder = MagicMock()

    first_buffer = True

    def blocking_record(numframes):
        nonlocal first_buffer
        if first_buffer:
            first_buffer = False
            return np.zeros((128, 1), dtype=np.float32)
        entered.set()
        release.wait(5)
        return None

    recorder.record.side_effect = blocking_record
    loopback = _make_loopback()
    loopback.recorder.return_value = recorder
    source = SoundCardLoopbackSource()
    try:
        with patch.object(SoundCardLoopbackSource, '_find_loopback', return_value=loopback):
            source.open(16000, 1)
        assert source.read(0.05) is not None
        assert source.read(0.05) is None
        assert not entered.is_set()
        start = time.monotonic()
        source.close()
        assert time.monotonic() - start < 0.5
    finally:
        release.set()
        source.close()


@pytest.mark.parametrize('stream_state', [3, 4])
def test_stopped_pulse_stream_reports_failure(monkeypatch, pulse_backend, stream_state):
    monkeypatch.setattr(sys, 'platform', 'linux')
    pulse_backend._pulse._pa_stream_get_state.return_value = stream_state
    recorder = MagicMock()
    source = SoundCardLoopbackSource()
    source._reader(recorder, 16000)
    with pytest.raises(RuntimeError, match='PulseAudio stream stopped'):
        source.read(0.01)
    recorder.record.assert_not_called()


def test_pulse_readable_size_error_is_not_a_packet(monkeypatch, pulse_backend):
    monkeypatch.setattr(sys, 'platform', 'linux')
    pulse_backend._pulse._pa_stream_readable_size.return_value = 2**64 - 1
    recorder = MagicMock()
    source = SoundCardLoopbackSource()
    source._reader(recorder, 16000)
    with pytest.raises(RuntimeError, match='PulseAudio readable size failed'):
        source.read(0.01)
    recorder.record.assert_not_called()


def test_ready_pulse_packet_preserves_silent_frames(pulse_backend):
    recorder = MagicMock()
    expected = np.zeros((128, 1), dtype=np.float32)
    recorder.record.return_value = expected
    result = SoundCardLoopbackSource._read_available_pulse(recorder)
    assert result is expected
    recorder.record.assert_called_once_with(numframes=None)


def test_windows_keeps_fixed_frame_reads(monkeypatch):
    monkeypatch.setattr(sys, 'platform', 'win32')
    source = SoundCardLoopbackSource()
    recorder = MagicMock()

    def record(numframes):
        source._stop.set()
        return np.zeros((numframes, 1), dtype=np.float32)

    recorder.record.side_effect = record
    with patch.object(loopback_mod, '_win_com_init'), patch.object(loopback_mod, '_win_com_uninit'):
        source._reader(recorder, 16000)
    recorder.record.assert_called_once_with(numframes=1600)


def test_open_waits_for_first_delivered_buffer(monkeypatch):
    import threading

    monkeypatch.setattr(sys, 'platform', 'linux')
    release = threading.Event()
    recording = threading.Event()
    opened = threading.Event()
    errors = []
    recorder = MagicMock()

    def record(numframes):
        recording.set()
        release.wait(5)
        return np.zeros((128, 1), dtype=np.float32)

    recorder.record.side_effect = record
    loopback = _make_loopback()
    loopback.recorder.return_value = recorder
    source = SoundCardLoopbackSource()

    def open_source():
        try:
            source.open(16000, 1)
        except Exception as error:
            errors.append(error)
        finally:
            opened.set()

    opening = threading.Thread(target=open_source)
    try:
        with patch.object(SoundCardLoopbackSource, '_find_loopback', return_value=loopback):
            opening.start()
            assert recording.wait(1)
            assert not opened.wait(0.05)
            release.set()
            assert opened.wait(1)
        assert not errors
        assert source.read(0.1) is not None
    finally:
        release.set()
        opening.join(1)
        source.close()


def test_stop_during_linux_startup_reports_cancellation(monkeypatch, pulse_backend):
    import threading

    monkeypatch.setattr(sys, 'platform', 'linux')
    pulse_backend._pulse._pa_stream_readable_size.return_value = 0
    loopback = _make_loopback()
    recorder = MagicMock()
    loopback.recorder.return_value = recorder
    source = SoundCardLoopbackSource()
    errors = []

    def open_source():
        try:
            source.open(16000, 1)
        except Exception as error:
            errors.append(error)

    opening = threading.Thread(target=open_source)
    with patch.object(SoundCardLoopbackSource, '_find_loopback', return_value=loopback):
        opening.start()
        deadline = time.monotonic() + 1
        while not pulse_backend._pulse._pa_stream_readable_size.called and time.monotonic() < deadline:
            time.sleep(0.001)
        source.close()
        opening.join(1)
    assert not opening.is_alive()
    assert len(errors) == 1
    assert 'stopped during startup' in str(errors[0])
    recorder.record.assert_not_called()
    recorder.__exit__.assert_called_once()


def test_failure_during_linux_startup_reports_capture_error(monkeypatch, pulse_backend):
    monkeypatch.setattr(sys, 'platform', 'linux')
    pulse_backend._pulse._pa_stream_get_state.return_value = 3
    loopback = _make_loopback()
    recorder = MagicMock()
    loopback.recorder.return_value = recorder
    source = SoundCardLoopbackSource()
    with patch.object(SoundCardLoopbackSource, '_find_loopback', return_value=loopback):
        with pytest.raises(RuntimeError, match='PulseAudio stream stopped'):
            source.open(16000, 1)
    assert source._thread is None
    recorder.record.assert_not_called()
    recorder.__exit__.assert_called_once()
