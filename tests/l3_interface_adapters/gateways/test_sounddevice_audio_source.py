"""Tests for SounddeviceAudioSource gateway — patches sd.InputStream."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

MODULE = 'lazy_take_notes.l3_interface_adapters.gateways.sounddevice_audio_source'


# Default fixture uses native rate = target rate, so ratio=1 (passthrough — no decimation).
# Tests that exercise decimation supply a separate device info with a non-target native rate.
_FAKE_DEVICE_INFO = {'name': 'Test Input', 'default_samplerate': 16000.0}
_FAKE_DEVICE_INFO_48K = {'name': 'Test Input 48k', 'default_samplerate': 48000.0}
_FAKE_DEVICE_INFO_96K = {'name': 'Test Input 96k', 'default_samplerate': 96000.0}
_FAKE_DEVICE_INFO_44_1K = {'name': 'Test Input 44.1k', 'default_samplerate': 44100.0}


class TestSounddeviceAudioSource:
    @patch(f'{MODULE}.sd.query_devices', return_value=_FAKE_DEVICE_INFO)
    @patch(f'{MODULE}.sd.InputStream')
    def test_open_creates_and_starts_stream(self, mock_stream_cls, _mock_qd):
        from lazy_take_notes.l3_interface_adapters.gateways.sounddevice_audio_source import SounddeviceAudioSource

        mock_stream = MagicMock()
        mock_stream_cls.return_value = mock_stream

        src = SounddeviceAudioSource()
        src.open(16000, 1)

        mock_stream_cls.assert_called_once()
        call_kwargs = mock_stream_cls.call_args.kwargs
        assert call_kwargs['samplerate'] == 16000
        assert call_kwargs['channels'] == 1
        assert call_kwargs['dtype'] == 'float32'
        mock_stream.start.assert_called_once()

    @patch(f'{MODULE}.sd.query_devices', return_value=_FAKE_DEVICE_INFO_96K)
    @patch(f'{MODULE}.sd.InputStream')
    def test_open_at_native_rate_for_integer_decimation(self, mock_stream_cls, _mock_qd):
        """When native rate is an integer multiple of the target, open at native
        and decimate in Python — bypasses PortAudio's resampling which silently
        under-delivers samples on macOS CoreAudio at 96k → 16k."""
        from lazy_take_notes.l3_interface_adapters.gateways.sounddevice_audio_source import SounddeviceAudioSource

        mock_stream_cls.return_value = MagicMock()

        SounddeviceAudioSource().open(16000, 1)

        call_kwargs = mock_stream_cls.call_args.kwargs
        # Stream opened at native rate, blocksize a multiple of the decimation ratio
        assert call_kwargs['samplerate'] == 96000
        assert call_kwargs['blocksize'] % 6 == 0

    @patch(f'{MODULE}.sd.query_devices', return_value=_FAKE_DEVICE_INFO_44_1K)
    @patch(f'{MODULE}.sd.InputStream')
    def test_open_falls_back_to_target_rate_for_non_integer_ratio(self, mock_stream_cls, _mock_qd):
        """Devices whose native rate isn't a clean multiple of the target (e.g.
        44.1 kHz) fall back to PortAudio's resampling. Documented limitation."""
        from lazy_take_notes.l3_interface_adapters.gateways.sounddevice_audio_source import SounddeviceAudioSource

        mock_stream_cls.return_value = MagicMock()

        SounddeviceAudioSource().open(16000, 1)

        call_kwargs = mock_stream_cls.call_args.kwargs
        assert call_kwargs['samplerate'] == 16000

    @patch(f'{MODULE}.sd.query_devices', return_value=_FAKE_DEVICE_INFO)
    @patch(f'{MODULE}.sd.InputStream')
    def test_callback_wires_to_queue(self, mock_stream_cls, _mock_qd):
        from lazy_take_notes.l3_interface_adapters.gateways.sounddevice_audio_source import SounddeviceAudioSource

        mock_stream = MagicMock()
        mock_stream_cls.return_value = mock_stream

        src = SounddeviceAudioSource()
        src.open(16000, 1)

        # Extract the callback from the InputStream constructor
        callback = mock_stream_cls.call_args.kwargs['callback']
        # Simulate audio data arriving (passthrough path: ratio=1, no decimation)
        fake_data = np.array([[0.1], [0.2]], dtype=np.float32)
        callback(fake_data, 2, None, None)

        # Data should be in the queue
        result = src.read(timeout=0.1)
        assert result is not None
        np.testing.assert_allclose(result, [0.1, 0.2], atol=1e-6)

    @patch(f'{MODULE}.sd.query_devices', return_value=_FAKE_DEVICE_INFO_48K)
    @patch(f'{MODULE}.sd.InputStream')
    def test_callback_decimates_when_ratio_greater_than_one(self, mock_stream_cls, _mock_qd):
        """48 kHz native → 16 kHz target → ratio 3. Each group of 3 input samples
        averages into one output sample at the target rate."""
        from lazy_take_notes.l3_interface_adapters.gateways.sounddevice_audio_source import SounddeviceAudioSource

        mock_stream_cls.return_value = MagicMock()

        src = SounddeviceAudioSource()
        src.open(16000, 1)

        callback = mock_stream_cls.call_args.kwargs['callback']
        # 6 input frames @ 48k → 2 output frames @ 16k. Pairs averaged.
        fake_data = np.array(
            [[0.0], [0.3], [0.6], [0.9], [0.6], [0.3]],
            dtype=np.float32,
        )
        callback(fake_data, 6, None, None)

        result = src.read(timeout=0.1)
        assert result is not None
        # Groups: mean([0.0, 0.3, 0.6]) = 0.3 ; mean([0.9, 0.6, 0.3]) = 0.6
        np.testing.assert_allclose(result, [0.3, 0.6], atol=1e-5)

    @patch(f'{MODULE}.sd.query_devices', return_value=_FAKE_DEVICE_INFO_48K)
    @patch(f'{MODULE}.sd.InputStream')
    def test_callback_skips_when_frames_smaller_than_ratio(self, mock_stream_cls, _mock_qd):
        """If a callback delivers fewer frames than the decimation ratio, drop
        them — blocksize is configured to prevent this, but be defensive."""
        from lazy_take_notes.l3_interface_adapters.gateways.sounddevice_audio_source import SounddeviceAudioSource

        mock_stream_cls.return_value = MagicMock()

        src = SounddeviceAudioSource()
        src.open(16000, 1)

        callback = mock_stream_cls.call_args.kwargs['callback']
        callback(np.array([[0.1], [0.2]], dtype=np.float32), 2, None, None)

        assert src.read(timeout=0.01) is None

    @patch(f'{MODULE}.sd.query_devices', return_value=_FAKE_DEVICE_INFO)
    @patch(f'{MODULE}.sd.InputStream')
    def test_read_timeout_returns_none(self, mock_stream_cls, _mock_qd):
        from lazy_take_notes.l3_interface_adapters.gateways.sounddevice_audio_source import SounddeviceAudioSource

        mock_stream_cls.return_value = MagicMock()

        src = SounddeviceAudioSource()
        src.open(16000, 1)

        # Queue is empty — should return None on timeout
        result = src.read(timeout=0.01)
        assert result is None

    @patch(f'{MODULE}.sd.query_devices', return_value=_FAKE_DEVICE_INFO)
    @patch(f'{MODULE}.sd.InputStream')
    def test_drain_discards_buffered_chunks(self, mock_stream_cls, _mock_qd):
        from lazy_take_notes.l3_interface_adapters.gateways.sounddevice_audio_source import SounddeviceAudioSource

        mock_stream_cls.return_value = MagicMock()

        src = SounddeviceAudioSource()
        src.open(16000, 1)

        callback = mock_stream_cls.call_args.kwargs['callback']
        callback(np.array([[0.1], [0.2]], dtype=np.float32), 2, None, None)
        callback(np.array([[0.3], [0.4]], dtype=np.float32), 2, None, None)

        src.drain()

        # After drain, read() should block-and-timeout because queue is empty.
        assert src.read(timeout=0.01) is None

    @patch(f'{MODULE}.sd.query_devices', return_value=_FAKE_DEVICE_INFO)
    @patch(f'{MODULE}.sd.InputStream')
    def test_drain_on_empty_queue_is_noop(self, mock_stream_cls, _mock_qd):
        from lazy_take_notes.l3_interface_adapters.gateways.sounddevice_audio_source import SounddeviceAudioSource

        mock_stream_cls.return_value = MagicMock()

        src = SounddeviceAudioSource()
        src.open(16000, 1)

        src.drain()  # should not raise
        assert src.read(timeout=0.01) is None

    @patch(f'{MODULE}.sd.query_devices', return_value=_FAKE_DEVICE_INFO)
    @patch(f'{MODULE}.sd.InputStream')
    def test_close_stops_and_closes_stream(self, mock_stream_cls, _mock_qd):
        from lazy_take_notes.l3_interface_adapters.gateways.sounddevice_audio_source import SounddeviceAudioSource

        mock_stream = MagicMock()
        mock_stream_cls.return_value = mock_stream

        src = SounddeviceAudioSource()
        src.open(16000, 1)
        src.close()

        mock_stream.stop.assert_called_once()
        mock_stream.close.assert_called_once()
        assert src._stream is None

    @patch(f'{MODULE}.sd.query_devices', return_value=_FAKE_DEVICE_INFO)
    @patch(f'{MODULE}.sd.InputStream')
    def test_callback_warns_on_portaudio_status(self, mock_stream_cls, _mock_qd):
        """Non-empty PortAudio status in callback triggers log.warning."""
        from lazy_take_notes.l3_interface_adapters.gateways.sounddevice_audio_source import SounddeviceAudioSource

        mock_stream_cls.return_value = MagicMock()

        src = SounddeviceAudioSource()
        src.open(16000, 1)

        callback = mock_stream_cls.call_args.kwargs['callback']
        fake_data = np.array([[0.1], [0.2]], dtype=np.float32)
        callback(fake_data, 2, None, 'input overflow')

        result = src.read(timeout=0.1)
        assert result is not None
        np.testing.assert_allclose(result, [0.1, 0.2], atol=1e-6)

    def test_close_when_none_stream_is_noop(self):
        from lazy_take_notes.l3_interface_adapters.gateways.sounddevice_audio_source import SounddeviceAudioSource

        src = SounddeviceAudioSource()
        # Should not raise
        src.close()
        assert src._stream is None
