"""Tests for SounddeviceAudioSource gateway — patches sd.InputStream."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

MODULE = 'lazy_take_notes.l3_interface_adapters.gateways.sounddevice_audio_source'


_FAKE_DEVICE_INFO = {'name': 'Test Input', 'default_samplerate': 48000.0}


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
        # Simulate audio data arriving
        fake_data = np.array([[0.1], [0.2]], dtype=np.float32)
        callback(fake_data, 2, None, None)

        # Data should be in the queue
        result = src.read(timeout=0.1)
        assert result is not None
        np.testing.assert_allclose(result, [0.1, 0.2], atol=1e-6)

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
    def test_drain_concatenates_chunks(self, mock_stream_cls, _mock_qd):
        from lazy_take_notes.l3_interface_adapters.gateways.sounddevice_audio_source import SounddeviceAudioSource

        mock_stream_cls.return_value = MagicMock()

        src = SounddeviceAudioSource()
        src.open(16000, 1)

        callback = mock_stream_cls.call_args.kwargs['callback']
        callback(np.array([[0.1], [0.2]], dtype=np.float32), 2, None, None)
        callback(np.array([[0.3], [0.4]], dtype=np.float32), 2, None, None)

        result = src.drain()
        assert result is not None
        np.testing.assert_allclose(result, [0.1, 0.2, 0.3, 0.4], atol=1e-6)

    @patch(f'{MODULE}.sd.query_devices', return_value=_FAKE_DEVICE_INFO)
    @patch(f'{MODULE}.sd.InputStream')
    def test_drain_empty_returns_none(self, mock_stream_cls, _mock_qd):
        from lazy_take_notes.l3_interface_adapters.gateways.sounddevice_audio_source import SounddeviceAudioSource

        mock_stream_cls.return_value = MagicMock()

        src = SounddeviceAudioSource()
        src.open(16000, 1)

        result = src.drain()
        assert result is None

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
