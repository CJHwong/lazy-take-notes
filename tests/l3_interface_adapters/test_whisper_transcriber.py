"""Tests for WhisperTranscriber gateway — patches pywhispercpp.model.Model."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

MODULE = 'lazy_take_notes.l3_interface_adapters.gateways.whisper_transcriber'


@patch(f'{MODULE}.os.close')
@patch(f'{MODULE}.os.dup2')
@patch(f'{MODULE}.os.dup')
@patch(f'{MODULE}.os.open', return_value=99)
class TestSuppressCStdout:
    def test_redirects_and_restores_fds(self, mock_open, mock_dup, mock_dup2, mock_close):
        from lazy_take_notes.l3_interface_adapters.gateways.whisper_transcriber import (
            _suppress_c_stdout,  # noqa: PLC2701 -- testing private helper
        )

        mock_dup.side_effect = [10, 11]  # saved stdout, saved stderr

        with _suppress_c_stdout():
            pass

        # Verify devnull opened and redirected to fd 1 and 2
        assert mock_dup2.call_count == 4  # 2 redirects in + 2 restores out
        # Verify cleanup: devnull, old_stdout, old_stderr closed
        assert mock_close.call_count == 3


@patch(f'{MODULE}.os.close')
@patch(f'{MODULE}.os.dup2')
@patch(f'{MODULE}.os.dup', side_effect=[10, 11])
@patch(f'{MODULE}.os.open', return_value=99)
@patch(f'{MODULE}.Model')
class TestLoadModel:
    def test_calls_model_with_correct_args(self, mock_model_cls, _open, _dup, _dup2, _close):
        from lazy_take_notes.l3_interface_adapters.gateways.whisper_transcriber import WhisperTranscriber

        t = WhisperTranscriber()
        t.load_model('/path/to/model.bin')

        mock_model_cls.assert_called_once_with('/path/to/model.bin', print_progress=False, print_realtime=False)


@patch(f'{MODULE}.os.close')
@patch(f'{MODULE}.os.dup2')
@patch(f'{MODULE}.os.dup', side_effect=[10, 11])
@patch(f'{MODULE}.os.open', return_value=99)
@patch(f'{MODULE}.Model')
class TestClose:
    def test_close_with_model_deletes_it(self, mock_model_cls, _open, _dup, _dup2, _close):
        from lazy_take_notes.l3_interface_adapters.gateways.whisper_transcriber import WhisperTranscriber

        t = WhisperTranscriber()
        t.load_model('/path/model.bin')
        assert t._model is not None

        # Reset mocks for close (which also uses _suppress_c_stdout)
        _dup.side_effect = [10, 11]
        t.close()
        assert t._model is None

    def test_close_without_model_is_noop(self, mock_model_cls, _open, _dup, _dup2, _close):
        from lazy_take_notes.l3_interface_adapters.gateways.whisper_transcriber import WhisperTranscriber

        t = WhisperTranscriber()
        # Should not raise
        t.close()
        assert t._model is None


class TestTranscribe:
    def test_raises_before_load(self):
        from lazy_take_notes.l3_interface_adapters.gateways.whisper_transcriber import WhisperTranscriber

        t = WhisperTranscriber()
        with pytest.raises(RuntimeError, match='Model not loaded'):
            t.transcribe(np.zeros(16000, dtype=np.float32), language='en')

    @patch(f'{MODULE}.os.close')
    @patch(f'{MODULE}.os.dup2')
    @patch(f'{MODULE}.os.dup', side_effect=[10, 11, 10, 11])  # load + transcribe
    @patch(f'{MODULE}.os.open', return_value=99)
    @patch(f'{MODULE}.Model')
    def test_empty_hints_omits_initial_prompt(self, mock_model_cls, _open, _dup, _dup2, _close):
        from lazy_take_notes.l3_interface_adapters.gateways.whisper_transcriber import WhisperTranscriber

        mock_instance = MagicMock()
        mock_instance.transcribe.return_value = []
        mock_model_cls.return_value = mock_instance

        t = WhisperTranscriber()
        t.load_model('/m.bin')
        t.transcribe(np.zeros(16000, dtype=np.float32), language='en', hints=[])

        call_kwargs = mock_instance.transcribe.call_args.kwargs
        assert 'initial_prompt' not in call_kwargs

    @patch(f'{MODULE}.os.close')
    @patch(f'{MODULE}.os.dup2')
    @patch(f'{MODULE}.os.dup', side_effect=[10, 11, 10, 11])
    @patch(f'{MODULE}.os.open', return_value=99)
    @patch(f'{MODULE}.Model')
    def test_with_hints_sets_initial_prompt(self, mock_model_cls, _open, _dup, _dup2, _close):
        from lazy_take_notes.l3_interface_adapters.gateways.whisper_transcriber import WhisperTranscriber

        mock_instance = MagicMock()
        mock_instance.transcribe.return_value = []
        mock_model_cls.return_value = mock_instance

        t = WhisperTranscriber()
        t.load_model('/m.bin')
        t.transcribe(np.zeros(16000, dtype=np.float32), language='zh', hints=['hello', 'world'])

        call_kwargs = mock_instance.transcribe.call_args.kwargs
        assert call_kwargs['initial_prompt'] == 'hello world'

    @patch(f'{MODULE}.os.close')
    @patch(f'{MODULE}.os.dup2')
    @patch(f'{MODULE}.os.dup', side_effect=[10, 11, 10, 11])
    @patch(f'{MODULE}.os.open', return_value=99)
    @patch(f'{MODULE}.Model')
    def test_centisecond_conversion_and_empty_text_filtering(self, mock_model_cls, _open, _dup, _dup2, _close):
        from lazy_take_notes.l3_interface_adapters.gateways.whisper_transcriber import WhisperTranscriber

        seg_good = MagicMock()
        seg_good.text = ' Hello world '
        seg_good.t0 = 100  # centiseconds → 1.0s
        seg_good.t1 = 250  # → 2.5s

        seg_empty = MagicMock()
        seg_empty.text = '   '  # whitespace-only → filtered out
        seg_empty.t0 = 250
        seg_empty.t1 = 300

        mock_instance = MagicMock()
        mock_instance.transcribe.return_value = [seg_good, seg_empty]
        mock_model_cls.return_value = mock_instance

        t = WhisperTranscriber()
        t.load_model('/m.bin')
        result = t.transcribe(np.zeros(16000, dtype=np.float32), language='en')

        assert len(result) == 1
        assert result[0].text == 'Hello world'
        assert result[0].wall_start == pytest.approx(1.0)
        assert result[0].wall_end == pytest.approx(2.5)
