"""Tests for SubprocessWhisperTranscriber — mocks the subprocess layer."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from lazy_take_notes.l1_entities.transcript import TranscriptSegment
from lazy_take_notes.l3_interface_adapters.gateways.subprocess_whisper_transcriber import (
    SubprocessWhisperTranscriber,
)


def _make_ctx(parent_responses: list[dict]) -> tuple[MagicMock, MagicMock, MagicMock]:
    """Build a mock mp context with Pipe returning a controlled parent Connection.

    parent_responses: sequence of dicts returned by parent_conn.recv() in order.
    parent_conn.poll() always returns True (data immediately available).
    """
    parent_conn = MagicMock()
    parent_conn.poll.return_value = True
    parent_conn.recv.side_effect = parent_responses

    child_conn = MagicMock()

    process = MagicMock()
    process.is_alive.return_value = False

    ctx = MagicMock()
    ctx.Pipe.return_value = (parent_conn, child_conn)
    ctx.Process.return_value = process

    return ctx, parent_conn, process


class TestSubprocessWhisperTranscriberLifecycle:
    def test_load_model_starts_process(self):
        ctx, parent_conn, process = _make_ctx([{'status': 'ready'}])
        with patch(
            'lazy_take_notes.l3_interface_adapters.gateways.subprocess_whisper_transcriber.mp.get_context',
            return_value=ctx,
        ):
            t = SubprocessWhisperTranscriber()
            t.load_model('/fake/model.bin')

        process.start.assert_called_once()
        parent_conn.poll.assert_called_once_with(timeout=120)
        parent_conn.recv.assert_called_once()

    def test_load_model_raises_on_subprocess_error(self):
        ctx, parent_conn, process = _make_ctx([{'status': 'error', 'error': 'GGML not found'}])
        with patch(
            'lazy_take_notes.l3_interface_adapters.gateways.subprocess_whisper_transcriber.mp.get_context',
            return_value=ctx,
        ):
            t = SubprocessWhisperTranscriber()
            with pytest.raises(RuntimeError, match='GGML not found'):
                t.load_model('/bad/path.bin')

    def test_load_model_raises_on_timeout(self):
        ctx, parent_conn, process = _make_ctx([])
        parent_conn.poll.return_value = False  # simulate timeout
        with patch(
            'lazy_take_notes.l3_interface_adapters.gateways.subprocess_whisper_transcriber.mp.get_context',
            return_value=ctx,
        ):
            t = SubprocessWhisperTranscriber()
            with pytest.raises(RuntimeError, match='Timeout'):
                t.load_model('/fake/model.bin')

    def test_load_model_raises_on_eof(self):
        ctx, parent_conn, process = _make_ctx([])
        parent_conn.poll.return_value = True
        parent_conn.recv.side_effect = EOFError
        with patch(
            'lazy_take_notes.l3_interface_adapters.gateways.subprocess_whisper_transcriber.mp.get_context',
            return_value=ctx,
        ):
            t = SubprocessWhisperTranscriber()
            with pytest.raises(RuntimeError, match='exited unexpectedly'):
                t.load_model('/fake/model.bin')

    def test_close_sends_shutdown_and_joins(self):
        ctx, parent_conn, process = _make_ctx([{'status': 'ready'}])
        with patch(
            'lazy_take_notes.l3_interface_adapters.gateways.subprocess_whisper_transcriber.mp.get_context',
            return_value=ctx,
        ):
            t = SubprocessWhisperTranscriber()
            t.load_model('/fake/model.bin')
            t.close()

        parent_conn.send.assert_called_with(None)
        parent_conn.close.assert_called()
        process.join.assert_called_once_with(timeout=5)
        assert t._process is None
        assert t._conn is None

    def test_close_terminates_if_alive(self):
        ctx, parent_conn, process = _make_ctx([{'status': 'ready'}])
        process.is_alive.return_value = True
        with patch(
            'lazy_take_notes.l3_interface_adapters.gateways.subprocess_whisper_transcriber.mp.get_context',
            return_value=ctx,
        ):
            t = SubprocessWhisperTranscriber()
            t.load_model('/fake/model.bin')
            t.close()

        process.terminate.assert_called_once()


class TestSubprocessWhisperTranscriberTranscribe:
    def test_transcribe_returns_segments(self):
        seg = TranscriptSegment(text='hello', wall_start=0.0, wall_end=1.0)
        ctx, parent_conn, process = _make_ctx(
            [
                {'status': 'ready'},
                {'status': 'ok', 'segments': [seg]},
            ]
        )
        with patch(
            'lazy_take_notes.l3_interface_adapters.gateways.subprocess_whisper_transcriber.mp.get_context',
            return_value=ctx,
        ):
            t = SubprocessWhisperTranscriber()
            t.load_model('/fake/model.bin')
            audio = np.zeros(16000, dtype=np.float32)
            segs = t.transcribe(audio, language='zh', hints=['hint'])

        assert segs == [seg]
        parent_conn.send.assert_any_call({'audio': audio, 'language': 'zh', 'hints': ['hint']})

    def test_transcribe_raises_on_error_response(self):
        ctx, parent_conn, process = _make_ctx(
            [
                {'status': 'ready'},
                {'status': 'error', 'error': 'inference crashed'},
            ]
        )
        with patch(
            'lazy_take_notes.l3_interface_adapters.gateways.subprocess_whisper_transcriber.mp.get_context',
            return_value=ctx,
        ):
            t = SubprocessWhisperTranscriber()
            t.load_model('/fake/model.bin')
            with pytest.raises(RuntimeError, match='inference crashed'):
                t.transcribe(np.zeros(16000, dtype=np.float32), language='en')

    def test_transcribe_raises_on_timeout(self):
        ctx, parent_conn, process = _make_ctx([{'status': 'ready'}])
        # Second poll (for transcription) times out
        parent_conn.poll.side_effect = [True, False]
        with patch(
            'lazy_take_notes.l3_interface_adapters.gateways.subprocess_whisper_transcriber.mp.get_context',
            return_value=ctx,
        ):
            t = SubprocessWhisperTranscriber()
            t.load_model('/fake/model.bin')
            with pytest.raises(RuntimeError, match='Timeout'):
                t.transcribe(np.zeros(16000, dtype=np.float32), language='en')

    def test_transcribe_raises_on_eof(self):
        ctx, parent_conn, process = _make_ctx([{'status': 'ready'}])
        parent_conn.poll.side_effect = [True, True]
        parent_conn.recv.side_effect = [{'status': 'ready'}, EOFError]
        with patch(
            'lazy_take_notes.l3_interface_adapters.gateways.subprocess_whisper_transcriber.mp.get_context',
            return_value=ctx,
        ):
            t = SubprocessWhisperTranscriber()
            t.load_model('/fake/model.bin')
            with pytest.raises(RuntimeError, match='exited unexpectedly'):
                t.transcribe(np.zeros(16000, dtype=np.float32), language='en')

    def test_transcribe_before_load_raises(self):
        t = SubprocessWhisperTranscriber()
        with pytest.raises(RuntimeError, match='Model not loaded'):
            t.transcribe(np.zeros(16000, dtype=np.float32), language='en')


class TestSubprocessEntry:
    """Tests for _subprocess_entry directly — patches os and WhisperTranscriber via module objects."""

    def test_happy_path_ready_transcribe_shutdown(self):
        import lazy_take_notes.l3_interface_adapters.gateways.subprocess_whisper_transcriber as sp_mod
        import lazy_take_notes.l3_interface_adapters.gateways.whisper_transcriber as wt_mod
        from lazy_take_notes.l3_interface_adapters.gateways.subprocess_whisper_transcriber import (
            _subprocess_entry,  # noqa: PLC2701 -- testing private entry point
        )

        conn = MagicMock()
        audio = np.zeros(16000, dtype=np.float32)
        seg = TranscriptSegment(text='hi', wall_start=0.0, wall_end=1.0)

        conn.recv.side_effect = [
            {'audio': audio, 'language': 'en', 'hints': []},
            None,
        ]

        mock_transcriber = MagicMock()
        mock_transcriber.transcribe.return_value = [seg]

        mock_os = MagicMock()
        mock_os.open.return_value = 99
        mock_os.devnull = '/dev/null'
        mock_os.O_WRONLY = 1

        with (
            patch.object(sp_mod, 'os', mock_os),
            patch.object(wt_mod, 'WhisperTranscriber', return_value=mock_transcriber),
        ):
            _subprocess_entry('/model.bin', conn)

        assert conn.send.call_count == 2
        first_send = conn.send.call_args_list[0][0][0]
        assert first_send == {'status': 'ready'}
        second_send = conn.send.call_args_list[1][0][0]
        assert second_send['status'] == 'ok'
        conn.close.assert_called_once()

    def test_init_failure_sends_error_and_closes(self):
        import lazy_take_notes.l3_interface_adapters.gateways.subprocess_whisper_transcriber as sp_mod
        import lazy_take_notes.l3_interface_adapters.gateways.whisper_transcriber as wt_mod
        from lazy_take_notes.l3_interface_adapters.gateways.subprocess_whisper_transcriber import (
            _subprocess_entry,  # noqa: PLC2701 -- testing private entry point
        )

        conn = MagicMock()

        mock_transcriber = MagicMock()
        mock_transcriber.load_model.side_effect = RuntimeError('GGML init failed')

        mock_os = MagicMock()
        mock_os.open.return_value = 99
        mock_os.devnull = '/dev/null'
        mock_os.O_WRONLY = 1

        with (
            patch.object(sp_mod, 'os', mock_os),
            patch.object(wt_mod, 'WhisperTranscriber', return_value=mock_transcriber),
        ):
            _subprocess_entry('/bad.bin', conn)

        sent = conn.send.call_args_list[0][0][0]
        assert sent['status'] == 'error'
        assert 'GGML init failed' in sent['error']
        conn.close.assert_called_once()

    def test_transcribe_exception_sends_error(self):
        import lazy_take_notes.l3_interface_adapters.gateways.subprocess_whisper_transcriber as sp_mod
        import lazy_take_notes.l3_interface_adapters.gateways.whisper_transcriber as wt_mod
        from lazy_take_notes.l3_interface_adapters.gateways.subprocess_whisper_transcriber import (
            _subprocess_entry,  # noqa: PLC2701 -- testing private entry point
        )

        conn = MagicMock()
        conn.recv.side_effect = [
            {'audio': np.zeros(100, dtype=np.float32), 'language': 'en'},
            None,
        ]

        mock_transcriber = MagicMock()
        mock_transcriber.transcribe.side_effect = RuntimeError('inference OOM')

        mock_os = MagicMock()
        mock_os.open.return_value = 99
        mock_os.devnull = '/dev/null'
        mock_os.O_WRONLY = 1

        with (
            patch.object(sp_mod, 'os', mock_os),
            patch.object(wt_mod, 'WhisperTranscriber', return_value=mock_transcriber),
        ):
            _subprocess_entry('/model.bin', conn)

        assert conn.send.call_count == 2
        error_send = conn.send.call_args_list[1][0][0]
        assert error_send['status'] == 'error'
        assert 'OOM' in error_send['error']


class TestCloseEdgeCases:
    def test_close_send_none_raises_oserror(self):
        ctx, parent_conn, process = _make_ctx([{'status': 'ready'}])
        with patch(
            'lazy_take_notes.l3_interface_adapters.gateways.subprocess_whisper_transcriber.mp.get_context',
            return_value=ctx,
        ):
            t = SubprocessWhisperTranscriber()
            t.load_model('/fake/model.bin')

            parent_conn.send.side_effect = OSError('Broken pipe')
            # Should not raise — close handles OSError gracefully
            t.close()

        assert t._conn is None
        assert t._process is None

    def test_close_conn_close_raises(self):
        ctx, parent_conn, process = _make_ctx([{'status': 'ready'}])
        with patch(
            'lazy_take_notes.l3_interface_adapters.gateways.subprocess_whisper_transcriber.mp.get_context',
            return_value=ctx,
        ):
            t = SubprocessWhisperTranscriber()
            t.load_model('/fake/model.bin')

            parent_conn.close.side_effect = OSError('already closed')
            # Should not raise
            t.close()

        assert t._conn is None
