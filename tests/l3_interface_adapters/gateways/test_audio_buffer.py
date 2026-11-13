"""Duration bounds for audio producer buffers."""

import queue

import numpy as np
import pytest

from lazy_take_notes.l3_interface_adapters.gateways._audio_buffer import (  # noqa: PLC2701 -- exercise private adapter buffer
    AudioBuffer,
)


def test_variable_chunks_evict_oldest_frames(caplog):
    buffer = AudioBuffer(max_frames=8, sample_rate=8)
    buffer.put(np.ones(3, dtype=np.float32))
    buffer.put(np.full(5, 2, dtype=np.float32))
    buffer.put(np.full(4, 3, dtype=np.float32))

    assert buffer.frames == 4
    assert buffer.dropped_frames == 8
    np.testing.assert_array_equal(buffer.get(0), np.full(4, 3, dtype=np.float32))
    assert buffer.frames == 0
    assert 'dropped' in caplog.text


def test_oversized_chunk_keeps_only_recent_frames():
    buffer = AudioBuffer(max_frames=4, sample_rate=4)
    buffer.put(np.arange(10, dtype=np.float32))
    result = buffer.get(0)
    np.testing.assert_array_equal(result, [6, 7, 8, 9])
    assert result.flags.owndata
    assert buffer.dropped_frames == 6


def test_clear_resets_queued_frames_without_rewriting_drop_count():
    buffer = AudioBuffer(max_frames=4, sample_rate=4)
    buffer.put(np.ones(8, dtype=np.float32))
    buffer.clear()
    assert buffer.frames == 0
    assert buffer.dropped_frames == 4
    with pytest.raises(queue.Empty):
        buffer.get(0)


def test_large_producer_run_remains_bounded():
    buffer = AudioBuffer(max_frames=16000, sample_rate=16000)
    for sequence in range(10000):
        buffer.put(np.full(512, sequence, dtype=np.float32))
    assert buffer.frames <= 16000
    assert buffer.dropped_frames > 0


def test_invalid_budget_is_rejected():
    with pytest.raises(ValueError, match='must be positive'):
        AudioBuffer(max_frames=0, sample_rate=16000)


def test_empty_chunks_do_not_accumulate():
    buffer = AudioBuffer(max_frames=4, sample_rate=4)
    for _ in range(100):
        buffer.put(np.array([], dtype=np.float32))
    assert buffer.empty()
    assert buffer.qsize() == 0
