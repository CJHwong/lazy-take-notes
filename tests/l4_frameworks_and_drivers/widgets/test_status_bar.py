"""Tests for StatusBar helper — _rms_to_char dB-scaled level meter."""

from lazy_take_notes.l4_frameworks_and_drivers.widgets.status_bar import (
    _rms_to_char,  # noqa: PLC2701 -- testing module-private helper directly
)


class TestRmsToChar:
    """Cover the dB-scaled _rms_to_char mapping."""

    def test_silence_returns_lowest_bar(self):
        assert _rms_to_char(0.0) == '▁'

    def test_near_zero_returns_lowest_bar(self):
        assert _rms_to_char(1e-8) == '▁'

    def test_quiet_ambient_mic(self):
        # ~0.005 RMS ≈ -46 dB → should be above baseline
        char = _rms_to_char(0.005)
        assert char in ('▂', '▃')

    def test_normal_speech_mic(self):
        # ~0.05 RMS ≈ -26 dB → mid-range bar
        char = _rms_to_char(0.05)
        assert char in ('▅', '▆')

    def test_loud_system_audio(self):
        # ~0.3 RMS ≈ -10 dB → highest bar
        assert _rms_to_char(0.3) == '█'

    def test_monotonically_increasing(self):
        levels = [0.001, 0.01, 0.05, 0.1, 0.3]
        chars = [_rms_to_char(r) for r in levels]
        indices = ['▁▂▃▄▅▆▇█'.index(c) for c in chars]
        assert indices == sorted(indices)
        assert indices[-1] > indices[0]
