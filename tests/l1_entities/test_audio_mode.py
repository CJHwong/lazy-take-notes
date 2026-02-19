"""Tests for AudioMode enum."""

from __future__ import annotations

from lazy_take_notes.l1_entities.audio_mode import AudioMode


class TestAudioMode:
    def test_all_members_present(self):
        assert AudioMode.MIC_ONLY.value == 'mic_only'
        assert AudioMode.SYSTEM_ONLY.value == 'system_only'
        assert AudioMode.MIX.value == 'mix'

    def test_member_count(self):
        assert len(AudioMode) == 3

    def test_from_value(self):
        assert AudioMode('mic_only') is AudioMode.MIC_ONLY
        assert AudioMode('system_only') is AudioMode.SYSTEM_ONLY
        assert AudioMode('mix') is AudioMode.MIX
