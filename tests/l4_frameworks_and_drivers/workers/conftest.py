"""Worker tests feed constant blocks to test the plumbing; the speech gate has its own L2 tests."""

import pytest

from lazy_take_notes.l2_use_cases import transcribe_audio_use_case
from tests.conftest import level_only_voiced_seconds


@pytest.fixture(autouse=True)
def _no_speech_gate(monkeypatch):
    monkeypatch.setattr(transcribe_audio_use_case, 'MIN_VOICED_S', 0.0)
    monkeypatch.setattr(transcribe_audio_use_case, 'voiced_seconds', level_only_voiced_seconds)
