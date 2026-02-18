"""Tests for DigestState entity."""

from lazy_take_notes.l1_entities.digest_state import DigestState


class TestDigestState:
    def test_init_messages(self):
        state = DigestState()
        state.init_messages('You are a helpful assistant.')
        assert len(state.messages) == 1
        assert state.messages[0].role == 'system'

    def test_default_values(self):
        state = DigestState()
        assert state.buffer == []
        assert state.all_lines == []
        assert state.digest_count == 0
        assert state.consecutive_failures == 0
        assert state.prompt_tokens == 0
