"""Tests for CompactMessagesUseCase."""

from lazy_take_notes.l1_entities.chat_message import ChatMessage
from lazy_take_notes.l1_entities.digest_state import DigestState
from lazy_take_notes.l2_use_cases.compact_messages_use_case import CompactMessagesUseCase

VALID_DIGEST_RESPONSE = """\
## 目前議題
討論 REST vs GraphQL API 設計方案。

## 已結案議題
### 開場介紹
**結論：** 完成

## 待釐清問題
- 要選哪個框架？

## 建議行動
- **進行效能測試**

## 停車場
（無）
"""


class TestCompactMessages:
    def test_compacts_to_three_messages(self):
        state = DigestState()
        state.init_messages('System prompt')
        state.messages.extend(
            [
                ChatMessage(role='user', content='some transcript'),
                ChatMessage(role='assistant', content='some digest'),
                ChatMessage(role='user', content='more transcript'),
                ChatMessage(role='assistant', content='updated digest'),
            ]
        )

        uc = CompactMessagesUseCase()
        uc.execute(state, VALID_DIGEST_RESPONSE, 'System prompt')

        assert len(state.messages) == 3
        assert state.messages[0].role == 'system'
        assert state.messages[1].role == 'user'
        assert state.messages[2].role == 'assistant'
        assert state.prompt_tokens == 0

    def test_compact_contains_markdown(self):
        state = DigestState()
        state.init_messages('System prompt')

        uc = CompactMessagesUseCase()
        uc.execute(state, VALID_DIGEST_RESPONSE, 'System prompt')

        user_msg = state.messages[1].content
        assert '目前議題' in user_msg
        assert state.messages[2].content == VALID_DIGEST_RESPONSE
