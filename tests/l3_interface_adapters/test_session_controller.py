"""Tests for SessionController — uses fakes, not mocks of concrete libs."""

from __future__ import annotations

import pytest

from lazy_take_notes.l1_entities.transcript import TranscriptSegment
from lazy_take_notes.l3_interface_adapters.controllers.session_controller import SessionController
from lazy_take_notes.l3_interface_adapters.gateways.yaml_template_loader import YamlTemplateLoader
from lazy_take_notes.l4_frameworks_and_drivers.infra_config import build_app_config
from tests.conftest import FakeLLMClient, FakePersistence

VALID_DIGEST = '## Topic\nSome content'


@pytest.fixture
def controller():
    config = build_app_config({})
    template = YamlTemplateLoader().load('default_zh_tw')
    fake_llm = FakeLLMClient(response=VALID_DIGEST, prompt_tokens=50)
    fake_persist = FakePersistence()
    ctrl = SessionController(
        config=config,
        template=template,
        llm_client=fake_llm,
        persistence=fake_persist,
    )
    return ctrl, fake_llm, fake_persist


class TestOnTranscriptSegments:
    def test_buffers_segments(self, controller):
        ctrl, _, _ = controller
        segs = [
            TranscriptSegment(text='Hello', wall_start=0.0, wall_end=1.0),
            TranscriptSegment(text='World', wall_start=1.0, wall_end=2.0),
        ]
        ctrl.on_transcript_segments(segs)

        assert len(ctrl.all_segments) == 2
        assert len(ctrl.digest_state.buffer) == 2
        assert ctrl.digest_state.buffer[0] == 'Hello'

    def test_persists_transcript(self, controller):
        ctrl, _, fake_persist = controller
        segs = [TranscriptSegment(text='Test', wall_start=0.0, wall_end=1.0)]
        ctrl.on_transcript_segments(segs)

        assert len(fake_persist.transcript_calls) == 1

    def test_returns_trigger_decision(self, controller):
        ctrl, _, _ = controller
        # Not enough lines for trigger (default min_lines=15)
        segs = [TranscriptSegment(text='Line', wall_start=0.0, wall_end=1.0)]
        assert ctrl.on_transcript_segments(segs) is False


class TestRunDigest:
    @pytest.mark.asyncio
    async def test_success(self, controller):
        ctrl, _, fake_persist = controller
        # Buffer some content
        ctrl.digest_state.buffer = [f'Line {i}' for i in range(20)]
        ctrl.digest_state.all_lines = ctrl.digest_state.buffer.copy()

        result = await ctrl.run_digest()

        assert result.ok
        assert ctrl.latest_digest is not None
        assert len(fake_persist.digest_calls) == 1
        assert len(fake_persist.history_calls) == 1

    @pytest.mark.asyncio
    async def test_final_digest(self, controller):
        ctrl, fake_llm, fake_persist = controller
        ctrl.digest_state.buffer = ['Line 1']
        ctrl.digest_state.all_lines = ['Line 1', 'Line 2']

        result = await ctrl.run_digest(is_final=True)

        assert result.ok
        # Verify final flag was passed to persistence
        _, _, is_final = fake_persist.history_calls[0]
        assert is_final is True


class TestRunQuickAction:
    @pytest.mark.asyncio
    async def test_existing_key(self, controller):
        ctrl, _, _ = controller
        template = ctrl._template
        first_key = template.quick_actions[0].key

        result = await ctrl.run_quick_action(first_key)
        assert result is not None

    @pytest.mark.asyncio
    async def test_unknown_key(self, controller):
        ctrl, _, _ = controller
        result = await ctrl.run_quick_action('nonexistent')
        assert result is None
