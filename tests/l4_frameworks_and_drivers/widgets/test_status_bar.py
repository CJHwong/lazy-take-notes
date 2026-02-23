"""Tests for StatusBar helper — _rms_to_char dB-scaled level meter and transcribing indicator."""

from unittest.mock import patch

import pytest

from lazy_take_notes.l4_frameworks_and_drivers.apps.record import RecordApp
from lazy_take_notes.l4_frameworks_and_drivers.widgets.status_bar import (
    StatusBar,
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


def _make_app(tmp_path):
    from lazy_take_notes.l3_interface_adapters.controllers.session_controller import SessionController
    from lazy_take_notes.l3_interface_adapters.gateways.yaml_template_loader import YamlTemplateLoader
    from lazy_take_notes.l4_frameworks_and_drivers.config import build_app_config
    from tests.conftest import FakeLLMClient, FakePersistence

    config = build_app_config({})
    template = YamlTemplateLoader().load('default_en')
    output_dir = tmp_path / 'output'
    output_dir.mkdir()
    controller = SessionController(
        config=config,
        template=template,
        llm_client=FakeLLMClient(),
        persistence=FakePersistence(output_dir),
    )
    return RecordApp(config=config, template=template, output_dir=output_dir, controller=controller)


class TestTranscribingIndicator:
    @pytest.mark.asyncio
    async def test_render_shows_transcribing_when_active(self, tmp_path):
        app = _make_app(tmp_path)
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as _pilot:
                bar = app.query_one('#status-bar', StatusBar)
                bar.transcribing = True
                rendered = bar.render()
                assert 'Transcribing' in rendered

    @pytest.mark.asyncio
    async def test_render_hides_transcribing_when_inactive(self, tmp_path):
        app = _make_app(tmp_path)
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as _pilot:
                bar = app.query_one('#status-bar', StatusBar)
                bar.transcribing = False
                rendered = bar.render()
                assert 'Transcribing' not in rendered

    @pytest.mark.asyncio
    async def test_transcribing_and_activity_both_shown(self, tmp_path):
        app = _make_app(tmp_path)
        with patch.object(app, '_start_audio_worker'):
            async with app.run_test() as _pilot:
                bar = app.query_one('#status-bar', StatusBar)
                bar.transcribing = True
                bar.activity = 'Digesting...'
                rendered = bar.render()
                assert 'Transcribing' in rendered
                assert 'Digesting' in rendered
