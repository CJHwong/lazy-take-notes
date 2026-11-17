"""Tests for the Audio tab."""

from __future__ import annotations

from textual.app import App, ComposeResult
from textual.widgets import Button, Select, Static

from lazy_take_notes.l4_frameworks_and_drivers.apps.config.audio_tab import AudioTab, LevelMeter
from lazy_take_notes.l4_frameworks_and_drivers.messages import (
    AudioProbeDone,
    AudioProbeLevel,
    AudioProbePermission,
)


class Host(App):
    def __init__(self, raw: dict) -> None:
        super().__init__()
        self._raw = raw

    def compose(self) -> ComposeResult:
        yield AudioTab(self._raw)


RAW = {
    'transcription': {'silence_threshold': 0.01, 'input_device': None},
    'diarization': {'enabled': False, 'binary': '', 'model': '', 'device': 'auto', 'interval_s': 60},
}


class TestAudioTab:
    async def test_renders_probe_and_device(self):
        app = Host(RAW)
        async with app.run_test(size=(120, 50)) as pilot:
            await pilot.pause()
            assert app.query_one('#audio-start', Button)
            assert app.query_one('#audio-meter-mic', LevelMeter)
            assert app.query_one('#cfg-audio-device', Select)

    async def test_default_device_is_none(self):
        app = Host(RAW)
        async with app.run_test(size=(120, 50)) as pilot:
            await pilot.pause()
            assert app.query_one(AudioTab).device_value() is None

    async def test_level_updates_the_meter(self):
        app = Host(RAW)
        async with app.run_test(size=(120, 50)) as pilot:
            await pilot.pause()
            app.query_one(AudioTab).post_message(AudioProbeLevel('mic', 0.5))
            await pilot.pause()
            text = str(app.query_one('#audio-meter-mic', LevelMeter).render())
            assert '█' in text

    async def test_permission_updates_the_label(self):
        app = Host(RAW)
        async with app.run_test(size=(120, 50)) as pilot:
            await pilot.pause()
            app.query_one(AudioTab).post_message(AudioProbePermission('system', 'waiting'))
            await pilot.pause()
            label = app.query_one('#audio-perm-system', Static)
            assert str(label.render()) == 'Waiting for permission'
            assert label.has_class('waiting')

    async def test_done_reenables_start(self):
        app = Host(RAW)
        async with app.run_test(size=(120, 50)) as pilot:
            await pilot.pause()
            start = app.query_one('#audio-start', Button)
            start.disabled = True
            app.query_one(AudioTab).post_message(AudioProbeDone())
            await pilot.pause()
            assert start.disabled is False


class TestAudioTabFaults:
    async def test_markup_like_device_name_does_not_crash(self, monkeypatch):
        weird = 'USB Mic [/x] (2)'
        monkeypatch.setattr(
            'lazy_take_notes.l4_frameworks_and_drivers.apps.config.audio_tab.SounddeviceAudioSource.list_input_devices',
            staticmethod(lambda: [weird]),
        )
        raw = {**RAW, 'transcription': {'silence_threshold': 0.01, 'input_device': weird}}
        app = Host(raw)
        async with app.run_test(size=(120, 50)) as pilot:
            await pilot.pause()
            assert app.query_one(AudioTab).device_value() == weird

    async def test_source_build_failure_notifies(self, monkeypatch):
        def _boom(*_args, **_kwargs):
            raise ImportError('soundcard missing')

        monkeypatch.setattr(
            'lazy_take_notes.l4_frameworks_and_drivers.apps.config.audio_tab.DependencyContainer.build_audio_sources',
            _boom,
        )
        app = Host(RAW)
        async with app.run_test(size=(120, 50)) as pilot:
            await pilot.pause()
            app.query_one(AudioTab)._start_probe()
            await pilot.pause()
            assert app.query_one('#audio-start', Button).disabled is False
            assert any('soundcard missing' in str(n.message) for n in app._notifications)
