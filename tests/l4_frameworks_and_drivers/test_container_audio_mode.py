"""Tests for DependencyContainer audio mode wiring."""

from __future__ import annotations

import sys

import pytest

from lazy_take_notes.l1_entities.audio_mode import AudioMode
from lazy_take_notes.l3_interface_adapters.gateways.mixed_audio_source import MixedAudioSource
from lazy_take_notes.l3_interface_adapters.gateways.sounddevice_audio_source import SounddeviceAudioSource
from lazy_take_notes.l3_interface_adapters.gateways.yaml_template_loader import YamlTemplateLoader
from lazy_take_notes.l4_frameworks_and_drivers.config import build_app_config
from lazy_take_notes.l4_frameworks_and_drivers.container import DependencyContainer


@pytest.fixture
def basic_container_args(tmp_path):
    config = build_app_config({})
    template = YamlTemplateLoader().load('default_en')
    return config, template, tmp_path / 'out'


class TestContainerAudioMode:
    def test_mic_only_wires_sounddevice_source(self, basic_container_args):
        config, template, out_dir = basic_container_args
        container = DependencyContainer(config, template, out_dir, audio_mode=AudioMode.MIC_ONLY)
        assert isinstance(container.audio_source, SounddeviceAudioSource)

    @pytest.mark.skipif(sys.platform != 'darwin', reason='CoreAudioTapSource is macOS only')
    def test_system_only_wires_coreaudio_tap_source_on_macos(self, basic_container_args):
        from lazy_take_notes.l3_interface_adapters.gateways.coreaudio_tap_source import CoreAudioTapSource

        config, template, out_dir = basic_container_args
        container = DependencyContainer(config, template, out_dir, audio_mode=AudioMode.SYSTEM_ONLY)
        assert isinstance(container.audio_source, CoreAudioTapSource)

    @pytest.mark.skipif(sys.platform != 'darwin', reason='CoreAudioTapSource is macOS only')
    def test_mix_wires_mixed_audio_source_on_macos(self, basic_container_args):
        config, template, out_dir = basic_container_args
        container = DependencyContainer(config, template, out_dir, audio_mode=AudioMode.MIX)
        assert isinstance(container.audio_source, MixedAudioSource)

    def test_default_is_mic_only(self, basic_container_args):
        config, template, out_dir = basic_container_args
        container = DependencyContainer(config, template, out_dir)
        assert isinstance(container.audio_source, SounddeviceAudioSource)

    def test_system_only_wires_soundcard_loopback_on_linux(self, monkeypatch, basic_container_args):
        import lazy_take_notes.l4_frameworks_and_drivers.container as container_mod

        monkeypatch.setattr(container_mod.sys, 'platform', 'linux')
        config, template, out_dir = basic_container_args
        container = DependencyContainer(config, template, out_dir, audio_mode=AudioMode.SYSTEM_ONLY)

        from lazy_take_notes.l3_interface_adapters.gateways.soundcard_loopback_source import SoundCardLoopbackSource

        assert isinstance(container.audio_source, SoundCardLoopbackSource)

    def test_mix_wires_mixed_with_soundcard_on_linux(self, monkeypatch, basic_container_args):
        import lazy_take_notes.l4_frameworks_and_drivers.container as container_mod

        monkeypatch.setattr(container_mod.sys, 'platform', 'linux')
        config, template, out_dir = basic_container_args
        container = DependencyContainer(config, template, out_dir, audio_mode=AudioMode.MIX)
        assert isinstance(container.audio_source, MixedAudioSource)
