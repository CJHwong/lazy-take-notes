"""Tests for DependencyContainer audio mode wiring."""

from __future__ import annotations

import pytest

from lazy_take_notes.l1_entities.audio_mode import AudioMode
from lazy_take_notes.l3_interface_adapters.gateways.coreaudio_tap_source import CoreAudioTapSource
from lazy_take_notes.l3_interface_adapters.gateways.mixed_audio_source import MixedAudioSource
from lazy_take_notes.l3_interface_adapters.gateways.sounddevice_audio_source import SounddeviceAudioSource
from lazy_take_notes.l3_interface_adapters.gateways.yaml_template_loader import YamlTemplateLoader
from lazy_take_notes.l4_frameworks_and_drivers.container import DependencyContainer
from lazy_take_notes.l4_frameworks_and_drivers.infra_config import build_app_config


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

    def test_system_only_wires_coreaudio_tap_source(self, basic_container_args):
        config, template, out_dir = basic_container_args
        container = DependencyContainer(config, template, out_dir, audio_mode=AudioMode.SYSTEM_ONLY)
        assert isinstance(container.audio_source, CoreAudioTapSource)

    def test_mix_wires_mixed_audio_source(self, basic_container_args):
        config, template, out_dir = basic_container_args
        container = DependencyContainer(config, template, out_dir, audio_mode=AudioMode.MIX)
        assert isinstance(container.audio_source, MixedAudioSource)

    def test_default_is_mic_only(self, basic_container_args):
        config, template, out_dir = basic_container_args
        container = DependencyContainer(config, template, out_dir)
        assert isinstance(container.audio_source, SounddeviceAudioSource)
