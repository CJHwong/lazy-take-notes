"""Tests for DependencyContainer audio source wiring — always MixedAudioSource."""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest

from lazy_take_notes.l3_interface_adapters.gateways.mixed_audio_source import MixedAudioSource
from lazy_take_notes.l3_interface_adapters.gateways.yaml_template_loader import YamlTemplateLoader
from lazy_take_notes.l4_frameworks_and_drivers.config import build_app_config
from lazy_take_notes.l4_frameworks_and_drivers.container import DependencyContainer


@pytest.fixture
def basic_container_args(tmp_path):
    config = build_app_config({})
    template = YamlTemplateLoader().load('default_en')
    return config, template, tmp_path / 'out'


class TestContainerAudioSource:
    @patch('lazy_take_notes.l4_frameworks_and_drivers.container.DependencyContainer._build_mixed_source')
    def test_always_builds_mixed_source(self, mock_build, basic_container_args):
        mock_build.return_value = MixedAudioSource.__new__(MixedAudioSource)
        config, template, out_dir = basic_container_args
        container = DependencyContainer(config, template, out_dir)
        mock_build.assert_called_once()
        assert container.audio_source is mock_build.return_value

    @pytest.mark.skipif(sys.platform != 'darwin', reason='CoreAudioTapSource is macOS only')
    def test_mixed_source_uses_coreaudio_on_macos(self, basic_container_args):
        from lazy_take_notes.l3_interface_adapters.gateways.coreaudio_tap_source import CoreAudioTapSource

        config, template, out_dir = basic_container_args
        container = DependencyContainer(config, template, out_dir)
        assert isinstance(container.audio_source, MixedAudioSource)
        assert isinstance(container.audio_source._system, CoreAudioTapSource)

    def test_mixed_source_uses_soundcard_on_linux(self, monkeypatch, basic_container_args):
        import lazy_take_notes.l4_frameworks_and_drivers.container as container_mod

        monkeypatch.setattr(container_mod.sys, 'platform', 'linux')
        config, template, out_dir = basic_container_args
        container = DependencyContainer(config, template, out_dir)
        assert isinstance(container.audio_source, MixedAudioSource)

        from lazy_take_notes.l3_interface_adapters.gateways.soundcard_loopback_source import SoundCardLoopbackSource

        assert isinstance(container.audio_source._system, SoundCardLoopbackSource)

    def test_no_audio_source_when_build_audio_false(self, basic_container_args):
        config, template, out_dir = basic_container_args
        container = DependencyContainer(config, template, out_dir, build_audio=False)
        assert container.audio_source is None
