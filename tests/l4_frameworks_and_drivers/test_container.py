"""Tests for the dependency container."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from lazy_take_notes.l3_interface_adapters.gateways.yaml_template_loader import YamlTemplateLoader
from lazy_take_notes.l4_frameworks_and_drivers.config import build_app_config
from lazy_take_notes.l4_frameworks_and_drivers.container import DependencyContainer


class TestDependencyContainer:
    @patch('lazy_take_notes.l4_frameworks_and_drivers.container.SubprocessWhisperTranscriber')
    @patch('lazy_take_notes.l4_frameworks_and_drivers.container.SounddeviceAudioSource')
    def test_creates_all_components(self, mock_audio, mock_whisper, tmp_path: Path):
        config = build_app_config({})
        template = YamlTemplateLoader().load('default_zh_tw')
        output_dir = tmp_path / 'output'

        container = DependencyContainer(config, template, output_dir)

        assert container.config is config
        assert container.template is template
        assert container.persistence is not None
        assert container.llm_client is not None
        assert container.controller is not None

    def test_config_loader_factory(self):
        loader = DependencyContainer.config_loader()
        assert hasattr(loader, 'load')

    def test_template_loader_factory(self):
        loader = DependencyContainer.template_loader()
        assert hasattr(loader, 'load')
        assert hasattr(loader, 'list_templates')
