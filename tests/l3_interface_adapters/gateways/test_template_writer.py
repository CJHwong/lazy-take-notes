"""Tests for template_writer — save + roundtrip via YamlTemplateLoader."""

from __future__ import annotations

from pathlib import Path

from lazy_take_notes.l1_entities.template import QuickAction, SessionTemplate, TemplateMetadata
from lazy_take_notes.l3_interface_adapters.gateways.template_writer import save_user_template
from lazy_take_notes.l3_interface_adapters.gateways.yaml_template_loader import YamlTemplateLoader


def _make_template() -> SessionTemplate:
    return SessionTemplate(
        metadata=TemplateMetadata(name='My Custom', description='Custom template', locale='en'),
        system_prompt='You are a helpful assistant.',
        digest_user_template='Lines ({line_count}):\n{new_lines}\n{user_context}',
        final_user_template='Final ({line_count}):\n{new_lines}\n{user_context}\n{full_transcript}',
        quick_actions=[
            QuickAction(
                label='Summary',
                description='Quick summary',
                prompt_template='Digest:\n{digest_markdown}\n{recent_transcript}',
            ),
        ],
    )


class TestSaveUserTemplate:
    def test_saves_yaml_file(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(
            'lazy_take_notes.l3_interface_adapters.gateways.template_writer.USER_TEMPLATES_DIR',
            tmp_path,
        )
        template = _make_template()
        path = save_user_template(template, 'my_custom')

        assert path == tmp_path / 'my_custom.yaml'
        assert path.exists()

    def test_roundtrip_via_loader(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(
            'lazy_take_notes.l3_interface_adapters.gateways.template_writer.USER_TEMPLATES_DIR',
            tmp_path,
        )
        template = _make_template()
        save_user_template(template, 'roundtrip_test')

        # Load it back via YamlTemplateLoader (by file path)
        loader = YamlTemplateLoader()
        loaded = loader.load(str(tmp_path / 'roundtrip_test.yaml'))

        assert loaded.metadata.name == 'My Custom'
        assert loaded.metadata.locale == 'en'
        assert loaded.system_prompt == 'You are a helpful assistant.'
        assert '{line_count}' in loaded.digest_user_template
        assert '{full_transcript}' in loaded.final_user_template
        assert len(loaded.quick_actions) == 1
        assert loaded.quick_actions[0].label == 'Summary'

    def test_creates_directory_if_missing(self, tmp_path: Path, monkeypatch):
        nested = tmp_path / 'deep' / 'nested'
        monkeypatch.setattr(
            'lazy_take_notes.l3_interface_adapters.gateways.template_writer.USER_TEMPLATES_DIR',
            nested,
        )
        template = _make_template()
        path = save_user_template(template, 'nested_test')

        assert path.exists()
        assert nested.is_dir()

    def test_key_not_persisted(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(
            'lazy_take_notes.l3_interface_adapters.gateways.template_writer.USER_TEMPLATES_DIR',
            tmp_path,
        )
        template = _make_template()
        template.metadata.key = 'should_not_appear'
        save_user_template(template, 'no_key_test')

        content = (tmp_path / 'no_key_test.yaml').read_text(encoding='utf-8')
        assert 'should_not_appear' not in content

    def test_overwrites_existing(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(
            'lazy_take_notes.l3_interface_adapters.gateways.template_writer.USER_TEMPLATES_DIR',
            tmp_path,
        )
        template = _make_template()
        save_user_template(template, 'overwrite_test')

        template.metadata.name = 'Updated Name'
        save_user_template(template, 'overwrite_test')

        loader = YamlTemplateLoader()
        loaded = loader.load(str(tmp_path / 'overwrite_test.yaml'))
        assert loaded.metadata.name == 'Updated Name'
