"""Tests for YAML template loader gateway."""

from __future__ import annotations

from pathlib import Path

import pytest

from lazy_take_notes.l1_entities.template import SessionTemplate
from lazy_take_notes.l3_interface_adapters.gateways.yaml_template_loader import (
    YamlTemplateLoader,
    all_template_names,
    builtin_names,
    user_template_names,
)


class TestLoadBuiltinTemplate:
    def test_load_default_zh_tw(self, default_template: SessionTemplate):
        assert default_template.metadata.name == '預設'
        assert default_template.metadata.locale == 'zh-TW'
        assert '會議智慧助手' in default_template.system_prompt
        assert len(default_template.quick_actions) >= 1

    def test_quick_actions_have_required_fields(self, default_template: SessionTemplate):
        for qa in default_template.quick_actions:
            assert qa.label
            assert qa.prompt_template

    def test_digest_templates_have_placeholders(self, default_template: SessionTemplate):
        assert '{line_count}' in default_template.digest_user_template
        assert '{new_lines}' in default_template.digest_user_template
        assert '{full_transcript}' in default_template.final_user_template


class TestLoadCustomTemplate:
    def test_load_from_file(self, tmp_path: Path):
        custom = tmp_path / 'custom.yaml'
        custom.write_text(
            """\
metadata:
  name: "custom_en"
  locale: "en-US"
system_prompt: "You are a meeting assistant."
digest_user_template: "New transcript ({line_count} lines):\\n{new_lines}"
final_user_template: "Meeting ended.\\n{new_lines}\\n{full_transcript}"
quick_actions:
  - label: "Catch up"
    prompt_template: "Summarize: {digest_markdown}"
""",
            encoding='utf-8',
        )
        loader = YamlTemplateLoader()
        tmpl = loader.load(str(custom))
        assert tmpl.metadata.name == 'custom_en'
        assert tmpl.metadata.locale == 'en-US'
        assert len(tmpl.quick_actions) == 1

    def test_load_by_display_name(self):
        loader = YamlTemplateLoader()
        tmpl = loader.load('預設')
        assert tmpl.metadata.name == '預設'
        assert tmpl.metadata.locale == 'zh-TW'

    def test_load_nonexistent_raises(self):
        loader = YamlTemplateLoader()
        with pytest.raises(FileNotFoundError, match='Template not found'):
            loader.load('nonexistent_template_name')


class TestAllBuiltinTemplates:
    def test_metadata_is_populated(self, any_builtin_template: SessionTemplate):
        meta = any_builtin_template.metadata
        assert meta.name
        assert meta.description
        assert meta.locale

    def test_system_prompt_is_substantial(self, any_builtin_template: SessionTemplate):
        assert len(any_builtin_template.system_prompt) >= 200

    def test_digest_template_placeholders(self, any_builtin_template: SessionTemplate):
        tmpl = any_builtin_template
        assert '{line_count}' in tmpl.digest_user_template
        assert '{new_lines}' in tmpl.digest_user_template
        assert '{user_context}' in tmpl.digest_user_template

    def test_final_template_placeholders(self, any_builtin_template: SessionTemplate):
        tmpl = any_builtin_template
        assert '{full_transcript}' in tmpl.final_user_template
        assert '{new_lines}' in tmpl.final_user_template

    def test_quick_actions_count_within_limit(self, any_builtin_template: SessionTemplate):
        assert len(any_builtin_template.quick_actions) <= 5

    def test_quick_actions_have_required_fields(self, any_builtin_template: SessionTemplate):
        for qa in any_builtin_template.quick_actions:
            assert qa.label
            assert qa.prompt_template

    def test_has_at_least_one_quick_action(self, any_builtin_template: SessionTemplate):
        assert len(any_builtin_template.quick_actions) >= 1


class TestListTemplates:
    def test_returns_all_builtins(self, monkeypatch):
        import lazy_take_notes.l3_interface_adapters.gateways.yaml_template_loader as mod

        monkeypatch.setattr(mod, 'USER_TEMPLATES_DIR', Path('/nonexistent/user/templates'))
        loader = YamlTemplateLoader()
        result = loader.list_templates()
        keys = {t.key for t in result}
        assert keys == builtin_names()

    def test_sorted_by_name(self, monkeypatch):
        import lazy_take_notes.l3_interface_adapters.gateways.yaml_template_loader as mod

        monkeypatch.setattr(mod, 'USER_TEMPLATES_DIR', Path('/nonexistent/user/templates'))
        loader = YamlTemplateLoader()
        result = loader.list_templates()
        keys = [t.key for t in result]
        assert keys == sorted(keys)

    def test_each_has_description(self, monkeypatch):
        import lazy_take_notes.l3_interface_adapters.gateways.yaml_template_loader as mod

        monkeypatch.setattr(mod, 'USER_TEMPLATES_DIR', Path('/nonexistent/user/templates'))
        loader = YamlTemplateLoader()
        for t in loader.list_templates():
            assert t.description


_USER_TEMPLATE_YAML = """\
metadata:
  name: "my_custom"
  description: "A user-defined template"
  locale: "en-US"
system_prompt: "You are an assistant."
digest_user_template: "Lines: {line_count}\\n{new_lines}"
final_user_template: "Done.\\n{new_lines}\\n{full_transcript}"
quick_actions: []
"""


class TestUserTemplates:
    def test_user_template_names_empty_when_no_dir(self, monkeypatch):
        import lazy_take_notes.l3_interface_adapters.gateways.yaml_template_loader as mod

        monkeypatch.setattr(mod, 'USER_TEMPLATES_DIR', Path('/nonexistent/user/templates'))
        assert user_template_names() == set()

    def test_user_template_names_discovers_yaml_files(self, tmp_path: Path, monkeypatch):
        import lazy_take_notes.l3_interface_adapters.gateways.yaml_template_loader as mod

        monkeypatch.setattr(mod, 'USER_TEMPLATES_DIR', tmp_path)
        (tmp_path / 'my_custom.yaml').write_text(_USER_TEMPLATE_YAML, encoding='utf-8')
        (tmp_path / 'not_a_template.txt').write_text('ignore me', encoding='utf-8')
        names = user_template_names()
        assert names == {'my_custom'}

    def test_load_user_template_by_name(self, tmp_path: Path, monkeypatch):
        import lazy_take_notes.l3_interface_adapters.gateways.yaml_template_loader as mod

        monkeypatch.setattr(mod, 'USER_TEMPLATES_DIR', tmp_path)
        (tmp_path / 'my_custom.yaml').write_text(_USER_TEMPLATE_YAML, encoding='utf-8')
        loader = YamlTemplateLoader()
        tmpl = loader.load('my_custom')
        assert tmpl.metadata.name == 'my_custom'
        assert tmpl.metadata.locale == 'en-US'

    def test_user_overrides_builtin(self, tmp_path: Path, monkeypatch):
        import lazy_take_notes.l3_interface_adapters.gateways.yaml_template_loader as mod

        monkeypatch.setattr(mod, 'USER_TEMPLATES_DIR', tmp_path)
        override_yaml = _USER_TEMPLATE_YAML.replace('my_custom', 'default_en').replace('en-US', 'en-OVERRIDE')
        (tmp_path / 'default_en.yaml').write_text(override_yaml, encoding='utf-8')
        loader = YamlTemplateLoader()
        tmpl = loader.load('default_en')
        assert tmpl.metadata.locale == 'en-OVERRIDE'

    def test_list_templates_includes_user(self, tmp_path: Path, monkeypatch):
        import lazy_take_notes.l3_interface_adapters.gateways.yaml_template_loader as mod

        monkeypatch.setattr(mod, 'USER_TEMPLATES_DIR', tmp_path)
        (tmp_path / 'my_custom.yaml').write_text(_USER_TEMPLATE_YAML, encoding='utf-8')
        loader = YamlTemplateLoader()
        keys = {t.key for t in loader.list_templates()}
        assert 'my_custom' in keys
        assert builtin_names().issubset(keys)

    def test_all_template_names_is_union(self, tmp_path: Path, monkeypatch):
        import lazy_take_notes.l3_interface_adapters.gateways.yaml_template_loader as mod

        monkeypatch.setattr(mod, 'USER_TEMPLATES_DIR', tmp_path)
        (tmp_path / 'my_custom.yaml').write_text(_USER_TEMPLATE_YAML, encoding='utf-8')
        result = all_template_names()
        assert result == builtin_names() | {'my_custom'}
