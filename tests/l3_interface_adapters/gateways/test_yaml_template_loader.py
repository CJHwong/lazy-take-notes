"""Tests for YAML template loader gateway."""

from __future__ import annotations

from pathlib import Path

import pytest

from lazy_take_notes.l1_entities.template import SessionTemplate
from lazy_take_notes.l3_interface_adapters.gateways.yaml_template_loader import (
    YamlTemplateLoader,
    all_template_names,
    builtin_names,
    delete_user_template,
    ensure_user_copy,
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


class TestEnsureUserCopy:
    def test_copies_builtin_to_user_dir(self, tmp_path: Path, monkeypatch):
        import lazy_take_notes.l3_interface_adapters.gateways.yaml_template_loader as mod

        user_dir = tmp_path / 'templates'
        monkeypatch.setattr(mod, 'USER_TEMPLATES_DIR', user_dir)
        assert not user_dir.exists()

        path = ensure_user_copy('default_en')

        assert path == user_dir / 'default_en.yaml'
        assert path.exists()
        # Must be valid YAML that loads as a template
        loader = YamlTemplateLoader()
        tmpl = loader.load(str(path))
        assert tmpl.metadata.locale.startswith('en')

    def test_returns_existing_user_path_without_overwriting(self, tmp_path: Path, monkeypatch):
        import lazy_take_notes.l3_interface_adapters.gateways.yaml_template_loader as mod

        monkeypatch.setattr(mod, 'USER_TEMPLATES_DIR', tmp_path)
        user_file = tmp_path / 'my_custom.yaml'
        user_file.write_text(_USER_TEMPLATE_YAML, encoding='utf-8')
        original_content = user_file.read_text(encoding='utf-8')

        path = ensure_user_copy('my_custom')

        assert path == user_file
        assert path.read_text(encoding='utf-8') == original_content

    def test_raises_for_unknown_name(self, tmp_path: Path, monkeypatch):
        import lazy_take_notes.l3_interface_adapters.gateways.yaml_template_loader as mod

        monkeypatch.setattr(mod, 'USER_TEMPLATES_DIR', tmp_path)
        with pytest.raises(FileNotFoundError, match='Template not found'):
            ensure_user_copy('totally_nonexistent_template')

    def test_does_not_overwrite_existing_user_override(self, tmp_path: Path, monkeypatch):
        """If user already has a copy of a built-in name, return it as-is."""
        import lazy_take_notes.l3_interface_adapters.gateways.yaml_template_loader as mod

        monkeypatch.setattr(mod, 'USER_TEMPLATES_DIR', tmp_path)
        # Create a user file with the same name as a built-in but different content
        custom_yaml = _USER_TEMPLATE_YAML.replace('my_custom', 'default_en')
        user_file = tmp_path / 'default_en.yaml'
        user_file.write_text(custom_yaml, encoding='utf-8')

        path = ensure_user_copy('default_en')

        assert path == user_file
        # Content should be the user's version, not the built-in
        assert 'my_custom' not in path.read_text(encoding='utf-8')  # wasn't overwritten with built-in


class TestDeleteUserTemplate:
    def test_deletes_user_template(self, tmp_path: Path, monkeypatch):
        import lazy_take_notes.l3_interface_adapters.gateways.yaml_template_loader as mod

        monkeypatch.setattr(mod, 'USER_TEMPLATES_DIR', tmp_path)
        user_file = tmp_path / 'my_custom.yaml'
        user_file.write_text(_USER_TEMPLATE_YAML, encoding='utf-8')
        assert user_file.exists()

        delete_user_template('my_custom')

        assert not user_file.exists()
        assert 'my_custom' not in user_template_names()

    def test_raises_for_non_user_template(self, tmp_path: Path, monkeypatch):
        import lazy_take_notes.l3_interface_adapters.gateways.yaml_template_loader as mod

        monkeypatch.setattr(mod, 'USER_TEMPLATES_DIR', tmp_path)
        with pytest.raises(ValueError, match='is not a user template'):
            delete_user_template('default_en')

    def test_raises_for_unknown_template(self, tmp_path: Path, monkeypatch):
        import lazy_take_notes.l3_interface_adapters.gateways.yaml_template_loader as mod

        monkeypatch.setattr(mod, 'USER_TEMPLATES_DIR', tmp_path)
        with pytest.raises(ValueError, match='is not a user template'):
            delete_user_template('totally_nonexistent')
