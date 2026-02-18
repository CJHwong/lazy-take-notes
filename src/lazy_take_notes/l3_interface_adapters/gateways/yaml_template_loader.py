"""Gateway: YAML template loader — implements TemplateLoader port."""

from __future__ import annotations

from importlib import resources
from pathlib import Path

import yaml

from lazy_take_notes.l1_entities.template import SessionTemplate, TemplateMetadata
from lazy_take_notes.l3_interface_adapters.gateways.paths import USER_TEMPLATES_DIR

_TEMPLATES_DIR = resources.files('lazy_take_notes') / 'templates'


def builtin_names() -> set[str]:
    """Discover built-in template names from the templates directory."""
    return {p.name.removesuffix('.yaml') for p in _TEMPLATES_DIR.iterdir() if p.name.endswith('.yaml')}


def user_template_names() -> set[str]:
    """Discover user template names from the user templates directory."""
    if not USER_TEMPLATES_DIR.is_dir():
        return set()
    return {p.name.removesuffix('.yaml') for p in USER_TEMPLATES_DIR.iterdir() if p.name.endswith('.yaml')}


def all_template_names() -> set[str]:
    """Union of built-in and user template names."""
    return builtin_names() | user_template_names()


class YamlTemplateLoader:
    """Loads SessionTemplate from YAML files or built-in resources."""

    def load(self, template_ref: str) -> SessionTemplate:
        # 1. Explicit file path
        path = Path(template_ref)
        if path.exists() and path.is_file():
            return SessionTemplate.model_validate(yaml.safe_load(path.read_text(encoding='utf-8')) or {})
        # 2. User template (overrides built-in of the same name)
        if template_ref in user_template_names():
            return _load_user(template_ref)
        # 3. Built-in template
        if template_ref in builtin_names():
            return _load_builtin(template_ref)
        available = sorted(all_template_names())
        raise FileNotFoundError(f"Template not found: '{template_ref}'. Available templates: {', '.join(available)}")

    def list_templates(self) -> list[TemplateMetadata]:
        loaded: dict[str, TemplateMetadata] = {}
        # Built-ins first, then user overrides on top
        for name in builtin_names():
            loaded[name] = _load_builtin(name).metadata
        for name in user_template_names():
            loaded[name] = _load_user(name).metadata
        return [loaded[k] for k in sorted(loaded)]


def _load_builtin(name: str) -> SessionTemplate:
    template_file = _TEMPLATES_DIR / f'{name}.yaml'
    return SessionTemplate.model_validate(yaml.safe_load(template_file.read_text(encoding='utf-8')) or {})


def _load_user(name: str) -> SessionTemplate:
    template_file = USER_TEMPLATES_DIR / f'{name}.yaml'
    return SessionTemplate.model_validate(yaml.safe_load(template_file.read_text(encoding='utf-8')) or {})
