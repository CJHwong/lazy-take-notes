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


def ensure_user_copy(name: str) -> Path:
    """Return the user-templates path for *name*, copying the built-in if needed.

    - Already a user template → return its path (no overwrite).
    - Built-in only → copy YAML to user templates dir, return new path.
    - Unknown → raise FileNotFoundError.
    """
    if name in user_template_names():
        return USER_TEMPLATES_DIR / f'{name}.yaml'
    if name not in builtin_names():
        raise FileNotFoundError(f"Template not found: '{name}'")
    USER_TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
    source = _TEMPLATES_DIR / f'{name}.yaml'
    dest = USER_TEMPLATES_DIR / f'{name}.yaml'
    dest.write_text(source.read_text(encoding='utf-8'), encoding='utf-8')
    return dest


def delete_user_template(name: str) -> None:
    """Delete a user template YAML file.

    Raises ValueError if *name* is not a user template.
    """
    if name not in user_template_names():
        raise ValueError(f"'{name}' is not a user template")
    (USER_TEMPLATES_DIR / f'{name}.yaml').unlink()


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
        # 4. Match by display name (metadata.name) across all templates
        user_keys = user_template_names()
        for key in all_template_names():
            tmpl = _load_user(key) if key in user_keys else _load_builtin(key)
            if tmpl.metadata.name == template_ref:
                return tmpl
        available_keys = sorted(all_template_names())
        raise FileNotFoundError(
            f"Template not found: '{template_ref}'. Available templates: {', '.join(available_keys)}"
        )

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
    tmpl = SessionTemplate.model_validate(yaml.safe_load(template_file.read_text(encoding='utf-8')) or {})
    tmpl.metadata.key = name
    return tmpl


def _load_user(name: str) -> SessionTemplate:
    template_file = USER_TEMPLATES_DIR / f'{name}.yaml'
    tmpl = SessionTemplate.model_validate(yaml.safe_load(template_file.read_text(encoding='utf-8')) or {})
    tmpl.metadata.key = name
    return tmpl
