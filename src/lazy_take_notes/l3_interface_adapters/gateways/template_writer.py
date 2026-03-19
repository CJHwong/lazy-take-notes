"""Gateway: save a SessionTemplate to the user templates directory as YAML."""

from __future__ import annotations

from pathlib import Path

import yaml

from lazy_take_notes.l1_entities.template import SessionTemplate
from lazy_take_notes.l3_interface_adapters.gateways.paths import USER_TEMPLATES_DIR


def save_user_template(template: SessionTemplate, name: str) -> Path:
    """Serialize *template* to YAML and write to the user templates directory.

    Returns the path of the written file.
    """
    USER_TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
    data = template.model_dump()
    # key is set by the loader at load time — don't persist it
    data.get('metadata', {}).pop('key', None)
    dest = USER_TEMPLATES_DIR / f'{name}.yaml'
    dest.write_text(
        yaml.dump(data, default_flow_style=False, allow_unicode=True, sort_keys=False),
        encoding='utf-8',
    )
    return dest
