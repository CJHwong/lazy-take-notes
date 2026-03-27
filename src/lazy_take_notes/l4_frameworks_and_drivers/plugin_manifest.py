"""Plugin manifest: manage uvx --with specs persisted in plugins.yaml + plugins.txt."""

from __future__ import annotations

import subprocess  # noqa: S404 -- used for uvx validation only, fixed arg list
from pathlib import Path

import yaml

from lazy_take_notes.l3_interface_adapters.gateways.paths import PLUGINS_TXT, PLUGINS_YAML


def parse_spec_name(spec: str) -> str:
    """Extract the package name from a pip/uvx spec.

    Examples:
        "ltn-youtube" -> "ltn-youtube"
        "ltn-youtube @ git+https://..." -> "ltn-youtube"
        "ltn-youtube>=1.0" -> "ltn-youtube"
    """
    spec = spec.strip()
    for sep in (' @', '>=', '<=', '==', '!=', '~=', '<', '>'):
        if sep in spec:
            return spec.split(sep)[0].strip()
    return spec


def load_plugins(config_dir: Path | None = None) -> list[str]:
    """Read plugin specs from plugins.yaml."""
    yaml_path = (config_dir / 'plugins.yaml') if config_dir else PLUGINS_YAML
    try:
        data = yaml.safe_load(yaml_path.read_text(encoding='utf-8')) or {}
    except FileNotFoundError:
        return []
    return list(data.get('plugins', []))


def save_plugins(specs: list[str], config_dir: Path | None = None) -> None:
    """Write plugin specs to plugins.yaml and regenerate plugins.txt.

    plugins.txt is written first since it is the hot-path artifact read by the
    wrapper script and can be trivially regenerated from the YAML.
    """
    if config_dir:
        yaml_path = config_dir / 'plugins.yaml'
        txt_path = config_dir / 'plugins.txt'
    else:
        yaml_path = PLUGINS_YAML
        txt_path = PLUGINS_TXT
    yaml_path.parent.mkdir(parents=True, exist_ok=True)

    txt_path.write_text('\n'.join(specs) + '\n' if specs else '', encoding='utf-8')
    yaml_path.write_text(
        yaml.dump({'plugins': specs}, default_flow_style=False, sort_keys=False, allow_unicode=True),
        encoding='utf-8',
    )


def validate_spec(spec: str) -> tuple[bool, str]:
    """Dry-run validate a plugin spec via uvx --with.

    Returns (ok, error_message).
    """
    try:
        result = subprocess.run(  # noqa: S603 -- fixed arg list, not shell=True
            [  # noqa: S607 -- uvx resolved via PATH, trusted tool
                'uvx',
                '--from',
                'lazy-take-notes @ git+https://github.com/CJHwong/lazy-take-notes.git',
                '--with',
                spec,
                'lazy-take-notes',
                '--version',
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except FileNotFoundError:
        return False, 'uvx not found on PATH'
    except subprocess.TimeoutExpired:
        return False, 'validation timed out (120s)'

    if result.returncode == 0:
        return True, ''
    stderr = result.stderr.strip()
    return False, stderr or f'uvx exited with code {result.returncode}'


def add_plugin(spec: str, config_dir: Path | None = None, skip_validation: bool = False) -> str | None:
    """Add a plugin spec. Returns None on success or an error message."""
    name = parse_spec_name(spec)
    existing = load_plugins(config_dir)

    for existing_spec in existing:
        if parse_spec_name(existing_spec) == name:
            return None  # already installed, idempotent

    if not skip_validation:
        ok, err = validate_spec(spec)
        if not ok:
            return f'Validation failed: {err}'

    existing.append(spec)
    save_plugins(existing, config_dir)
    return None


def remove_plugin(name: str, config_dir: Path | None = None) -> bool:
    """Remove a plugin by package name. Returns True if removed, False if not found."""
    existing = load_plugins(config_dir)
    filtered = [s for s in existing if parse_spec_name(s) != name]
    if len(filtered) == len(existing):
        return False
    save_plugins(filtered, config_dir)
    return True
