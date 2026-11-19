"""Check that the Rust and Python versions read and write the same files.

Works on copies of the real config, user templates and sessions. Prints only
pass or fail and the differing key paths, never values: the config can hold
an API key.

Run from the repo root: uv run python rust/scripts/cross_check.py
"""

from __future__ import annotations

import json
import shutil
import subprocess  # noqa: S404 -- fixed cargo command, no shell
import tempfile
from pathlib import Path

import yaml

from lazy_take_notes.l1_entities.template import SessionTemplate
from lazy_take_notes.l3_interface_adapters.gateways.paths import DEFAULT_CONFIG_PATHS, USER_TEMPLATES_DIR
from lazy_take_notes.l3_interface_adapters.gateways.session_reader import list_sessions
from lazy_take_notes.l4_frameworks_and_drivers.config import InfraConfig, build_app_config

RUST_DIR = Path(__file__).resolve().parent.parent
failures: list[str] = []


def rust(*args: str) -> object:
    result = subprocess.run(  # noqa: S603 -- fixed argument list
        ['cargo', 'run', '-q', '--example', 'cross_check', '--', *args],  # noqa: S607 -- cargo from PATH
        cwd=RUST_DIR,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def differences(left: object, right: object, path: str = '') -> list[str]:
    """Key paths where two JSON values differ."""
    if isinstance(left, dict) and isinstance(right, dict):
        out = []
        for key in sorted(set(left) | set(right)):
            out += differences(left.get(key), right.get(key), f'{path}.{key}')
        return out
    if isinstance(left, list) and isinstance(right, list) and len(left) == len(right):
        out = []
        for index, (a, b) in enumerate(zip(left, right, strict=True)):
            out += differences(a, b, f'{path}[{index}]')
        return out
    return [] if left == right else [path or '(root)']


def check(name: str, left: object, right: object) -> None:
    diff = differences(left, right)
    print(f'{"PASS" if not diff else "FAIL"}  {name}' + (f'  differs at: {", ".join(diff)}' if diff else ''))
    if diff:
        failures.append(name)


def python_config(path: Path) -> dict:
    raw = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
    infra = InfraConfig.model_validate(raw).model_dump(exclude={'__pydantic_extra__'})
    infra = {key: infra[key] for key in InfraConfig.model_fields}
    return {'app': build_app_config(raw).model_dump(), 'infra': infra}


def main() -> None:
    work = Path(tempfile.mkdtemp(prefix='ltn-cross-'))
    config = next((p for p in DEFAULT_CONFIG_PATHS if p.exists()), None)
    if config is None:
        print('SKIP  config: no config file')
    else:
        copy = work / 'config.yaml'
        shutil.copy(config, copy)
        check('config read by Rust', rust('config', str(copy)), python_config(copy))
        rust('rewrite-config', str(copy))
        original = yaml.safe_load(config.read_text(encoding='utf-8')) or {}
        check('config written by Rust, read by Python', yaml.safe_load(copy.read_text(encoding='utf-8')), original)

    for path in sorted(USER_TEMPLATES_DIR.glob('*.yaml')):
        python = SessionTemplate.model_validate(yaml.safe_load(path.read_text(encoding='utf-8')) or {}).model_dump()
        check(f'template {path.name} read by Rust', rust('template', str(path)), python)
        saved = Path(str(rust('save-template', str(path), str(work), path.stem)))
        back = SessionTemplate.model_validate(yaml.safe_load(saved.read_text(encoding='utf-8'))).model_dump()
        check(f'template {path.name} written by Rust, read by Python', back, python)

    output = Path(
        build_app_config(yaml.safe_load(config.read_text(encoding='utf-8')) or {} if config else {}).output.directory
    )
    output = output.expanduser().resolve()
    check(
        f'session list ({len(list_sessions(output))} sessions)',
        rust('sessions', str(output)),
        [s.name for s in list_sessions(output)],
    )

    shutil.rmtree(work)
    raise SystemExit(1 if failures else 0)


if __name__ == '__main__':
    main()
