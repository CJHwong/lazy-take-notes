"""Record what Python's Settings screen reads from config.yaml and writes back.

Each case opens ConfigApp with HOME in a fresh folder, applies form edits by
widget id, and records the form data at open, then the file after Save.
tests/settings_data.rs checks the Rust screen against the result.

Run from the repo root: uv run python rust/scripts/gen_settings_fixtures.py
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess  # noqa: S404 -- runs this script again, one child per case
import sys
import tempfile
from pathlib import Path

from textual.widgets import Button, Input, Select, Switch, TextArea

from lazy_take_notes.l4_frameworks_and_drivers.apps.config import ConfigApp
from lazy_take_notes.l4_frameworks_and_drivers.apps.config.sections import LanguageRow

OUT = Path(__file__).resolve().parent.parent / 'tests' / 'fixtures' / 'settings.json'
# Both versions create this file, so the custom model path is the same in each.
CUSTOM_MODEL = '/tmp/ltn-settings-fixture/model.bin'  # noqa: S108 -- the Rust test uses this fixed path

CASES = [
    {'name': 'no_file', 'config': None, 'edits': []},
    {'name': 'empty_file', 'config': '{}\n', 'edits': []},
    {
        'name': 'remove_zh_override',
        'config': None,
        'edits': [['remove_language', 'zh']],
    },
    {
        'name': 'blank_max_lines',
        'config': 'digest:\n  max_lines: 40\n',
        'edits': [['input', 'cfg-digest-max-lines', '']],
    },
    {'name': 'api_key_kept', 'config': 'openai:\n  api_key: stub-key\n', 'edits': []},
    {
        'name': 'api_key_emptied',
        'config': 'openai:\n  api_key: stub-key\n',
        'edits': [['input', 'cfg-openai-api-key', '']],
    },
    {
        'name': 'api_key_cleared',
        'config': 'openai:\n  api_key: stub-key\n',
        'edits': [['press', 'cfg-openai-clear-key']],
    },
    {
        'name': 'custom_model_path',
        'config': None,
        'edits': [['input', 'cfg-trans-custom-path', CUSTOM_MODEL]],
    },
    {'name': 'theme_nord', 'config': 'theme: nord\nllm_provider: openai\n', 'edits': []},
    {
        'name': 'many_fields',
        'config': 'recognition_hints:\n- Kubernetes\n- GoFreight\nplugin_key:\n  kept: true\n',
        'edits': [
            ['input', 'cfg-digest-min-lines', '20'],
            ['input', 'cfg-trans-chunk', '30'],
            ['switch', 'cfg-output-save-debug-log', True],
            ['switch', 'cfg-show-builtin-templates', False],
            ['select', 'cfg-diarization-device', 'cpu'],
            ['textarea', 'cfg-recognition-hints', 'Kubernetes\n\n  Rust  \n'],
        ],
    },
    {
        'name': 'bad_number',
        'config': None,
        'edits': [['input', 'cfg-trans-silence', '2'], ['input', 'cfg-digest-min-lines', 'x']],
    },
    {'name': 'unreadable_yaml', 'config': 'a: [\n', 'edits': []},
]


def config_file(home: Path) -> Path:
    return home / 'Library' / 'Application Support' / 'lazy-take-notes' / 'config.yaml'


async def run_case(case: dict) -> dict:
    path = config_file(Path.home())
    app = ConfigApp()
    toasts: list[str] = []
    app.notify = lambda message, **_kwargs: toasts.append(str(message))  # type: ignore[method-assign]
    async with app.run_test(size=(120, 50)) as pilot:
        await pilot.pause()
        opened = app._collect_form_data()
        for edit in case['edits']:
            kind = edit[0]
            if kind == 'input':
                app.query_one(f'#{edit[1]}', Input).value = edit[2]
            elif kind == 'switch':
                app.query_one(f'#{edit[1]}', Switch).value = edit[2]
            elif kind == 'select':
                app.query_one(f'#{edit[1]}', Select).value = edit[2]
            elif kind == 'textarea':
                app.query_one(f'#{edit[1]}', TextArea).text = edit[2]
            elif kind == 'press':
                app.query_one(f'#{edit[1]}', Button).press()
            elif kind == 'remove_language':
                row = next(row for row in app.query(LanguageRow) if row.locale == edit[1])
                row.query_one(Button).press()
            await pilot.pause()
        unsaved = app._has_unsaved_changes()
        app.action_save_config()
        await pilot.pause()
    saved = path.read_text(encoding='utf-8') if path.exists() else None
    return {
        'opened': opened,
        'unsaved_before_save': unsaved,
        'saved_file': saved,
        'toasts': toasts,
    }


def child(index: int) -> None:
    case = CASES[index]
    config = case['config']
    if isinstance(config, str):
        path = config_file(Path.home())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(config, encoding='utf-8')
    result = asyncio.run(run_case(case))
    print(json.dumps(result))


def main() -> None:
    Path(CUSTOM_MODEL).parent.mkdir(parents=True, exist_ok=True)
    Path(CUSTOM_MODEL).write_bytes(b'')
    results = []
    for index, case in enumerate(CASES):
        with tempfile.TemporaryDirectory() as home:
            env = {**os.environ, 'HOME': home}
            done = subprocess.run(  # noqa: S603 -- fixed argv: this script and an index
                [sys.executable, __file__, '--case', str(index)],
                env=env,
                capture_output=True,
                text=True,
                check=True,
            )
            result = json.loads(done.stdout.strip().splitlines()[-1])
        results.append({**case, **result})
        print(case['name'], 'saved' if result['saved_file'] else 'not saved', result['toasts'])
    OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--case':
        child(int(sys.argv[2]))
    else:
        main()
