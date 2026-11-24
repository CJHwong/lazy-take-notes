"""Dump the key bindings of every Python screen, app and widget for the Rust port.

Key bindings are part of the data contract. Textual merges BINDINGS along the
class MRO, so this records the merged table each class really has.

Run from the repo root: uv run python rust/scripts/gen_bindings.py
"""

from __future__ import annotations

import importlib
import inspect
import json
import pkgutil
from pathlib import Path

from textual.dom import DOMNode

import lazy_take_notes.l4_frameworks_and_drivers as l4

OUT = Path(__file__).resolve().parent.parent / 'tests' / 'fixtures' / 'bindings.json'


def l4_classes():
    for module_info in pkgutil.walk_packages(l4.__path__, l4.__name__ + '.'):
        module = importlib.import_module(module_info.name)
        for name, cls in inspect.getmembers(module, inspect.isclass):
            if cls.__module__ == module.__name__ and issubclass(cls, DOMNode) and 'BINDINGS' in vars(cls):
                yield f'{module.__name__.removeprefix(l4.__name__ + ".")}.{name}', cls


def merged(cls) -> list[dict]:
    rows = []
    for key, bindings in cls._merge_bindings().key_to_bindings.items():
        for binding in bindings:
            rows.append(
                {
                    'key': key,
                    'action': binding.action,
                    'description': binding.description,
                    'show': binding.show,
                    'key_display': binding.key_display,
                    'priority': binding.priority,
                }
            )
    return rows


def main() -> None:
    table = {name: merged(cls) for name, cls in sorted(l4_classes())}
    OUT.write_text(json.dumps(table, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f'{len(table)} classes -> {OUT}')


if __name__ == '__main__':
    main()
