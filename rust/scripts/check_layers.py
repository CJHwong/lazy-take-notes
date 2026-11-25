# /// script
# requires-python = ">=3.11"
# ///
"""Fail when a Rust module imports a layer above its own.

Layers, lowest first. Each one may import only the layers before it:
domain, ports, usecases, adapters (with config.rs), app, tui, cli.
ratatui and crossterm belong to tui and cli only. tests/ is out of scope.
"""

import re
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / 'src'
ORDER = ['domain', 'ports', 'usecases', 'adapters', 'app', 'tui', 'cli']
# Top-level modules that sit in a layer under another name.
ALIASES = {'config': 'adapters'}
CRATE_PATH = re.compile(r'\bcrate::([a-z_]+)')
TERMINAL_USE = re.compile(r'\b(ratatui|crossterm)\b')
CRATE_GROUP = re.compile(r'\bcrate::\{')
# Line comments, so a doc example never counts. Keeps the newline for line numbers.
COMMENT = re.compile(r'//[^\n]*')


def layer_of(module: str) -> str | None:
    module = ALIASES.get(module, module)
    return module if module in ORDER else None


def group_modules(text: str, start: int) -> list[str]:
    """The first module of each path in the `{...}` group that opens at `start`."""
    depth, item, modules = 0, '', []
    for char in text[start:]:
        if char == '{':
            depth += 1
            if depth == 1:
                continue
        elif char == '}':
            depth -= 1
            if depth == 0:
                break
        if depth == 1 and char == ',':
            modules.append(item)
            item = ''
            continue
        item += char
    modules.append(item)
    return [m.strip().split('::')[0] for m in modules if m.strip()]


def imported_modules(text: str) -> list[tuple[int, str]]:
    """(line, module) for each `crate::module` path, groups included."""
    found = []
    for match in CRATE_PATH.finditer(text):
        found.append((text.count('\n', 0, match.start()) + 1, match.group(1)))
    for match in CRATE_GROUP.finditer(text):
        line = text.count('\n', 0, match.start()) + 1
        found.extend((line, module) for module in group_modules(text, match.end() - 1))
    return found


def violations(path: Path) -> list[str]:
    relative = path.relative_to(SRC)
    own = layer_of(relative.parts[0].removesuffix('.rs'))
    if own is None:
        return []
    rank = ORDER.index(own)
    text = COMMENT.sub('', path.read_text(encoding='utf-8'))
    found = []
    for number, module in imported_modules(text):
        target = layer_of(module)
        if target is not None and ORDER.index(target) > rank:
            found.append(f'{relative}:{number}: {own} imports {target}')
    if rank < ORDER.index('tui'):
        for match in TERMINAL_USE.finditer(text):
            number = text.count('\n', 0, match.start()) + 1
            found.append(f'{relative}:{number}: {own} uses a terminal crate')
    return found


def main() -> int:
    found = [v for path in sorted(SRC.rglob('*.rs')) for v in violations(path)]
    for violation in found:
        print(violation)
    if found:
        print(f'{len(found)} layer violations. Order, lowest first: {", ".join(ORDER)}.')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
