"""Dump Textual's built-in themes as the color variables Textual generates from them.

config.yaml stores a Textual theme name, so the Rust TUI needs the same colors.
Values stay as Textual writes them: '#RRGGBB', '#RRGGBBAA', or 'auto NN%'
(white or black by the background's brightness, at that alpha).

Run from the repo root: uv run python rust/scripts/gen_themes.py
"""

from __future__ import annotations

import json
from pathlib import Path

from textual.theme import BUILTIN_THEMES

OUT = Path(__file__).resolve().parent.parent / 'src' / 'app' / 'themes.json'


def main() -> None:
    themes = {}
    for name, theme in BUILTIN_THEMES.items():
        variables = theme.to_color_system().generate()
        themes[name] = {
            'dark': theme.dark,
            'variables': {key: str(value) for key, value in sorted(variables.items()) if 'ansi' not in key},
        }
    OUT.write_text(json.dumps(themes, indent=1, sort_keys=True) + '\n', encoding='utf-8')
    print(f'{len(themes)} themes -> {OUT}')


if __name__ == '__main__':
    main()
