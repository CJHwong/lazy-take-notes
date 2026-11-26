# /// script
# requires-python = ">=3.11"
# ///
"""Fail when a window view uses a size, color or weight that is not a token.

Every space, size, radius, type size, tint and weight in `src/gui/` comes
from `src/gui/design.rs`, so all screens keep one rhythm. This check finds a
literal anywhere else in `src/gui/`: a `px(...)`, a number in `.opacity(...)`,
a color literal, the kit's rem helpers (`.p_2()`, `.gap_3()`, `.text_sm()`,
`.rounded_lg()`), or a `FontWeight::` name. `_0()` helpers (`.min_w_0()`) are
flex resets, not sizes, so they pass.
"""

import re
import sys
from pathlib import Path

GUI = Path(__file__).resolve().parent.parent / 'src' / 'gui'
TOKENS = GUI / 'design.rs'
SIDES = (
    r'(?:p|px|py|pt|pb|pl|pr|m|mx|my|mt|mb|ml|mr|gap|gap_x|gap_y'
    r'|w|h|size|min_w|min_h|max_w|max_h|top|left|right|bottom)'
)
RULES = [
    (re.compile(r'(?<![\w.])px\('), 'a px() literal: use design::space, size, radius or text'),
    (re.compile(r'\.opacity\(\s*[0-9]'), 'a number in .opacity(): use design::tint'),
    (re.compile(r'(?<![\w.])(?:rgb|rgba|hsla)\(\s*(?:0x|[0-9])'), 'a color literal: use the theme or design.rs'),
    (re.compile(rf'\.{SIDES}_(?:0p5|[1-9][0-9]*(?:p5)?)\(\)'), 'a kit rem helper: use a design token'),
    (re.compile(r'\.text_(?:xs|sm|base|lg|xl|2xl|3xl)\(\)'), 'a kit text size: use design::text'),
    (re.compile(r'\.rounded_(?:xs|sm|md|lg|xl|2xl|3xl)\(\)'), 'a kit radius: use design::radius'),
    (re.compile(r'FontWeight::'), 'a font weight: use design::text::STRONG or BRAND'),
]
# Line comments, so a doc example never counts.
COMMENT = re.compile(r'//.*')


def violations(path: Path) -> list[str]:
    found = []
    for number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), start=1):
        code = COMMENT.sub('', line)
        for pattern, why in RULES:
            if pattern.search(code):
                found.append(f'{path.relative_to(GUI.parent.parent)}:{number}: {why}: {line.strip()}')
    return found


def main() -> int:
    found = [line for path in sorted(GUI.rglob('*.rs')) if path != TOKENS for line in violations(path)]
    for line in found:
        print(line)
    if found:
        hint = 'Move each value into src/gui/design.rs and use the token.'
        print(f'<design-check failed="{len(found)}">{hint}</design-check>')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
