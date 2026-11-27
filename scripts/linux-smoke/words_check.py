"""Check that a transcript holds at least half of the words in a script.

    python3 words_check.py expected.txt transcript.txt

Both files are lowercased, punctuation is stripped, and comment lines (#) in
the expected file are skipped. Extra words in the transcript do not count.
"""

from __future__ import annotations

import string
import sys
from pathlib import Path

PUNCTUATION = str.maketrans('', '', string.punctuation)


def words(text: str) -> list[str]:
    return text.lower().translate(PUNCTUATION).split()


def expected_words(path: Path) -> set[str]:
    lines = [line for line in path.read_text(encoding='utf-8').splitlines() if not line.startswith('#')]
    return set(words('\n'.join(lines)))


def main() -> int:
    if len(sys.argv) != 3:
        print('usage: words_check.py EXPECTED TRANSCRIPT', file=sys.stderr)
        return 2
    expected = expected_words(Path(sys.argv[1]))
    heard = set(words(Path(sys.argv[2]).read_text(encoding='utf-8')))
    if not expected:
        print('FAIL: the expected file has no words', file=sys.stderr)
        return 1
    found = len(expected & heard)
    print(f'{found} of {len(expected)} expected words found')
    return 0 if found * 2 >= len(expected) else 1


if __name__ == '__main__':
    raise SystemExit(main())
