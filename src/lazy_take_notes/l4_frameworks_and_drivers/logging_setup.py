"""File-based debug logging setup."""

from __future__ import annotations

import logging
from pathlib import Path

from lazy_take_notes.l1_entities.session_files import DEBUG_LOG


def setup_file_logging(output_dir: Path, *, enabled: bool = True) -> None:
    """Configure file-based debug logging into the output directory.

    When *enabled* is False, only attach a NullHandler so library loggers
    stay silent without FileHandler side-effects.
    """
    root = logging.getLogger('ltn')
    if not enabled:
        root.addHandler(logging.NullHandler())
        return
    log_path = output_dir / DEBUG_LOG.name
    handler = logging.FileHandler(log_path, encoding='utf-8')
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    root.setLevel(logging.DEBUG)
    root.addHandler(handler)
    logging.getLogger('ltn.llm').info('Debug logging started → %s', log_path)
