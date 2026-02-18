"""File-based debug logging setup."""

from __future__ import annotations

import logging
from pathlib import Path


def setup_file_logging(output_dir: Path) -> None:
    """Configure file-based debug logging into the output directory."""
    log_path = output_dir / 'ltn_debug.log'
    handler = logging.FileHandler(log_path, encoding='utf-8')
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    root = logging.getLogger('ltn')
    root.setLevel(logging.DEBUG)
    root.addHandler(handler)
    logging.getLogger('ltn.llm').info('Debug logging started → %s', log_path)
