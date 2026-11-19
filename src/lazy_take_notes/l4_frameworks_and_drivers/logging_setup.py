"""File-based debug logging setup."""

from __future__ import annotations

import faulthandler
import logging
import sys
from pathlib import Path

from lazy_take_notes.l1_entities.session_files import DEBUG_LOG

_SESSION_HANDLER_FLAG = '_ltn_session_handler'


def _detach_previous_session(root: logging.Logger) -> None:
    """The kiosk runs many sessions in one process; each must log only to its own file."""
    for handler in list(root.handlers):
        if getattr(handler, _SESSION_HANDLER_FLAG, False):
            if isinstance(handler, logging.FileHandler) and faulthandler.is_enabled():
                faulthandler.disable()  # it holds this handler's file; never write to a closed one
            root.removeHandler(handler)
            handler.close()


def close_file_logging() -> None:
    """Close the session's log file, so the session folder can be removed."""
    _detach_previous_session(logging.getLogger('ltn'))


def see_debug_log() -> str:
    """' (see debug.log)' while this session writes one, else ''.

    With save_debug_log off there is no file, and a hint must not point at one.
    """
    writing = any(
        isinstance(handler, logging.FileHandler) and getattr(handler, _SESSION_HANDLER_FLAG, False)
        for handler in logging.getLogger('ltn').handlers
    )
    return f' (see {DEBUG_LOG.name})' if writing else ''


def _attach(root: logging.Logger, handler: logging.Handler) -> None:
    setattr(handler, _SESSION_HANDLER_FLAG, True)
    root.addHandler(handler)


def setup_file_logging(output_dir: Path, *, enabled: bool = True) -> None:
    """Configure file-based debug logging into the output directory.

    When *enabled* is False, only attach a NullHandler so library loggers
    stay silent without FileHandler side-effects. A log file that cannot be
    opened also falls back to the NullHandler: losing the log must not stop
    the recording.
    """
    root = logging.getLogger('ltn')
    _detach_previous_session(root)
    if not enabled:
        _attach(root, logging.NullHandler())
        return
    log_path = output_dir / DEBUG_LOG.name
    try:
        handler = logging.FileHandler(log_path, encoding='utf-8')
    except OSError:
        _attach(root, logging.NullHandler())
        return
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    root.setLevel(logging.DEBUG)
    _attach(root, handler)
    # A native crash (segfault in an audio or numeric library) kills the
    # process before any Python handler runs. faulthandler writes the stack of
    # every thread straight into the same file.
    faulthandler.enable(file=handler.stream or sys.stderr, all_threads=True)  # stream is open: delay=False
    logging.getLogger('ltn.llm').info('Debug logging started → %s', log_path)
