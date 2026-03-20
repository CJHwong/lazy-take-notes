"""Tests for file-based debug logging setup."""

from __future__ import annotations

import logging
from pathlib import Path

from lazy_take_notes.l1_entities.session_files import DEBUG_LOG
from lazy_take_notes.l4_frameworks_and_drivers.logging_setup import setup_file_logging


class TestSetupFileLogging:
    def test_enabled_creates_log_file(self, tmp_path: Path):
        setup_file_logging(tmp_path, enabled=True)
        log_path = tmp_path / DEBUG_LOG.name
        logging.getLogger('ltn.test').info('test message')
        assert log_path.exists()
        assert 'test message' in log_path.read_text(encoding='utf-8')

    def test_disabled_does_not_create_file(self, tmp_path: Path):
        setup_file_logging(tmp_path, enabled=False)
        log_path = tmp_path / DEBUG_LOG.name
        logging.getLogger('ltn.test').info('should not appear')
        assert not log_path.exists()

    def test_disabled_adds_null_handler(self, tmp_path: Path):
        root = logging.getLogger('ltn')
        handler_count_before = len(root.handlers)
        setup_file_logging(tmp_path, enabled=False)
        assert len(root.handlers) == handler_count_before + 1
        assert isinstance(root.handlers[-1], logging.NullHandler)
