"""Tests for file-based debug logging setup."""

from __future__ import annotations

import logging
from pathlib import Path

from lazy_take_notes.l1_entities.session_files import DEBUG_LOG
from lazy_take_notes.l4_frameworks_and_drivers.logging_setup import see_debug_log, setup_file_logging


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
        setup_file_logging(tmp_path, enabled=False)
        session_handlers = [h for h in root.handlers if getattr(h, '_ltn_session_handler', False)]
        assert len(session_handlers) == 1
        assert isinstance(session_handlers[0], logging.NullHandler)


class TestSeeDebugLog:
    def test_points_at_the_log_while_one_is_written(self, tmp_path: Path):
        setup_file_logging(tmp_path, enabled=True)
        assert see_debug_log() == ' (see debug.log)'

    def test_says_nothing_with_the_log_turned_off(self, tmp_path: Path):
        setup_file_logging(tmp_path, enabled=False)
        assert not see_debug_log()

    def test_says_nothing_when_the_log_could_not_open(self, tmp_path: Path):
        setup_file_logging(tmp_path / 'missing', enabled=True)
        assert not see_debug_log()


class TestRepeatedSessions:
    def test_a_new_session_detaches_the_previous_session_log(self, tmp_path: Path):
        first, second = tmp_path / 'one', tmp_path / 'two'
        first.mkdir()
        second.mkdir()

        setup_file_logging(first, enabled=True)
        setup_file_logging(second, enabled=True)
        logging.getLogger('ltn.test').info('second session only')

        assert 'second session only' not in (first / DEBUG_LOG.name).read_text(encoding='utf-8')
        assert 'second session only' in (second / DEBUG_LOG.name).read_text(encoding='utf-8')

    def test_handlers_do_not_pile_up_across_sessions(self, tmp_path: Path):
        root = logging.getLogger('ltn')
        setup_file_logging(tmp_path, enabled=False)
        count = len(root.handlers)

        for index in range(5):
            session = tmp_path / str(index)
            session.mkdir()
            setup_file_logging(session, enabled=index % 2 == 0)

        assert len(root.handlers) == count

    def test_an_unwritable_session_dir_falls_back_to_no_file_log(self, tmp_path: Path):
        setup_file_logging(tmp_path / 'missing', enabled=True)

        logging.getLogger('ltn.test').info('nowhere to go')


class TestNativeCrashCapture:
    def test_a_native_crash_leaves_a_traceback_in_the_session_log(self, tmp_path: Path):
        import subprocess  # noqa: PLC0415 -- only this test spawns a process
        import sys  # noqa: PLC0415 -- only this test spawns a process

        script = (
            'import faulthandler, pathlib, sys\n'
            'from lazy_take_notes.l4_frameworks_and_drivers.logging_setup import setup_file_logging\n'
            'setup_file_logging(pathlib.Path(sys.argv[1]), enabled=True)\n'
            'faulthandler._sigsegv()\n'
        )
        result = subprocess.run(  # noqa: S603 -- fixed interpreter and inline script
            [sys.executable, '-c', script, str(tmp_path)],
            capture_output=True,
            timeout=60,
            check=False,
        )

        assert result.returncode != 0
        log_text = (tmp_path / DEBUG_LOG.name).read_text(encoding='utf-8', errors='replace')
        assert 'Fatal Python error' in log_text
