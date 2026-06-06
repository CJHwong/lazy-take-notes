"""Tests for the session status gateway (write/read/liveness)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import cast

from lazy_take_notes.l1_entities.session_status import RECORDING, SessionStatus
from lazy_take_notes.l3_interface_adapters.gateways.session_status import (
    STATUS_FILE,
    is_pid_alive,
    read_status,
    write_status,
)


def _status() -> SessionStatus:
    return SessionStatus(
        state=RECORDING,
        pid=4242,
        started_at='2026-06-06T09:00:00',
        updated_at='2026-06-06T09:01:00',
        segment_count=7,
        digest_count=2,
    )


class TestWriteRead:
    def test_round_trip(self, tmp_path: Path):
        write_status(tmp_path, _status())

        result = read_status(tmp_path)

        assert result == _status()

    def test_write_is_atomic_no_leftover_tmp(self, tmp_path: Path):
        write_status(tmp_path, _status())
        assert (tmp_path / STATUS_FILE).exists()
        assert not (tmp_path / (STATUS_FILE + '.tmp')).exists()

    def test_read_missing_returns_none(self, tmp_path: Path):
        assert read_status(tmp_path) is None

    def test_read_corrupt_returns_none(self, tmp_path: Path):
        (tmp_path / STATUS_FILE).write_text('{not json', encoding='utf-8')
        assert read_status(tmp_path) is None

    def test_read_wrong_schema_returns_none(self, tmp_path: Path):
        (tmp_path / STATUS_FILE).write_text('{"unexpected": 1}', encoding='utf-8')
        assert read_status(tmp_path) is None


class TestPidLiveness:
    def test_current_process_is_alive(self):
        assert is_pid_alive(os.getpid()) is True

    def test_zero_and_negative_are_dead(self):
        assert is_pid_alive(0) is False
        assert is_pid_alive(-1) is False

    def test_unused_pid_is_dead(self):
        # Very high PID unlikely to exist on a normal system.
        assert is_pid_alive(2_000_000_000) is False

    def test_non_int_pid_is_dead(self):
        # Guards a corrupt status file that stored pid as a wrong type (cast to
        # satisfy the type checker — the runtime guard is exactly what's tested).
        assert is_pid_alive(cast(int, '1234')) is False
        assert is_pid_alive(cast(int, None)) is False

    def test_permission_error_means_alive(self, monkeypatch):
        def _raise(_pid, _sig):
            raise PermissionError

        monkeypatch.setattr(os, 'kill', _raise)
        # A process we can't signal still exists.
        assert is_pid_alive(12345) is True

    def test_other_oserror_means_dead(self, monkeypatch):
        def _raise(_pid, _sig):
            raise OSError

        monkeypatch.setattr(os, 'kill', _raise)
        assert is_pid_alive(12345) is False
