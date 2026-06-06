"""Tests for the keep-awake (caffeinate) inhibitor."""

from __future__ import annotations

from unittest.mock import MagicMock

from lazy_take_notes.l4_frameworks_and_drivers import keep_awake as ka


class TestInhibitSleep:
    def test_non_darwin_is_noop(self, monkeypatch):
        monkeypatch.setattr(ka.sys, 'platform', 'linux')
        assert ka.inhibit_sleep() is None

    def test_darwin_spawns_caffeinate(self, monkeypatch):
        monkeypatch.setattr(ka.sys, 'platform', 'darwin')
        fake_popen = MagicMock(return_value='handle')
        monkeypatch.setattr(ka.subprocess, 'Popen', fake_popen)

        result = ka.inhibit_sleep()

        assert result == 'handle'
        args = fake_popen.call_args.args[0]
        assert args[0] == 'caffeinate'
        assert '-di' in args
        # Own session so a terminal Ctrl-C can't kill the inhibitor mid-shutdown.
        assert fake_popen.call_args.kwargs['start_new_session'] is True


class TestReleaseSleep:
    def test_none_is_noop(self):
        ka.release_sleep(None)  # must not raise

    def test_terminates_handle(self):
        handle = MagicMock()
        ka.release_sleep(handle)
        handle.terminate.assert_called_once()


class TestKeepAwakeContext:
    def test_inhibits_then_releases(self, monkeypatch):
        handle = MagicMock()
        monkeypatch.setattr(ka, 'inhibit_sleep', lambda: handle)

        with ka.keep_awake():
            handle.terminate.assert_not_called()

        handle.terminate.assert_called_once()
