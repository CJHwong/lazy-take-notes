"""PreflightScreen: start only when the checks pass, or when the user overrides a soft failure."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

import pytest
from textual.widgets import Button, Static

from lazy_take_notes.l4_frameworks_and_drivers.apps.preflight_screen import PreflightScreen
from lazy_take_notes.l4_frameworks_and_drivers.preflight import FAIL, PASS, PreflightReport, new_checks

_RUN = 'lazy_take_notes.l4_frameworks_and_drivers.apps.preflight_screen.run_preflight'


def _report(**statuses) -> PreflightReport:
    checks = new_checks()
    for check in checks:
        check.set(statuses.get(check.key, PASS), f'{check.key} detail [/x]', f'{check.key} fix')
    return PreflightReport(checks)


async def _settle(pilot, turns=6):
    for _ in range(turns):
        await pilot.pause()


@pytest.mark.asyncio
async def test_all_checks_passing_starts_the_session():
    report = _report()
    screen = PreflightScreen(MagicMock())
    with patch(_RUN, return_value=report):
        async with screen.run_test() as pilot:
            await _settle(pilot)
    assert screen.return_value is report


@pytest.mark.asyncio
async def test_a_hard_failure_blocks_and_shows_the_fix():
    screen = PreflightScreen(MagicMock())
    with patch(_RUN, return_value=_report(mic=FAIL)):
        async with screen.run_test() as pilot:
            await _settle(pilot)
            line = str(screen.query_one('#check-mic', Static).render())
            assert 'mic fix' in line
            assert '[/x]' in line  # markup in a device error renders literally
            assert not screen.query_one('#preflight-override', Button).display
            assert not screen.query_one('#preflight-retry', Button).disabled
            await pilot.press('escape')
            await _settle(pilot)
    assert screen.return_value is None


@pytest.mark.asyncio
async def test_back_cancels_the_checks_and_never_waits_for_them():
    started = threading.Event()
    release = threading.Event()
    seen: list[bool] = []

    def slow_checks(plan, on_update=None):
        seen.append(threading.current_thread().daemon)
        started.set()
        release.wait(5)
        return _report()

    plan = MagicMock()
    screen = PreflightScreen(plan)
    with patch(_RUN, side_effect=slow_checks):
        async with screen.run_test() as pilot:
            started.wait(5)
            await pilot.press('escape')
            await _settle(pilot)
            plan.cancel.set.assert_called_once()
    release.set()
    assert screen.return_value is None
    assert seen == [True]  # a daemon thread: the app exits without waiting for a download


@pytest.mark.asyncio
async def test_an_llm_failure_offers_transcript_only():
    screen = PreflightScreen(MagicMock())
    with patch(_RUN, return_value=_report(llm=FAIL)):
        async with screen.run_test() as pilot:
            await _settle(pilot)
            override = screen.query_one('#preflight-override', Button)
            assert override.display
            assert 'without AI notes' in str(screen.query_one('#preflight-summary', Static).render())
            await pilot.click('#preflight-override')
            await _settle(pilot)
    assert screen.return_value is not None
    assert screen.return_value.overridden


@pytest.mark.asyncio
async def test_retry_runs_the_checks_again():
    results = iter([_report(mic=FAIL), _report()])
    screen = PreflightScreen(MagicMock())
    with patch(_RUN, side_effect=lambda *_args, **_kwargs: next(results)) as run:
        async with screen.run_test() as pilot:
            await _settle(pilot)
            await pilot.press('r')
            await _settle(pilot)
    assert run.call_count == 2
    assert screen.return_value is not None


@pytest.mark.asyncio
async def test_a_crash_in_the_checks_keeps_the_screen_alive():
    screen = PreflightScreen(MagicMock())
    with patch(_RUN, side_effect=RuntimeError('bug')):
        async with screen.run_test() as pilot:
            await _settle(pilot)
            assert 'bug' in str(screen.query_one('#preflight-summary', Static).render())
            assert screen.app.is_running


@pytest.mark.asyncio
async def test_retry_reads_the_config_again():
    # A user who fixes config.yaml after a failed check expects Retry to see the fix.
    first, second = MagicMock(owns_transcriber=True), MagicMock()
    results = iter([_report(mic=FAIL), _report()])
    seen: list = []

    def checks(plan, on_update=None):
        seen.append(plan)
        return next(results)

    screen = PreflightScreen(first, reload=lambda: second)
    with patch(_RUN, side_effect=checks):
        async with screen.run_test() as pilot:
            await _settle(pilot)
            await pilot.press('r')
            await _settle(pilot)
    assert seen == [first, second]
    assert screen.plan is second
    first.transcriber.close.assert_called_once()  # the old speech model child is not left running


@pytest.mark.asyncio
async def test_a_config_that_fails_to_load_on_retry_keeps_the_screen():
    def broken():
        raise FileNotFoundError('config.yaml was removed')

    plan = MagicMock()
    screen = PreflightScreen(plan, reload=broken)
    with patch(_RUN, return_value=_report(mic=FAIL)) as run:
        async with screen.run_test() as pilot:
            await _settle(pilot)
            await pilot.press('r')
            await _settle(pilot)
            summary = str(screen.query_one('#preflight-summary', Static).render())
            assert 'config.yaml' in summary
            await pilot.press('escape')
            await _settle(pilot)
    assert run.call_count == 1
    assert screen.plan is plan
