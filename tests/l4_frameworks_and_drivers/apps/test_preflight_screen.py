"""PreflightScreen: start only when the checks pass, or when the user overrides a soft failure."""

from __future__ import annotations

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
