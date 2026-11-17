"""Preflight screen: shows each requirement check before a recording starts."""

from __future__ import annotations

import logging
from collections.abc import Callable

from rich.markup import escape
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, VerticalScroll
from textual.message import Message
from textual.widgets import Button, Static

from lazy_take_notes.l4_frameworks_and_drivers.preflight import (
    FAIL,
    PASS,
    RUNNING,
    SKIP,
    WARN,
    Check,
    PreflightPlan,
    PreflightReport,
    new_checks,
    run_preflight,
)
from lazy_take_notes.l4_frameworks_and_drivers.screen_base import ShellScreen

log = logging.getLogger('ltn.preflight')

_MARKS = {
    PASS: '[green]✓[/]',
    WARN: '[yellow]![/]',
    FAIL: '[red]✗[/]',
    SKIP: '[dim]-[/]',
    RUNNING: '[cyan]…[/]',
}


class PreflightScreen(ShellScreen[PreflightReport | None]):
    """Runs the checks. Dismisses with the report to start, or None to go back."""

    DEFAULT_CSS = """
    PreflightScreen { align: center middle; }
    #preflight-box { width: 90%; max-width: 110; height: auto; max-height: 90%; border: round $accent; padding: 1 2; }
    #preflight-title { text-style: bold; margin-bottom: 1; }
    .preflight-check { height: auto; }
    #preflight-summary { margin-top: 1; height: auto; }
    #preflight-buttons { height: auto; margin-top: 1; }
    #preflight-buttons Button { margin-right: 2; }
    """

    BINDINGS = [
        Binding('r', 'retry', 'Retry'),
        Binding('escape', 'back', 'Back'),
    ]

    def __init__(self, plan: PreflightPlan) -> None:
        super().__init__()
        self._plan = plan
        self._report: PreflightReport | None = None
        self._checks_running = False
        self._left = False

    def compose(self) -> ComposeResult:
        with VerticalScroll(id='preflight-box'):
            yield Static('Checking everything before the meeting starts', id='preflight-title')
            for check in new_checks():
                yield Static(self._line(check), id=f'check-{check.key}', classes='preflight-check')
            yield Static('', id='preflight-summary')
            with Horizontal(id='preflight-buttons'):
                yield Button('Retry [r]', id='preflight-retry', variant='primary')
                yield Button('Start anyway', id='preflight-override', variant='warning')
                yield Button('Back [Esc]', id='preflight-back')

    def on_mount(self) -> None:
        self._begin_checks()

    def _begin_checks(self) -> None:
        if self._checks_running:
            return
        self._checks_running = True
        self._report = None
        self.query_one('#preflight-summary', Static).update('')
        self.query_one('#preflight-retry', Button).disabled = True
        self.query_one('#preflight-override', Button).display = False
        self.run_worker(self._run_checks, thread=True, exclusive=True, group='preflight')

    def _run_checks(self) -> None:
        try:
            report = run_preflight(self._plan, on_update=self._post_update)
        except Exception as error:  # run_preflight catches per check; this is the last guard
            log.exception('preflight crashed')
            self._call_ui(self._show_crash, error)
            return
        if self._left and self._plan.owns_transcriber:
            # Back was pressed while the model loaded; nobody will use this child.
            self._plan.transcriber.close()
            return
        self._call_ui(self._show_report, report)

    def _post_update(self, check: Check) -> None:
        snapshot = Check(check.key, check.title, check.status, check.detail, check.fix, check.overridable)
        self._call_ui(self._show_check, snapshot)

    class UiUpdate(Message):
        """A screen update sent from the check thread."""

        def __init__(self, callback: Callable, argument) -> None:
            super().__init__()
            self.callback = callback
            self.argument = argument

    def _call_ui(self, callback: Callable, argument) -> None:
        """Queue a screen update from the check thread without waiting for it.

        call_from_thread blocks the check thread until the event loop runs the
        update. With it, a real start hung: the loop thread waited on a lock and
        three checks stopped. A posted message never waits, so no check thread
        can hold anything while it depends on the loop.
        """
        if not self._left:
            self.post_message(self.UiUpdate(callback, argument))

    def on_preflight_screen_ui_update(self, message: UiUpdate) -> None:
        if not self._left:
            message.callback(message.argument)

    def _leave(self, result: PreflightReport | None) -> None:
        if self._left:
            return
        self._left = True
        self.dismiss(result)

    def _show_check(self, check: Check) -> None:
        if self._left:
            return
        self.query_one(f'#check-{check.key}', Static).update(self._line(check))

    @staticmethod
    def _line(check: Check) -> str:
        mark = _MARKS.get(check.status, '[dim]·[/]')
        line = f'{mark} [bold]{escape(check.title)}[/]'
        if check.detail:
            line += f'  {escape(check.detail)}'
        if check.status in (FAIL, WARN) and check.fix:
            line += f'\n    [dim]Fix:[/] {escape(check.fix)}'
        return line

    def _show_report(self, report: PreflightReport) -> None:
        self._checks_running = False
        if self._left:
            return
        self._report = report
        for check in report.checks:
            self._show_check(check)
        if report.ok:
            self._leave(report)
            return
        self.query_one('#preflight-retry', Button).disabled = False
        summary = self.query_one('#preflight-summary', Static)
        if report.blocked:
            summary.update('[red]Fix the items marked ✗, then press Retry.[/]')
            return
        summary.update(self._override_text(report))
        self.query_one('#preflight-override', Button).display = True

    @staticmethod
    def _override_text(report: PreflightReport) -> str:
        if not report.llm_ok:
            return '[yellow]You can start without AI notes: the transcript is still recorded.[/]'
        return '[yellow]You can start anyway, but this problem stays for the whole meeting.[/]'

    def _show_crash(self, error: Exception) -> None:
        self._checks_running = False
        self.query_one('#preflight-retry', Button).disabled = False
        self.query_one('#preflight-summary', Static).update(
            f'[red]The checks stopped: {escape(f"{type(error).__name__}: {error}")}. Press Retry.[/]'
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        actions = {
            'preflight-retry': self.action_retry,
            'preflight-override': self._override,
            'preflight-back': self.action_back,
        }
        action = actions.get(event.button.id or '')
        if action is not None:
            action()

    def action_retry(self) -> None:
        self._begin_checks()

    def action_back(self) -> None:
        self._leave(None)

    def _override(self) -> None:
        if self._report is None or self._report.blocked:
            return
        self._report.overridden = True
        self._leave(self._report)

    def stop_for_shutdown(self) -> bool:
        # The checks close their own sources; the whisper child dies with the process.
        return False
