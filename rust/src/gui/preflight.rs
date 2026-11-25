//! The checks before a recording starts, as the TUI's preflight screen: one
//! row per check, a fix under each failure, the summary, then the buttons.

use std::rc::Rc;

use gpui_kit::component::Disableable as _;
use gpui_kit::component::button::{Button, ButtonVariants as _};
use gpui_kit::prelude::FluentBuilder as _;
use gpui_kit::*;

use crate::app::preflight::{Check, PreflightOutcome, Status};
use crate::app::preflight_runtime::{PreflightRuntime, Summary};
use crate::gui::bridge::{Bridge, Incoming};
use crate::gui::theme::{color, hsla};
use crate::gui::widgets::{footer, header, hint, panel};

const TITLE: &str = "Checking everything before the meeting starts";

/// The mark colors the TUI takes from Textual's markup color names.
const GREEN: crate::app::theme::Rgb = crate::app::theme::Rgb(0x00, 0x80, 0x00);
const YELLOW: crate::app::theme::Rgb = crate::app::theme::Rgb(0xff, 0xff, 0x00);
const RED: crate::app::theme::Rgb = crate::app::theme::Rgb(0xff, 0x00, 0x00);
const CYAN: crate::app::theme::Rgb = crate::app::theme::Rgb(0x00, 0xff, 0xff);

pub enum PreflightEvent {
    Back,
    /// The checks passed, or the user started anyway: `take_outcome` has it.
    Start,
}

pub struct PreflightView {
    runtime: PreflightRuntime,
    bridge: Bridge,
    outcome: Option<PreflightOutcome>,
    focus: FocusHandle,
    _pump: Task<()>,
}

impl EventEmitter<PreflightEvent> for PreflightView {}

impl PreflightView {
    pub fn new(runtime: PreflightRuntime, window: &mut Window, cx: &mut Context<Self>) -> Self {
        let mut bridge = Bridge::new();
        let pump = bridge.pump(window, cx, Self::incoming);
        let mut view = Self {
            runtime,
            bridge,
            outcome: None,
            focus: cx.focus_handle(),
            _pump: pump,
        };
        view.focus.focus(window, cx);
        view.runtime.begin(&view.bridge.tasks);
        view
    }

    /// The outcome of a passed preflight, once.
    pub fn take_outcome(&mut self) -> Option<PreflightOutcome> {
        self.outcome.take()
    }

    fn incoming(&mut self, incoming: Incoming, _: &mut Window, cx: &mut Context<Self>) {
        let Incoming::Task(payload) = incoming else {
            return;
        };
        if let Ok(Some(outcome)) = self.runtime.task_result(payload) {
            self.start(outcome, cx);
        }
    }

    fn start(&mut self, outcome: PreflightOutcome, cx: &mut Context<Self>) {
        self.outcome = Some(outcome);
        cx.emit(PreflightEvent::Start);
    }

    fn press(&mut self, name: &str, cx: &mut Context<Self>) {
        match name {
            "r" if !self.runtime.running => self.runtime.retry(&self.bridge.tasks),
            "escape" => {
                if self.runtime.back() {
                    cx.emit(PreflightEvent::Back);
                }
            }
            "start" => {
                if let Some(outcome) = self.runtime.start_anyway() {
                    self.start(outcome, cx);
                }
            }
            _ => {}
        }
        cx.notify();
    }

    fn check_row(check: &Check, cx: &App) -> Div {
        let (mark, mark_color) = match check.status {
            Status::Pass => ("✓", hsla(GREEN)),
            Status::Warn => ("!", hsla(YELLOW)),
            Status::Fail => ("✗", hsla(RED)),
            Status::Skip => ("-", color(cx, "text-muted")),
            Status::Running => ("…", hsla(CYAN)),
            Status::Pending => ("·", color(cx, "text-muted")),
        };
        let shows_fix =
            matches!(check.status, Status::Fail | Status::Warn) && !check.fix.is_empty();
        div()
            .flex()
            .flex_col()
            .gap_1()
            .child(
                div()
                    .flex()
                    .gap_2()
                    .child(
                        div()
                            .w(px(14.))
                            .flex_none()
                            .text_color(mark_color)
                            .child(mark),
                    )
                    .child(
                        div()
                            .flex_none()
                            .font_weight(FontWeight::BOLD)
                            .child(check.title),
                    )
                    .child(div().flex_1().child(check.detail.clone())),
            )
            .when(shows_fix, |row| {
                row.child(
                    div()
                        .pl(px(22.))
                        .flex()
                        .gap_1()
                        .child(div().text_color(color(cx, "text-muted")).child("Fix:"))
                        .child(div().flex_1().child(check.fix.clone())),
                )
            })
    }
}

impl Focusable for PreflightView {
    fn focus_handle(&self, _: &App) -> FocusHandle {
        self.focus.clone()
    }
}

impl Render for PreflightView {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let rows: Vec<Div> = self
            .runtime
            .checks
            .iter()
            .map(|check| Self::check_row(check, cx))
            .collect();
        let summary = match &self.runtime.summary {
            Summary::None => None,
            Summary::Red(message) => Some((message.clone(), hsla(RED))),
            Summary::Yellow(message) => Some((message.clone(), hsla(YELLOW))),
        };
        let running = self.runtime.running;
        let buttons = div()
            .flex()
            .gap_2()
            .pt_2()
            .child(
                Button::new("retry")
                    .primary()
                    .label("Retry")
                    .disabled(running)
                    .on_click(cx.listener(|view, _, _, cx| view.press("r", cx))),
            )
            .when(self.runtime.show_override, |buttons| {
                buttons.child(
                    Button::new("start-anyway")
                        .warning()
                        .label("Start anyway")
                        .on_click(cx.listener(|view, _, _, cx| view.press("start", cx))),
                )
            })
            .child(
                Button::new("back")
                    .label("Back")
                    .on_click(cx.listener(|view, _, _, cx| view.press("escape", cx))),
            );
        let body = panel(TITLE, true, cx)
            .w(relative(0.9))
            .max_w(px(900.))
            .max_h(relative(0.9))
            .px_5()
            .pt_5()
            .pb_4()
            .child(
                div()
                    .id("checks")
                    .flex()
                    .flex_col()
                    .gap_2()
                    .overflow_y_scroll()
                    .children(rows)
                    .when_some(summary, |checks, (message, tint)| {
                        checks.child(div().pt_2().text_color(tint).child(message))
                    })
                    .child(buttons),
            );
        let entity = cx.entity().downgrade();
        let on_press = Rc::new(move |name: &'static str, _: &mut Window, cx: &mut App| {
            let _ = entity.update(cx, |view, cx| view.press(name, cx));
        });
        let mut hints = vec![hint("esc", "Back", "escape")];
        if !running {
            hints.insert(0, hint("r", "Retry", "r"));
        }
        div()
            .key_context("Preflight")
            .track_focus(&self.focus)
            .on_key_down(cx.listener(|view, event: &KeyDownEvent, _, cx| {
                let key = event.keystroke.key.clone();
                view.press(&key, cx);
            }))
            .size_full()
            .flex()
            .flex_col()
            .child(header("lazy-take-notes", cx))
            .child(
                div()
                    .flex_1()
                    .min_h_0()
                    .flex()
                    .items_center()
                    .justify_center()
                    .child(body),
            )
            .child(footer(hints, on_press, cx))
    }
}
