//! The checks before a recording starts, inside the start panel: one row per
//! check, a fix under each failure, the summary, then the buttons.

use gpui_kit::component::button::{Button, ButtonVariants as _};
use gpui_kit::component::{ActiveTheme as _, Disableable as _, Icon, IconName};
use gpui_kit::prelude::FluentBuilder as _;
use gpui_kit::*;

use crate::app::preflight::{Check, PreflightOutcome, Status};
use crate::app::preflight_runtime::{PreflightRuntime, Summary};
use crate::gui::bridge::{Bridge, Incoming};
use crate::gui::template_picker::BAR_HEIGHT;

const TITLE: &str = "Checking your setup";

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

    fn retry(&mut self, cx: &mut Context<Self>) {
        if !self.runtime.running {
            self.runtime.retry(&self.bridge.tasks);
        }
        cx.notify();
    }

    fn back(&mut self, cx: &mut Context<Self>) {
        if self.runtime.back() {
            cx.emit(PreflightEvent::Back);
        }
    }

    fn start_anyway(&mut self, cx: &mut Context<Self>) {
        if let Some(outcome) = self.runtime.start_anyway() {
            self.start(outcome, cx);
        }
    }

    fn check_row(check: &Check, cx: &App) -> Div {
        let theme = cx.theme();
        let (icon, tint) = match check.status {
            Status::Pass => (IconName::CircleCheck, theme.success),
            Status::Warn => (IconName::TriangleAlert, theme.warning),
            Status::Fail => (IconName::CircleX, theme.danger),
            Status::Skip => (IconName::Minus, theme.muted_foreground),
            Status::Running => (IconName::LoaderCircle, theme.muted_foreground),
            Status::Pending => (IconName::Dash, theme.muted_foreground.opacity(0.5)),
        };
        let shows_fix =
            matches!(check.status, Status::Fail | Status::Warn) && !check.fix.is_empty();
        let muted = theme.muted_foreground;
        div()
            .flex()
            .gap_3()
            .py_2()
            .border_b_1()
            .border_color(theme.border)
            .child(
                div()
                    .flex_none()
                    .pt(px(2.))
                    .child(Icon::new(icon).text_color(tint)),
            )
            .child(
                div()
                    .flex_1()
                    .min_w_0()
                    .flex()
                    .flex_col()
                    .gap_1()
                    .child(div().font_weight(FontWeight::SEMIBOLD).child(check.title))
                    .child(
                        div()
                            .text_sm()
                            .text_color(muted)
                            .child(check.detail.clone()),
                    )
                    .when(shows_fix, |row| {
                        row.child(div().text_sm().child(format!("Fix: {}", check.fix)))
                    }),
            )
    }
}

impl Drop for PreflightView {
    /// A click on another template, Cancel or the end of the flow drops the
    /// view: its checks stop and the speech model it loaded closes.
    fn drop(&mut self) {
        self.runtime.back();
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
        let theme = cx.theme();
        let summary = match &self.runtime.summary {
            Summary::None => None,
            Summary::Red(message) => Some((message.clone(), theme.danger)),
            Summary::Yellow(message) => Some((message.clone(), theme.warning)),
        };
        let border = theme.border;
        let running = self.runtime.running;
        let buttons = div()
            .flex()
            .flex_none()
            .items_center()
            .gap_2()
            .h(BAR_HEIGHT)
            .px_4()
            .border_t_1()
            .border_color(border)
            .child(
                Button::new("preflight-back")
                    .ghost()
                    .icon(IconName::ArrowLeft)
                    .label("Back")
                    .tooltip("Back to the template")
                    .on_click(cx.listener(|view, _, _, cx| view.back(cx))),
            )
            .child(div().flex_1())
            .when(self.runtime.show_override, |buttons| {
                buttons.child(
                    Button::new("start-anyway")
                        .label("Start Anyway")
                        .on_click(cx.listener(|view, _, _, cx| view.start_anyway(cx))),
                )
            })
            .child(
                Button::new("retry")
                    .primary()
                    .icon(IconName::RefreshCw)
                    .label(if running {
                        "Checking…"
                    } else {
                        "Check Again"
                    })
                    .loading(running)
                    .disabled(running)
                    .on_click(cx.listener(|view, _, _, cx| view.retry(cx))),
            );
        div()
            .key_context("Preflight")
            .track_focus(&self.focus)
            .on_key_down(cx.listener(|view, event: &KeyDownEvent, _, cx| {
                if event.keystroke.key == "escape" {
                    view.back(cx);
                }
            }))
            .size_full()
            .flex()
            .flex_col()
            .child(
                div()
                    .id("checks")
                    .flex_1()
                    .min_h_0()
                    .px_6()
                    .py_4()
                    .overflow_y_scroll()
                    .child(
                        div()
                            .pb_2()
                            .text_lg()
                            .font_weight(FontWeight::SEMIBOLD)
                            .child(TITLE),
                    )
                    .children(rows)
                    .when_some(summary, |checks, (message, tint)| {
                        checks.child(
                            div()
                                .pt_3()
                                .font_weight(FontWeight::SEMIBOLD)
                                .text_color(tint)
                                .child(message),
                        )
                    }),
            )
            .child(buttons)
    }
}
