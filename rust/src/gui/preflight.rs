//! The checks before a recording starts, inside the start panel: one row per
//! check, a fix under each failure, the summary, then the buttons.

use gpui_kit::component::button::{Button, ButtonVariants as _};
use gpui_kit::component::{ActiveTheme as _, Icon, IconName, Sizable as _};
use gpui_kit::prelude::FluentBuilder as _;
use gpui_kit::*;

use crate::app::preflight::{Check, PreflightOutcome, Status};
use crate::app::preflight_runtime::{PreflightRuntime, Summary};
use crate::gui::bridge::{Bridge, Incoming};
use crate::gui::design::{control, size, space, tint};
use crate::gui::icons;
use crate::gui::widgets::{page_title, section_heading};

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

    /// The checks are still running: the start button waits on them.
    pub fn running(&self) -> bool {
        self.runtime.running
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
        let faint = theme.muted_foreground.opacity(tint::FAINT);
        let (icon, tint) = match check.status {
            Status::Pass => (Icon::new(IconName::CircleCheck), theme.success),
            Status::Warn => (Icon::new(IconName::TriangleAlert), theme.warning),
            Status::Fail => (Icon::new(IconName::CircleX), theme.danger),
            Status::Skip => (Icon::new(IconName::Minus), theme.muted_foreground),
            Status::Running => (Icon::new(IconName::LoaderCircle), theme.primary),
            Status::Pending => (icons::extra("circle-dashed"), faint),
        };
        let shows_fix =
            matches!(check.status, Status::Fail | Status::Warn) && !check.fix.is_empty();
        let muted = theme.muted_foreground;
        div()
            .flex()
            .gap(space::MD)
            .py(space::MD)
            .border_b(size::HAIRLINE)
            .border_color(theme.border)
            .child(
                div()
                    .flex_none()
                    .child(icon.size(size::ICON).text_color(tint)),
            )
            .child(
                div()
                    .flex_1()
                    .min_w_0()
                    .flex()
                    .flex_col()
                    .gap(space::XS)
                    .child(section_heading(check.title))
                    .when(!check.detail.is_empty(), |row| {
                        row.child(div().text_color(muted).child(check.detail.clone()))
                    })
                    .when(shows_fix, |row| {
                        row.child(div().child(format!("Fix: {}", check.fix)))
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
        let summary_shown = !matches!(self.runtime.summary, Summary::None);
        let summary = match &self.runtime.summary {
            Summary::None => None,
            Summary::Red(message) => Some((message.clone(), theme.danger)),
            Summary::Yellow(message) => Some((message.clone(), theme.warning)),
        };
        let running = self.runtime.running;
        // The way on from a failed check sits under its result. While the
        // checks run, the title bar's start button shows the wait.
        let actions = div()
            .flex()
            .items_center()
            .gap(space::SM)
            .pt(space::MD)
            .child(
                Button::new("preflight-back")
                    .with_size(control::CONTROL)
                    .icon(IconName::ArrowLeft)
                    .label("Back")
                    .on_click(cx.listener(|view, _, _, cx| view.back(cx))),
            )
            .child(
                Button::new("retry")
                    .with_size(control::CONTROL)
                    .icon(IconName::RefreshCw)
                    .label("Check Again")
                    .on_click(cx.listener(|view, _, _, cx| view.retry(cx))),
            )
            .when(self.runtime.show_override, |actions| {
                actions.child(
                    Button::new("start-anyway")
                        .primary()
                        .with_size(control::CONTROL)
                        .label("Start Anyway")
                        .on_click(cx.listener(|view, _, _, cx| view.start_anyway(cx))),
                )
            });
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
                    .px(space::XL)
                    .pb(space::LG)
                    .overflow_y_scroll()
                    .child(page_title(TITLE).pt(space::XL).pb(space::SM))
                    .children(rows)
                    .when_some(summary, |checks, (message, tint)| {
                        checks.child(section_heading(message).pt(space::MD).text_color(tint))
                    })
                    .when(!running && summary_shown, |checks| checks.child(actions)),
            )
    }
}
