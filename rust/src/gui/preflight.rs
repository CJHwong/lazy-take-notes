//! The checks before a recording starts.

use std::rc::Rc;

use gpui_kit::*;

use crate::app::preflight::PreflightOutcome;
use crate::app::preflight_runtime::PreflightRuntime;
use crate::gui::widgets::{footer, header, hint};

pub enum PreflightEvent {
    Back,
    Start,
}

pub struct PreflightView {
    _runtime: PreflightRuntime,
    outcome: Option<PreflightOutcome>,
}

impl EventEmitter<PreflightEvent> for PreflightView {}

impl PreflightView {
    pub fn new(runtime: PreflightRuntime, _: &mut Window, _: &mut Context<Self>) -> Self {
        Self {
            _runtime: runtime,
            outcome: None,
        }
    }

    /// The outcome of a passed preflight, once.
    pub fn take_outcome(&mut self) -> Option<PreflightOutcome> {
        self.outcome.take()
    }
}

impl Render for PreflightView {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let entity = cx.entity().downgrade();
        let on_press = Rc::new(move |_: &'static str, _: &mut Window, cx: &mut App| {
            let _ = entity.update(cx, |_, cx| cx.emit(PreflightEvent::Back));
        });
        div()
            .size_full()
            .flex()
            .flex_col()
            .child(header("Checking before recording", cx))
            .child(div().flex_1())
            .child(footer(vec![hint("esc", "Back", "escape")], on_press, cx))
    }
}
