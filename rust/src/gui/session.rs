//! A recording or a file transcription: transcript, digest, context, status.

use std::rc::Rc;

use gpui_kit::*;

use crate::app::session_start::{FileStart, RecordStart};
use crate::gui::widgets::{footer, header, hint};

pub enum SessionEvent {
    /// Back to the menu.
    Closed,
    /// The window closes: the app ends.
    Exit,
}

pub struct SessionView {}

impl EventEmitter<SessionEvent> for SessionView {}

impl SessionView {
    pub fn record(_start: RecordStart, _: &mut Window, _: &mut Context<Self>) -> Self {
        Self {}
    }

    pub fn file(_start: FileStart, _: &mut Window, _: &mut Context<Self>) -> Self {
        Self {}
    }

    /// The window may close now, or the session starts its way out first.
    pub fn should_close(&mut self, _: &mut Window, _: &mut Context<Self>) -> bool {
        true
    }
}

impl Render for SessionView {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let entity = cx.entity().downgrade();
        let on_press = Rc::new(move |_: &'static str, _: &mut Window, cx: &mut App| {
            let _ = entity.update(cx, |_, cx| cx.emit(SessionEvent::Closed));
        });
        div()
            .size_full()
            .flex()
            .flex_col()
            .child(header("Session", cx))
            .child(div().flex_1())
            .child(footer(vec![hint("esc", "Back", "escape")], on_press, cx))
    }
}
