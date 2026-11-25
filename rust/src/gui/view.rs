//! One saved session, read only.

use std::path::PathBuf;
use std::rc::Rc;

use gpui_kit::*;

use crate::gui::widgets::{footer, header, hint};

pub enum ReaderEvent {
    Closed,
}

pub struct SessionReader {}

impl EventEmitter<ReaderEvent> for SessionReader {}

impl SessionReader {
    pub fn new(_dir: PathBuf, _: &mut Window, _: &mut Context<Self>) -> Self {
        Self {}
    }
}

impl Render for SessionReader {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let entity = cx.entity().downgrade();
        let on_press = Rc::new(move |_: &'static str, _: &mut Window, cx: &mut App| {
            let _ = entity.update(cx, |_, cx| cx.emit(ReaderEvent::Closed));
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
