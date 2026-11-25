//! The saved sessions, newest first.

use std::path::{Path, PathBuf};
use std::rc::Rc;

use gpui_kit::*;

use crate::gui::widgets::{footer, header, hint};

pub enum SessionPickerEvent {
    Picked(PathBuf),
    Cancelled,
}

pub struct SessionPickerView {}

impl EventEmitter<SessionPickerEvent> for SessionPickerView {}

impl SessionPickerView {
    pub fn new(_base_dir: &Path, _: &mut Window, _: &mut Context<Self>) -> Self {
        Self {}
    }
}

impl Render for SessionPickerView {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let entity = cx.entity().downgrade();
        let on_press = Rc::new(move |_: &'static str, _: &mut Window, cx: &mut App| {
            let _ = entity.update(cx, |_, cx| cx.emit(SessionPickerEvent::Cancelled));
        });
        div()
            .size_full()
            .flex()
            .flex_col()
            .child(header("Saved sessions", cx))
            .child(div().flex_1())
            .child(footer(vec![hint("esc", "Back", "escape")], on_press, cx))
    }
}
