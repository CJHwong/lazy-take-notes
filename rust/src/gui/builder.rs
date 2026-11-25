//! The template builder: a chat with the AI on the left, the template it
//! proposes on the right.

use std::rc::Rc;

use gpui_kit::*;

use crate::app::template_builder::TemplateBuilderRuntime;
use crate::gui::widgets::{footer, header, hint};

pub enum BuilderEvent {
    Closed,
}

pub struct BuilderView {
    _runtime: TemplateBuilderRuntime,
}

impl EventEmitter<BuilderEvent> for BuilderView {}

impl BuilderView {
    pub fn new(runtime: TemplateBuilderRuntime, _: &mut Window, _: &mut Context<Self>) -> Self {
        Self { _runtime: runtime }
    }
}

impl Render for BuilderView {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let entity = cx.entity().downgrade();
        let on_press = Rc::new(move |_: &'static str, _: &mut Window, cx: &mut App| {
            let _ = entity.update(cx, |_, cx| cx.emit(BuilderEvent::Closed));
        });
        div()
            .size_full()
            .flex()
            .flex_col()
            .child(header("Template builder", cx))
            .child(div().flex_1())
            .child(footer(vec![hint("esc", "Back", "escape")], on_press, cx))
    }
}
