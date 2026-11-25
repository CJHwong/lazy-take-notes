//! The template list with its preview.

use std::rc::Rc;

use gpui_kit::*;

use crate::app::container::Loaded;
use crate::gui::widgets::{footer, header, hint};

pub enum TemplatePickerEvent {
    Picked(String),
    Create,
    EditWithAi(String),
    Cancelled,
}

pub struct TemplatePickerView {}

impl EventEmitter<TemplatePickerEvent> for TemplatePickerView {}

impl TemplatePickerView {
    pub fn new(_loaded: &Loaded, _: &mut Window, _: &mut Context<Self>) -> Self {
        Self {}
    }
}

impl Render for TemplatePickerView {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let entity = cx.entity().downgrade();
        let on_press = Rc::new(move |_: &'static str, _: &mut Window, cx: &mut App| {
            let _ = entity.update(cx, |_, cx| cx.emit(TemplatePickerEvent::Cancelled));
        });
        div()
            .size_full()
            .flex()
            .flex_col()
            .child(header("Choose a template", cx))
            .child(div().flex_1())
            .child(footer(vec![hint("esc", "Back", "escape")], on_press, cx))
    }
}
