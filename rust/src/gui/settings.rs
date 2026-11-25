//! The settings pages.

use std::path::PathBuf;
use std::rc::Rc;

use gpui_kit::*;

use crate::adapters::paths::Paths;
use crate::gui::widgets::{footer, header, hint};

pub enum SettingsEvent {
    Closed,
}

pub struct SettingsView {}

impl EventEmitter<SettingsEvent> for SettingsView {}

impl SettingsView {
    pub fn new(
        _paths: Paths,
        _config_path: Option<PathBuf>,
        _: &mut Window,
        _: &mut Context<Self>,
    ) -> Self {
        Self {}
    }
}

impl Render for SettingsView {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let entity = cx.entity().downgrade();
        let on_press = Rc::new(move |_: &'static str, _: &mut Window, cx: &mut App| {
            let _ = entity.update(cx, |_, cx| cx.emit(SettingsEvent::Closed));
        });
        div()
            .size_full()
            .flex()
            .flex_col()
            .child(header("Settings", cx))
            .child(div().flex_1())
            .child(footer(vec![hint("esc", "Back", "escape")], on_press, cx))
    }
}
