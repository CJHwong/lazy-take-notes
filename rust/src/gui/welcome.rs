//! The first screen: the mascot, then one row per mode, as the TUI's welcome
//! picker.

use std::rc::Rc;

use gpui_kit::prelude::FluentBuilder as _;
use gpui_kit::*;

use crate::app::modes::MODES;
use crate::gui::theme::{color, on};
use crate::gui::widgets::{brand, footer, hint, mascot};

actions!(welcome, [SelectPrev, SelectNext, Open, Leave]);

const CONTEXT: &str = "Welcome";

pub fn bind_keys(cx: &mut App) {
    cx.bind_keys([
        KeyBinding::new("up", SelectPrev, Some(CONTEXT)),
        KeyBinding::new("down", SelectNext, Some(CONTEXT)),
        KeyBinding::new("enter", Open, Some(CONTEXT)),
        KeyBinding::new("escape", Leave, Some(CONTEXT)),
    ]);
}

/// What the welcome screen asks the shell for.
pub enum WelcomeEvent {
    Open(&'static str),
    Quit,
}

pub struct Welcome {
    selected: usize,
    focus: FocusHandle,
}

impl EventEmitter<WelcomeEvent> for Welcome {}

impl Welcome {
    /// Starts on `last_mode` when the user comes back from it.
    pub fn new(last_mode: Option<&str>, window: &mut Window, cx: &mut Context<Self>) -> Self {
        let selected = last_mode
            .and_then(|mode| MODES.iter().position(|(key, _, _)| *key == mode))
            .unwrap_or(0);
        let focus = cx.focus_handle();
        focus.focus(window, cx);
        Self { selected, focus }
    }

    fn open(&mut self, index: usize, cx: &mut Context<Self>) {
        self.selected = index;
        cx.emit(WelcomeEvent::Open(MODES[index].0));
    }

    fn press(&mut self, name: &str, cx: &mut Context<Self>) {
        match name {
            "up" => self.selected = (self.selected + MODES.len() - 1) % MODES.len(),
            "down" => self.selected = (self.selected + 1) % MODES.len(),
            "enter" => self.open(self.selected, cx),
            "escape" => cx.emit(WelcomeEvent::Quit),
            _ => {}
        }
        cx.notify();
    }
}

impl Focusable for Welcome {
    fn focus_handle(&self, _: &App) -> FocusHandle {
        self.focus.clone()
    }
}

impl Render for Welcome {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let rows = MODES
            .iter()
            .enumerate()
            .map(|(index, (_, label, description))| {
                let selected = index == self.selected;
                let (background, text, muted) = if selected {
                    (
                        color(cx, "block-cursor-background"),
                        on(cx, "block-cursor-foreground", "block-cursor-background"),
                        on(cx, "block-cursor-foreground", "block-cursor-background"),
                    )
                } else {
                    (
                        color(cx, "background"),
                        color(cx, "foreground"),
                        color(cx, "text-muted"),
                    )
                };
                div()
                    .id(("mode", index))
                    .flex()
                    .items_baseline()
                    .gap_3()
                    .w_full()
                    .px_3()
                    .py_2()
                    .rounded(px(6.))
                    .bg(background)
                    .when(!selected, |row| {
                        row.hover(|row| row.bg(color(cx, "block-hover-background")))
                    })
                    .cursor_pointer()
                    .child(
                        div()
                            .w(px(130.))
                            .flex_none()
                            .font_weight(FontWeight::BOLD)
                            .text_color(text)
                            .child(*label),
                    )
                    .child(div().text_sm().text_color(muted).child(*description))
                    .on_click(cx.listener(move |this, _, _, cx| this.open(index, cx)))
            });
        let entity = cx.entity().downgrade();
        let on_press = Rc::new(move |name: &'static str, _: &mut Window, cx: &mut App| {
            let _ = entity.update(cx, |this, cx| this.press(name, cx));
        });
        div()
            .key_context(CONTEXT)
            .track_focus(&self.focus)
            .on_action(cx.listener(|this, _: &SelectPrev, _, cx| this.press("up", cx)))
            .on_action(cx.listener(|this, _: &SelectNext, _, cx| this.press("down", cx)))
            .on_action(cx.listener(|this, _: &Open, _, cx| this.press("enter", cx)))
            .on_action(cx.listener(|this, _: &Leave, _, cx| this.press("escape", cx)))
            .size_full()
            .flex()
            .flex_col()
            .bg(color(cx, "background"))
            .text_color(color(cx, "foreground"))
            .child(
                div()
                    .flex_1()
                    .flex()
                    .flex_col()
                    .items_center()
                    .justify_center()
                    .gap_6()
                    .child(
                        div()
                            .flex()
                            .flex_col()
                            .items_center()
                            .gap_3()
                            .child(mascot(px(7.)))
                            .child(brand(px(26.))),
                    )
                    .child(
                        div()
                            .w(px(520.))
                            .flex()
                            .flex_col()
                            .gap_1()
                            .p_2()
                            .border_1()
                            .border_color(color(cx, "border-blurred"))
                            .rounded(px(10.))
                            .children(rows),
                    ),
            )
            .child(footer(
                vec![
                    hint("⏎", "Select", "enter"),
                    hint("↑↓", "Navigate", "down"),
                    hint("esc", "Quit", "escape"),
                ],
                on_press,
                cx,
            ))
    }
}
