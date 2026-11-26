//! The pieces every screen is built from: the title strip, the brand and the
//! mascot.

use std::cell::Cell;
use std::rc::Rc;

use gpui_kit::*;

use crate::app::logo;
use crate::gui::theme::hsla;

/// The height of the title bar row, which the traffic lights sit in.
pub const TITLE_BAR_HEIGHT: Pixels = px(44.);

/// A press on the strip that may turn into a window move. The owner keeps it
/// across renders, so a re-render during the press does not lose it.
pub type DragArm = Rc<Cell<bool>>;

/// The top row of a pane, in the title bar: a drag moves the window, and a
/// double click does what the system setting says (zoom or minimize). A
/// button on the strip takes its own press.
pub fn title_strip(id: impl Into<ElementId>, arm: &DragArm) -> Stateful<Div> {
    let (down, up, motion) = (arm.clone(), arm.clone(), arm.clone());
    div()
        .id(id)
        .flex()
        .flex_none()
        .items_center()
        .h(TITLE_BAR_HEIGHT)
        .on_mouse_down(MouseButton::Left, move |_, window, _| {
            down.set(!window.default_prevented());
        })
        .on_mouse_up(MouseButton::Left, move |_, _, _| up.set(false))
        .on_mouse_move(move |event: &MouseMoveEvent, window, _| {
            if event.pressed_button != Some(MouseButton::Left) {
                motion.set(false);
            } else if motion.replace(false) {
                window.start_window_move();
            }
        })
        .on_click(|event, window, _| {
            if event.click_count() == 2 {
                window.titlebar_double_click();
            }
        })
}

/// The pig and teal title, as the terminal's welcome banner writes it.
pub fn brand(size: Pixels) -> Div {
    div()
        .flex()
        .text_size(size)
        .font_weight(FontWeight::BOLD)
        .child(div().text_color(hsla(logo::PIG)).child("lazy-"))
        .child(div().text_color(hsla(logo::TEAL)).child("take-notes"))
}

/// The mascot, one square per pixel.
pub fn mascot(pixel: Pixels) -> Div {
    let rows = logo::PIXELS.iter().map(move |row| {
        div().flex().children(row.bytes().map(move |code| {
            let cell = div().w(pixel).h(pixel);
            match logo::pixel(code) {
                Some(rgb) => cell.bg(hsla(rgb)),
                None => cell,
            }
        }))
    });
    div().flex().flex_col().children(rows)
}
