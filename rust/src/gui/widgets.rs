//! The pieces every screen is built from, in the terminal app's look: a header
//! bar, panels with their title in the border, a footer of key hints, and the
//! mascot.

use std::rc::Rc;

use gpui_kit::prelude::FluentBuilder as _;
use gpui_kit::*;

use crate::app::logo;
use crate::gui::theme::{color, fill, hsla, on};

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

/// The bar at the top, as the TUI's header: bold, centered, on `primary`.
pub fn header(title: impl Into<SharedString>, cx: &App) -> Div {
    div()
        .flex()
        .flex_none()
        .items_center()
        .justify_center()
        .h(px(30.))
        .px_4()
        .bg(color(cx, "primary"))
        .text_color(on(cx, "text", "primary"))
        .font_weight(FontWeight::BOLD)
        .text_sm()
        .child(
            div()
                .overflow_hidden()
                .whitespace_nowrap()
                .text_ellipsis()
                .child(title.into()),
        )
}

/// A box with a rounded border and its title in the top edge. The border
/// takes the primary color while the panel holds the focus.
pub fn panel(title: impl Into<SharedString>, focused: bool, cx: &App) -> Div {
    let border = if focused {
        color(cx, "primary")
    } else {
        color(cx, "border-blurred")
    };
    let title_color = if focused {
        color(cx, "primary")
    } else {
        color(cx, "text-muted")
    };
    let title: SharedString = title.into();
    div()
        .relative()
        .flex()
        .flex_col()
        .min_h_0()
        .min_w_0()
        .border_1()
        .border_color(border)
        .rounded(px(8.))
        .pt_3()
        .px_3()
        .pb_2()
        .when(!title.is_empty(), |panel| {
            panel.child(
                div()
                    .absolute()
                    .top(px(-9.))
                    .left(px(12.))
                    .px_1()
                    .bg(color(cx, "background"))
                    .text_xs()
                    .font_weight(FontWeight::SEMIBOLD)
                    .text_color(title_color)
                    .child(title),
            )
        })
}

/// One key the footer offers: the key as shown, and what it does.
#[derive(Clone)]
pub struct Hint {
    pub key: &'static str,
    pub label: SharedString,
    /// The key name the screen's key handler takes, as the TUI names it.
    pub name: &'static str,
}

pub fn hint(key: &'static str, label: impl Into<SharedString>, name: &'static str) -> Hint {
    Hint {
        key,
        label: label.into(),
        name,
    }
}

/// What a click on a footer hint runs, with the hint's key name.
pub type OnPress = Rc<dyn Fn(&'static str, &mut Window, &mut App)>;

/// The footer: each key in its key color, then its label. A click on a hint
/// does what the key does.
pub fn footer(hints: Vec<Hint>, on_press: OnPress, cx: &App) -> Div {
    let items = hints.into_iter().enumerate().map(|(index, hint)| {
        let on_press = on_press.clone();
        let name = hint.name;
        div()
            .id(("hint", index))
            .flex()
            .items_center()
            .h_full()
            .cursor_pointer()
            .hover(|item| item.bg(color(cx, "block-hover-background")))
            .child(
                div()
                    .px_2()
                    .h_full()
                    .flex()
                    .items_center()
                    .when_some(fill(cx, "footer-key-background"), |key, fill| key.bg(fill))
                    .text_color(on(cx, "footer-key-foreground", "footer-background"))
                    .font_weight(FontWeight::BOLD)
                    .child(hint.key),
            )
            .child(
                div()
                    .pl_1()
                    .pr_3()
                    .text_color(on(cx, "footer-description-foreground", "footer-background"))
                    .child(hint.label),
            )
            .on_click(move |_, window, cx| on_press(name, window, cx))
    });
    div()
        .flex()
        .flex_none()
        .items_center()
        .h(px(26.))
        .bg(color(cx, "footer-background"))
        .text_sm()
        .children(items)
}
