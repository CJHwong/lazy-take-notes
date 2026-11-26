//! The pieces every screen is built from. Each takes its sizes, spaces and
//! colors from `design`, so two screens that use one piece look the same.

use std::cell::Cell;
use std::rc::Rc;

use gpui_kit::component::text::{TextView, TextViewState, TextViewStyle};
use gpui_kit::component::{ActiveTheme as _, Icon};
use gpui_kit::*;

use crate::app::logo;
use crate::gui::design::{radius, size, space, text, tint};
use crate::gui::theme::hsla;

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
        .h(size::TITLE_BAR)
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

/// The title bar row of a pane: the pane inset, a title on the left, and room
/// for the pane's own controls after it.
pub fn pane_header(id: impl Into<ElementId>, arm: &DragArm) -> Stateful<Div> {
    title_strip(id, arm).px(space::LG).gap(space::SM)
}

/// A pane title, as the title bar row shows it.
pub fn pane_title(title: impl Into<SharedString>) -> Div {
    div()
        .min_w_0()
        .truncate()
        .text_size(text::TITLE)
        .font_weight(text::STRONG)
        .child(title.into())
}

/// A page heading inside a pane: "Checking your setup", a template's name.
pub fn page_title(title: impl Into<SharedString>) -> Div {
    div()
        .text_size(text::LARGE_TITLE)
        .font_weight(text::STRONG)
        .child(title.into())
}

/// The heading of a section in a form or a list.
pub fn section_heading(title: impl Into<SharedString>) -> Div {
    div()
        .text_size(text::HEADLINE)
        .font_weight(text::STRONG)
        .child(title.into())
}

/// Small muted text: timestamps, hints, subtitles.
pub fn caption(content: impl Into<SharedString>, cx: &App) -> Div {
    div()
        .text_size(text::CAPTION)
        .text_color(cx.theme().muted_foreground)
        .child(content.into())
}

/// The heading row over a column: a muted label, then the column's buttons.
pub fn column_header(label: impl Into<SharedString>, cx: &App) -> Div {
    div()
        .flex()
        .flex_none()
        .items_center()
        .justify_between()
        .h(size::COLUMN_HEADER)
        .px(space::LG)
        .gap(space::SM)
        .text_size(text::CAPTION)
        .font_weight(text::STRONG)
        .text_color(cx.theme().muted_foreground)
        .child(label.into())
}

/// A label on a row: "User", a locale.
pub fn tag(label: impl Into<SharedString>, cx: &App) -> Div {
    let theme = cx.theme();
    div()
        .flex_none()
        .px(space::XS)
        .rounded(radius::SM)
        .bg(theme.muted_foreground.opacity(tint::SUBTLE))
        .text_size(text::CAPTION)
        .text_color(theme.muted_foreground)
        .child(label.into())
}

/// What a pane shows when it has nothing yet: an icon and one line.
pub fn empty_state(icon: impl Into<Icon>, line: impl Into<SharedString>, cx: &App) -> Div {
    let muted = cx.theme().muted_foreground;
    div()
        .size_full()
        .flex()
        .flex_col()
        .items_center()
        .justify_center()
        .gap(space::SM)
        .p(space::XXL)
        .text_color(muted)
        .child(
            icon.into()
                .size(size::ICON)
                .text_color(muted.opacity(tint::FAINT)),
        )
        .child(div().text_size(text::BODY).text_center().child(line.into()))
}

/// A bar of buttons under a pane. Put the main action last, on the right.
pub fn action_bar(cx: &App) -> Div {
    div()
        .flex()
        .flex_none()
        .items_center()
        .gap(space::SM)
        .h(size::ACTION_BAR)
        .px(space::LG)
        .border_t(size::HAIRLINE)
        .border_color(cx.theme().border)
}

/// A strip of status text or chips under a pane.
pub fn status_strip(cx: &App) -> Div {
    div()
        .flex()
        .flex_none()
        .items_center()
        .gap(space::MD)
        .h(size::STATUS_STRIP)
        .px(space::LG)
        .border_t(size::HAIRLINE)
        .border_color(cx.theme().border)
        .text_size(text::CAPTION)
        .text_color(cx.theme().muted_foreground)
}

/// Markdown as every pane shows it: headings from the type scale, so notes
/// never outshout the pane title. Only the display changes; Copy keeps the text.
pub fn markdown(state: &Entity<TextViewState>) -> TextView {
    TextView::new(state).selectable(true).style(
        TextViewStyle::default().heading_font_size(|level, _| {
            text::MARKDOWN_HEADINGS[usize::from(level.clamp(1, 6)) - 1]
        }),
    )
}
