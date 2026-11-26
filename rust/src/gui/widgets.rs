//! The pieces every screen is built from. Each takes its sizes, spaces and
//! colors from `design`, so two screens that use one piece look the same.

use std::cell::Cell;
use std::rc::Rc;

use gpui_kit::component::button::{Button, ButtonVariants as _};
use gpui_kit::component::input::{Input, InputState};
use gpui_kit::component::text::{TextView, TextViewState, TextViewStyle};
use gpui_kit::component::tooltip::Tooltip;
use gpui_kit::component::{ActiveTheme as _, Icon, Sizable as _};
use gpui_kit::prelude::FluentBuilder as _;
use gpui_kit::*;

use crate::app::logo;
use crate::gui::design::{control, radius, size, space, text, tint};
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
        .font_weight(text::BRAND)
        .child(div().text_color(hsla(logo::PIG)).child("lazy-"))
        .child(div().text_color(hsla(logo::TEAL)).child("take-notes"))
}

/// The app icon at `edge` points, from the repo's high-resolution PNG.
pub fn app_icon(edge: Pixels) -> Img {
    img(crate::gui::icons::APP_ICON).size(edge)
}

/// The title bar row of a pane: the pane inset, a title on the left, room for
/// the pane's own controls after it, and the one hairline under it.
pub fn pane_header(id: impl Into<ElementId>, arm: &DragArm, cx: &App) -> Stateful<Div> {
    title_strip(id, arm)
        .px(space::LG)
        .gap(space::SM)
        .border_b(size::HAIRLINE)
        .border_color(cx.theme().border)
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

/// A button in a title bar: an icon and a word, flat until the pointer is
/// over it.
pub fn tool(id: impl Into<ElementId>, icon: impl Into<Icon>, label: &'static str) -> Button {
    Button::new(id)
        .ghost()
        .with_size(control::CONTROL)
        .icon(icon)
        .label(label)
}

/// A small button inside a column header: Copy, Digest Now.
pub fn inline_tool(id: impl Into<ElementId>, icon: impl Into<Icon>, label: &'static str) -> Button {
    Button::new(id)
        .ghost()
        .with_size(control::INLINE)
        .icon(icon)
        .label(label)
}

/// A session's title as an editable field in the title bar. Its text starts
/// at the pane inset, as every pane title does.
pub fn title_field(
    id: impl Into<ElementId>,
    state: &Entity<InputState>,
    cx: &App,
) -> Stateful<Div> {
    let hover = cx.theme().foreground.opacity(tint::SUBTLE);
    div()
        .id(id)
        .w(size::TITLE_FIELD)
        .min_w(size::TITLE_FIELD_MIN)
        .flex_shrink(1.)
        .rounded(radius::MD)
        .hover(|field| field.bg(hover))
        .tooltip(|window, cx| Tooltip::new("Rename").build(window, cx))
        // A press in the field selects text; it must not move the window.
        .on_mouse_down(MouseButton::Left, |_, _, cx| cx.stop_propagation())
        .child(
            Input::new(state)
                .appearance(false)
                .px(space::NONE)
                .text_size(text::TITLE)
                .font_weight(text::STRONG),
        )
}

/// Where a line sits in its speaker's turn. A turn draws as one rounded
/// block: the first line opens it under the speaker's name and time, the last
/// line closes it. A line with no speaker is a turn of its own.
#[derive(Clone, Copy)]
pub struct TurnEdge {
    pub first: bool,
    pub last: bool,
}

/// The colors of a transcript line: the block fill, the fill while its
/// speaker is pointed at, and the muted color of the time.
#[derive(Clone, Copy)]
pub struct TurnColors {
    pub fill: Hsla,
    pub marked: Option<Hsla>,
    pub muted: Hsla,
}

/// One transcript line, as a slice of its turn's block.
pub fn transcript_line(
    time: impl Into<SharedString>,
    speaker: Option<SharedString>,
    line: impl Into<SharedString>,
    edge: TurnEdge,
    colors: TurnColors,
) -> Div {
    let header = edge.first.then(|| {
        div()
            .flex()
            .items_baseline()
            .gap(space::SM)
            .px(space::XS)
            .pb(space::XS)
            .text_size(text::CAPTION)
            .when_some(speaker, |header, name| {
                header.child(div().font_weight(text::STRONG).child(name))
            })
            .child(
                div()
                    .font_family(text::MONO)
                    .text_color(colors.muted)
                    .child(time.into()),
            )
    });
    let block = div()
        .px(space::MD)
        .line_height(text::READING_LINE)
        .bg(colors.marked.unwrap_or(colors.fill))
        .when(edge.first, |block| {
            block.pt(space::SM).rounded_t(radius::LG)
        })
        .when(edge.last, |block| block.pb(space::SM).rounded_b(radius::LG))
        .when(!edge.last, |block| block.pb(space::XXS))
        .child(line.into());
    div()
        .w_full()
        .px(space::SM)
        .when(edge.first, |row| row.pt(space::MD))
        .children(header)
        .child(block)
}

/// A speaker with their spoken time. A click names them.
pub fn speaker_chip(
    id: impl Into<ElementId>,
    label: impl Into<SharedString>,
    selected: bool,
    cx: &App,
) -> Stateful<Div> {
    let theme = cx.theme();
    let (background, foreground) = match selected {
        true => (theme.primary, theme.primary_foreground),
        false => (theme.secondary, theme.secondary_foreground),
    };
    let hover = theme.secondary_hover;
    div()
        .id(id)
        .flex_none()
        .px(space::SM)
        .py(space::XXS)
        .rounded_full()
        .bg(background)
        .text_color(foreground)
        .text_size(text::CAPTION)
        .cursor_pointer()
        .when(!selected, |chip| chip.hover(|chip| chip.bg(hover)))
        .tooltip(|window, cx| Tooltip::new("Name this speaker").build(window, cx))
        .child(label.into())
}

/// A row of a source list, as the Finder sidebar draws one: a soft grey
/// under the selected row, the text unchanged. The sidebar and the Settings
/// page list use it.
pub fn source_row(id: impl Into<ElementId>, selected: bool, cx: &App) -> Stateful<Div> {
    let theme = cx.theme();
    let hover = theme.sidebar_foreground.opacity(tint::SUBTLE);
    let selected_bg = theme.sidebar_foreground.opacity(tint::STRONG);
    div()
        .id(id)
        .flex()
        .flex_none()
        .items_center()
        .gap(space::SM)
        .px(space::SM)
        .h(size::ROW)
        .rounded(radius::MD)
        .text_size(text::BODY)
        .text_color(theme.sidebar_foreground)
        .when(selected, |row| row.bg(selected_bg))
        .when(!selected, |row| row.hover(|row| row.bg(hover)))
}

/// A rounded card on the window background: a Settings section, a role of
/// a template.
pub fn card(cx: &App) -> Div {
    let theme = cx.theme();
    div()
        .flex()
        .flex_col()
        .rounded(radius::LG)
        .border(size::HAIRLINE)
        .border_color(theme.border)
        .bg(theme.foreground.opacity(tint::CARD))
}
