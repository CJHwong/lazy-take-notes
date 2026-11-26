//! The design system of the window: every size, space, radius, type size,
//! tint and accent color a view may use. Views take their values from here,
//! never from a literal, so all screens keep the same rhythm.
//! `scripts/check_design.py` fails on a literal anywhere else in `gui/`.
//!
//! The scale follows the macOS standard density: 13 pt body text, 4 pt
//! spacing steps, 28 pt list rows, a 16 pt pane inset.

use std::process::Command;

use gpui_kit::component::Theme;
use gpui_kit::*;

/// Spacing steps. A gap or a padding is always one of these.
pub mod space {
    use gpui_kit::{Pixels, px};
    pub const NONE: Pixels = px(0.);
    /// Between the bars of a meter.
    pub const XXS: Pixels = px(2.);
    /// Between an icon and its label, or between two chips.
    pub const XS: Pixels = px(4.);
    /// Between the controls of one group.
    pub const SM: Pixels = px(8.);
    /// Between the rows of a form, and inside a list row.
    pub const MD: Pixels = px(12.);
    /// The pane inset: from a pane edge to its content and its title.
    pub const LG: Pixels = px(16.);
    /// Between the sections of a page.
    pub const XL: Pixels = px(24.);
    /// Around an empty state or a page heading.
    pub const XXL: Pixels = px(32.);
}

/// Corner radii.
pub mod radius {
    use gpui_kit::{Pixels, px};
    /// Chips, tags and inputs.
    pub const SM: Pixels = px(4.);
    /// Buttons, list rows and menus.
    pub const MD: Pixels = px(6.);
    /// Dialogs, chat bubbles and cards.
    pub const LG: Pixels = px(10.);
}

/// Fixed sizes of the window's frame.
pub mod size {
    use gpui_kit::{Pixels, px};
    /// The title bar row, which the traffic lights sit in: the height of a
    /// macOS unified toolbar.
    pub const TITLE_BAR: Pixels = px(52.);
    /// The traffic lights' top edge, centered in the title bar row.
    pub const TRAFFIC_LIGHTS_Y: Pixels = px(20.);
    /// A bar of buttons under a pane: the start panel, Settings, the builder.
    pub const ACTION_BAR: Pixels = px(48.);
    /// The bar under the start panel's preview, with the start buttons.
    pub const START_BAR: Pixels = px(64.);
    /// A status or chip strip under a pane: the live session, the reader.
    pub const STATUS_STRIP: Pixels = px(28.);
    /// The heading row over a column: Transcript, Digest, Preview.
    pub const COLUMN_HEADER: Pixels = px(36.);
    pub const SIDEBAR: Pixels = px(240.);
    /// The template list column of the start panel.
    pub const LIST_COLUMN: Pixels = px(300.);
    /// A list row.
    pub const ROW: Pixels = px(28.);
    pub const ICON: Pixels = px(16.);
    pub const ICON_SM: Pixels = px(14.);
    /// A status dot: recording, a check state.
    pub const DOT: Pixels = px(8.);
    /// One bar of a level meter.
    pub const METER_BAR: Pixels = px(3.);
    pub const METER_HEIGHT: Pixels = px(12.);
    /// A hairline: borders and dividers.
    pub const HAIRLINE: Pixels = px(1.);
    /// The app icon on the empty window.
    pub const LOGO: Pixels = px(128.);
    /// The Settings page list, a form row, the control column of a form,
    /// and the widest a Settings page grows.
    pub const SETTINGS_NAV: Pixels = px(200.);
    pub const FORM_ROW: Pixels = px(40.);
    pub const FORM_FIELD: Pixels = px(280.);
    pub const FORM_PAGE: Pixels = px(720.);
    /// The size and status columns of a model row, and its buttons.
    pub const STAT_COLUMN: Pixels = px(72.);
    pub const ACTION_COLUMN: Pixels = px(160.);
    /// The label column of a quick action in a template's card.
    pub const ACTION_LABEL: Pixels = px(96.);
    /// A locale code field.
    pub const LOCALE_FIELD: Pixels = px(96.);
    /// The widest chat bubble of the template builder.
    pub const BUBBLE_MAX: Pixels = px(520.);
    /// The Session Context panel under the notes, while it has no focus.
    pub const CONTEXT_PANEL: Pixels = px(140.);
    /// The title field of a session: wide enough for most names.
    pub const TITLE_FIELD: Pixels = px(360.);
    pub const TITLE_FIELD_MIN: Pixels = px(120.);
    /// Dialog widths: a question or a form, and a long text.
    pub const DIALOG: Pixels = px(440.);
    pub const DIALOG_WIDE: Pixels = px(640.);
    /// The tallest a dialog's text grows before it scrolls.
    pub const DIALOG_BODY: Pixels = px(480.);
    /// The window at first launch, and the smallest it gets.
    pub const WINDOW_W: Pixels = px(1200.);
    pub const WINDOW_H: Pixels = px(800.);
    pub const WINDOW_MIN_W: Pixels = px(760.);
    pub const WINDOW_MIN_H: Pixels = px(520.);
    /// How far a long list draws past its visible part.
    pub const LIST_OVERDRAW: Pixels = px(400.);
}

/// Control sizes, as the kit names them. Every button, field and menu in a
/// title bar or a form is `CONTROL` (24 pt). `HERO` (32 pt) is for the start
/// buttons and a list's own search row. Inline buttons in a column header are
/// `INLINE` (20 pt).
pub mod control {
    use gpui_kit::component::Size;
    pub const CONTROL: Size = Size::Small;
    pub const HERO: Size = Size::Medium;
    pub const INLINE: Size = Size::XSmall;
}

/// Type sizes. Weight comes with the role: see `text::weight`.
pub mod text {
    use gpui_kit::{FontWeight, Pixels, px};
    /// The page heading inside a pane, and the brand on the empty window.
    pub const LARGE_TITLE: Pixels = px(20.);
    /// The pane title in the title bar row.
    pub const TITLE: Pixels = px(15.);
    /// A section heading, and the title of a list row.
    pub const HEADLINE: Pixels = px(13.);
    pub const BODY: Pixels = px(13.);
    /// Transcript and notes: long text read for minutes.
    pub const READING: Pixels = px(14.);
    /// Timestamps, subtitles, hints and column labels.
    pub const CAPTION: Pixels = px(11.);
    /// The line height of reading text, shared by a timestamp beside it.
    pub const READING_LINE: Pixels = px(20.);
    /// Clocks and timestamps: fixed-width digits, so columns stay put.
    pub const MONO: &str = "Menlo";
    pub const STRONG: FontWeight = FontWeight::SEMIBOLD;
    /// The brand name on the empty window.
    pub const BRAND: FontWeight = FontWeight::BOLD;
    /// Markdown heading sizes, h1 to h6: a step over the reading text each,
    /// and never up to the page heading.
    pub const MARKDOWN_HEADINGS: [Pixels; 6] =
        [px(16.), px(15.), px(14.), px(14.), px(14.), px(14.)];
}

/// The two strengths of a tinted fill over a theme color.
pub mod tint {
    /// Hover, and the background of a tag.
    pub const SUBTLE: f32 = 0.08;
    /// A selected row without focus, and a pressed control.
    pub const STRONG: f32 = 0.16;
    /// Text and icons that sit back: disabled controls, empty meter bars.
    pub const FAINT: f32 = 0.4;
    /// The fill of a Settings card over the window background.
    pub const CARD: f32 = 0.025;
    /// The fill of a speaker's turn in the transcript.
    pub const BLOCK: f32 = 0.05;
}

/// The macOS accent colors, light then dark, as `AppleAccentColor` numbers
/// them. -1 is graphite; no value means the default blue.
const ACCENTS: [(i32, u32, u32); 8] = [
    (-1, 0x8E8E93, 0x98989D),
    (0, 0xFF3B30, 0xFF453A),
    (1, 0xFF9500, 0xFF9F0A),
    (2, 0xFFCC00, 0xFFD60A),
    (3, 0x28CD41, 0x32D74B),
    (4, 0x007AFF, 0x0A84FF),
    (5, 0xAF52DE, 0xBF5AF2),
    (6, 0xFF2D55, 0xFF375F),
];
const BLUE: i32 = 4;
/// The system red, light then dark: the color of recording, and of a
/// destructive confirm. The kit draws a danger button as a pale tint; a
/// record button must be solid.
const RED: (u32, u32) = (0xFF3B30, 0xFF453A);
/// The macOS dark palette, as AppKit resolves it under `darkAqua`. The kit's
/// own dark theme is near black with grey text; a Mac window is not.
mod dark {
    /// `windowBackgroundColor`, and the content of a pane.
    pub const WINDOW: u32 = 0x1E1E1E;
    /// `underPageBackgroundColor`: the sidebar, menus and dialogs, which sit
    /// a step above the content.
    pub const RAISED: u32 = 0x282828;
    /// White over the window, as the system labels, separators and fills
    /// take it.
    pub const LABEL: f32 = 0.85;
    pub const SECONDARY_LABEL: f32 = 0.55;
    pub const SEPARATOR: f32 = 0.12;
    /// The border of a text field: a step over a separator, so it shows.
    pub const FIELD: f32 = 0.18;
    /// `quaternarySystemFill` to `systemFill`, then a pressed fill.
    pub const FILL_FAINT: f32 = 0.05;
    pub const FILL: f32 = 0.08;
    pub const FILL_HOVER: f32 = 0.10;
    pub const FILL_PRESSED: f32 = 0.14;
    pub const SCROLL_THUMB: f32 = 0.30;
}

/// Paint the kit's surfaces, text and lines with the macOS dark palette.
fn apply_dark(theme: &mut Theme) {
    let window: Hsla = rgb(dark::WINDOW).into();
    let raised: Hsla = rgb(dark::RAISED).into();
    let line = white().opacity(dark::SEPARATOR);
    let label = white().opacity(dark::LABEL);
    let secondary_label = white().opacity(dark::SECONDARY_LABEL);
    let fill = white().opacity(dark::FILL);
    let hover = white().opacity(dark::FILL_HOVER);
    let pressed = white().opacity(dark::FILL_PRESSED);
    theme.background = window;
    theme.foreground = label;
    theme.muted = hover;
    theme.muted_foreground = secondary_label;
    theme.border = line;
    theme.input = white().opacity(dark::FIELD);
    theme.window_border = line;
    theme.title_bar = window;
    theme.title_bar_border = line;
    theme.status_bar = window;
    theme.status_bar_border = line;
    theme.sidebar = raised;
    theme.sidebar_border = line;
    theme.sidebar_foreground = label;
    theme.sidebar_accent = hover;
    theme.sidebar_accent_foreground = label;
    theme.popover = raised;
    theme.popover_foreground = label;
    theme.secondary = fill;
    theme.secondary_hover = hover;
    theme.secondary_active = pressed;
    theme.secondary_foreground = label;
    theme.button = fill;
    theme.button_hover = hover;
    theme.button_active = pressed;
    theme.button_foreground = label;
    theme.accent = hover;
    theme.accent_foreground = label;
    theme.colors.list = window;
    theme.list_even = window;
    theme.list_head = window;
    theme.list_hover = white().opacity(dark::FILL_FAINT);
    theme.group_box = white().opacity(dark::FILL_FAINT);
    theme.group_box_foreground = label;
    theme.scrollbar_thumb = white().opacity(dark::SCROLL_THUMB);
    theme.scrollbar_thumb_hover = white().opacity(dark::SECONDARY_LABEL);
}

/// Yellow is the one accent too light for white text.
const YELLOW: i32 = 2;

/// The accent number the user picked in System Settings.
fn accent_setting() -> i32 {
    Command::new("defaults")
        .args(["read", "-g", "AppleAccentColor"])
        .output()
        .ok()
        .and_then(|out| String::from_utf8(out.stdout).ok())
        .and_then(|text| text.trim().parse().ok())
        .unwrap_or(BLUE)
}

/// The accent color for `setting` in light or dark mode.
pub fn accent(setting: i32, dark: bool) -> Hsla {
    let (_, light, night) = ACCENTS
        .iter()
        .find(|(number, _, _)| *number == setting)
        .copied()
        .unwrap_or(ACCENTS[5]);
    rgb(if dark { night } else { light }).into()
}

/// Paint every accent surface of the kit with the macOS accent, and set the
/// radii and the body size. Run it after each appearance change: the kit
/// resets its colors then.
pub fn apply(theme: &mut Theme, dark: bool) {
    let setting = accent_setting();
    let color = accent(setting, dark);
    let on_accent: Hsla = if setting == YELLOW { black() } else { white() };
    let hover = color.opacity(0.9);
    let active = color.opacity(0.8);
    if dark {
        apply_dark(theme);
    }
    theme.font_size = text::BODY;
    theme.radius = radius::MD;
    theme.radius_lg = radius::LG;
    theme.primary = color;
    theme.primary_hover = hover;
    theme.primary_active = active;
    theme.primary_foreground = on_accent;
    theme.button_primary = color;
    theme.button_primary_hover = hover;
    theme.button_primary_active = active;
    theme.button_primary_foreground = on_accent;
    theme.ring = color;
    theme.caret = color;
    theme.link = color;
    theme.link_hover = hover;
    theme.link_active = active;
    theme.switch = color;
    theme.slider_bar = color;
    theme.progress_bar = color;
    theme.selection = color.opacity(0.25);
    theme.list_active = color.opacity(tint::STRONG);
    theme.list_active_border = color;
    theme.sidebar_primary = color;
    theme.sidebar_primary_foreground = on_accent;
    let red: Hsla = rgb(if dark { RED.1 } else { RED.0 }).into();
    theme.danger = red;
    theme.danger_hover = red.opacity(0.9);
    theme.danger_active = red.opacity(0.8);
    theme.danger_foreground = white();
    theme.button_danger = red;
    theme.button_danger_hover = red.opacity(0.9);
    theme.button_danger_active = red.opacity(0.8);
    theme.button_danger_foreground = white();
}

#[cfg(test)]
mod tests {
    use super::{BLUE, accent};

    #[test]
    fn an_unknown_accent_number_falls_back_to_blue() {
        assert_eq!(accent(42, false), accent(BLUE, false));
    }

    #[test]
    fn dark_mode_takes_the_lighter_accent() {
        assert!(accent(BLUE, true).l > accent(BLUE, false).l);
    }
}
