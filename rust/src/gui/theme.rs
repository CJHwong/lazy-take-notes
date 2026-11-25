//! The saved Textual theme, applied to GPUI Kit. The kit's widgets get the
//! theme's colors, and views ask for any Textual variable by name, so the
//! window shows the same theme as the terminal.

use gpui_kit::component::{Theme, ThemeMode};
use gpui_kit::*;

use crate::app::theme::{Paint, Palette, Rgb};

/// The theme the window shows now.
pub struct GuiTheme(pub Palette);

impl Global for GuiTheme {}

/// The terminal colors xterm uses, for the `textual-ansi` theme.
fn ansi(name: &str) -> Rgb {
    match name {
        "black" => Rgb(0, 0, 0),
        "red" => Rgb(205, 0, 0),
        "green" => Rgb(0, 205, 0),
        "yellow" => Rgb(205, 205, 0),
        "blue" => Rgb(0, 0, 238),
        "magenta" => Rgb(205, 0, 205),
        "cyan" => Rgb(0, 205, 205),
        "white" => Rgb(229, 229, 229),
        "bright_black" => Rgb(127, 127, 127),
        "bright_red" => Rgb(255, 0, 0),
        "bright_green" => Rgb(0, 255, 0),
        "bright_yellow" => Rgb(255, 255, 0),
        "bright_blue" => Rgb(92, 92, 255),
        "bright_magenta" => Rgb(255, 0, 255),
        "bright_cyan" => Rgb(0, 255, 255),
        _ => Rgb(255, 255, 255),
    }
}

pub fn hsla(color: Rgb) -> Hsla {
    rgb((u32::from(color.0) << 16) | (u32::from(color.1) << 8) | u32::from(color.2)).into()
}

fn paint(palette: &Palette, paint: Paint) -> Hsla {
    match paint {
        Paint::Rgb(color) => hsla(color),
        Paint::Ansi(name) => hsla(ansi(name)),
        Paint::Missing => hsla(palette.rgb("foreground")),
    }
}

/// A Textual variable as it shows on the screen background.
pub fn color(cx: &App, variable: &str) -> Hsla {
    on(cx, variable, "background")
}

/// A Textual variable as it shows over another variable's color.
pub fn on(cx: &App, variable: &str, background: &str) -> Hsla {
    let palette = &cx.global::<GuiTheme>().0;
    paint(palette, palette.on(variable, background))
}

/// A background variable, or None when the theme leaves it `transparent`, so
/// the caller paints no background at all.
pub fn fill(cx: &App, variable: &str) -> Option<Hsla> {
    let palette = &cx.global::<GuiTheme>().0;
    match palette.on(variable, "background") {
        Paint::Missing => None,
        found => Some(paint(palette, found)),
    }
}

/// `variable` with `toward` over it at `alpha`, as Textual's `background-tint`.
pub fn tinted(cx: &App, variable: &str, toward: &str, alpha: f64) -> Hsla {
    hsla(cx.global::<GuiTheme>().0.tinted(variable, toward, alpha))
}

/// Show `name` (a Textual theme, or the default for an unknown one).
pub fn apply(name: &str, cx: &mut App) {
    let palette = Palette::load(name);
    let mode = if palette.dark {
        ThemeMode::Dark
    } else {
        ThemeMode::Light
    };
    // A mode change reloads the kit's colors, so the edits come after it.
    Theme::change(mode, None, cx);
    let c = |variable: &str| paint(&palette, palette.on(variable, "background"));
    let on = |variable: &str, under: &str| paint(&palette, palette.on(variable, under));
    Theme::update(cx, |theme| {
        theme.background = c("background");
        theme.foreground = c("foreground");
        theme.border = c("border-blurred");
        theme.input = c("border-blurred");
        theme.ring = c("border");
        theme.caret = c("input-cursor-background");
        theme.selection = c("input-selection-background");
        theme.muted = c("surface");
        theme.muted_foreground = c("text-muted");
        theme.popover = c("surface");
        theme.popover_foreground = on("foreground", "surface");
        theme.primary = c("primary");
        theme.primary_hover = c("primary-lighten-1");
        theme.primary_active = c("primary-darken-1");
        theme.primary_foreground = on("text", "primary");
        theme.secondary = c("panel");
        theme.secondary_hover = c("boost");
        theme.secondary_active = c("surface-active");
        theme.secondary_foreground = on("foreground", "panel");
        theme.accent = c("block-hover-background");
        theme.accent_foreground = c("foreground");
        theme.danger = c("error");
        theme.danger_hover = c("error-lighten-1");
        theme.danger_active = c("error-darken-1");
        theme.danger_foreground = on("text", "error");
        theme.warning = c("warning");
        theme.warning_hover = c("warning-lighten-1");
        theme.warning_active = c("warning-darken-1");
        theme.warning_foreground = on("text", "warning");
        theme.success = c("success");
        theme.success_hover = c("success-lighten-1");
        theme.success_active = c("success-darken-1");
        theme.success_foreground = on("text", "success");
        theme.info = c("accent");
        theme.info_foreground = on("text", "accent");
        theme.link = c("link-color");
        theme.link_hover = c("link-color-hover");
        theme.list_hover = c("block-hover-background");
        theme.list_active = c("block-cursor-blurred-background");
        theme.list_active_border = c("primary");
        theme.scrollbar = c("scrollbar-background");
        theme.scrollbar_thumb = c("scrollbar");
        theme.scrollbar_thumb_hover = c("scrollbar-hover");
        theme.status_bar = c("footer-background");
        theme.title_bar = c("panel");
        theme.title_bar_border = c("panel");
        theme.tab_bar = c("panel");
        theme.tab = c("panel");
        theme.tab_active = c("surface");
        theme.tab_foreground = c("text-muted");
        theme.tab_active_foreground = c("foreground");
        theme.sidebar = c("panel");
        theme.sidebar_foreground = on("foreground", "panel");
        theme.sidebar_border = c("border-blurred");
        theme.sidebar_accent = c("block-hover-background");
        theme.sidebar_accent_foreground = c("foreground");
        theme.sidebar_primary = c("primary");
        theme.sidebar_primary_foreground = on("text", "primary");
        theme.table_head = c("panel");
        theme.table_head_foreground = on("foreground", "panel");
        theme.progress_bar = c("primary");
        theme.window_border = c("border-blurred");
        theme.switch = c("panel");
        // Buttons, as Textual's: `surface` by default, the variant color otherwise.
        theme.button = c("surface");
        theme.button_hover = c("boost");
        theme.button_active = c("surface-active");
        theme.button_foreground = on("foreground", "surface");
        let colors = std::ops::DerefMut::deref_mut(theme);
        for (variable, fill, hover, active, text) in [
            (
                "primary",
                &mut colors.button_primary,
                &mut colors.button_primary_hover,
                &mut colors.button_primary_active,
                &mut colors.button_primary_foreground,
            ),
            (
                "secondary",
                &mut colors.button_secondary,
                &mut colors.button_secondary_hover,
                &mut colors.button_secondary_active,
                &mut colors.button_secondary_foreground,
            ),
            (
                "warning",
                &mut colors.button_warning,
                &mut colors.button_warning_hover,
                &mut colors.button_warning_active,
                &mut colors.button_warning_foreground,
            ),
            (
                "error",
                &mut colors.button_danger,
                &mut colors.button_danger_hover,
                &mut colors.button_danger_active,
                &mut colors.button_danger_foreground,
            ),
            (
                "success",
                &mut colors.button_success,
                &mut colors.button_success_hover,
                &mut colors.button_success_active,
                &mut colors.button_success_foreground,
            ),
            (
                "accent",
                &mut colors.button_info,
                &mut colors.button_info_hover,
                &mut colors.button_info_active,
                &mut colors.button_info_foreground,
            ),
        ] {
            *fill = c(variable);
            *hover = c(&format!("{variable}-lighten-1"));
            *active = c(&format!("{variable}-darken-1"));
            *text = on("text", variable);
        }
    });
    cx.set_global(GuiTheme(palette));
    cx.refresh_windows();
}
