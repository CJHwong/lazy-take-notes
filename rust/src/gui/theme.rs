//! The window follows the macOS appearance, light or dark. The Textual theme
//! stays a terminal setting.

use gpui_kit::component::{ActiveTheme, Theme};
use gpui_kit::*;

use crate::app::theme::Rgb;

pub fn hsla(color: Rgb) -> Hsla {
    rgb((u32::from(color.0) << 16) | (u32::from(color.1) << 8) | u32::from(color.2)).into()
}

/// Show the system appearance now, and again each time it changes. Keep the
/// returned subscription: the window stops following when it drops.
pub fn follow_system(window: &mut Window, cx: &mut App) -> Subscription {
    sync(window, cx);
    window.observe_window_appearance(sync)
}

/// macOS body text is 13 pt; the kit starts at 16.
const FONT_SIZE: Pixels = px(13.);

fn sync(window: &mut Window, cx: &mut App) {
    Theme::sync_system_appearance(Some(window), cx);
    Theme::update(cx, |theme| theme.font_size = FONT_SIZE);
}

/// The system color for a Textual variable name. The views from the terminal
/// app still name their colors this way.
pub fn color(cx: &App, variable: &str) -> Hsla {
    let theme = cx.theme();
    match variable {
        "background" => theme.background,
        "text-muted" | "footer-description-foreground" => theme.muted_foreground,
        "primary" | "border" => theme.primary,
        "surface" | "footer-background" | "footer-key-background" => theme.muted,
        "border-blurred" => theme.border,
        "block-cursor-background" => theme.list_active,
        "block-cursor-blurred-background" | "block-hover-background" => theme.list_hover,
        "error" => theme.danger,
        _ => theme.foreground,
    }
}

/// A text color on a background, by the same names.
pub fn on(cx: &App, variable: &str, background: &str) -> Hsla {
    let theme = cx.theme();
    match (variable, background) {
        ("text", "primary") => theme.primary_foreground,
        ("text", "error") => theme.danger_foreground,
        _ => color(cx, variable),
    }
}

/// A background, always painted.
pub fn fill(cx: &App, variable: &str) -> Option<Hsla> {
    Some(color(cx, variable))
}

/// `variable` with a little of `toward` over it.
pub fn tinted(cx: &App, variable: &str, toward: &str, alpha: f64) -> Hsla {
    let base = color(cx, variable);
    base.blend(color(cx, toward).opacity(alpha as f32))
}
