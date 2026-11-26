//! The window follows the macOS appearance, light or dark. The Textual theme
//! stays a terminal setting.

use gpui_kit::component::Theme;
use gpui_kit::*;

use crate::app::theme::Rgb;
use crate::gui::widgets::TITLE_BAR_HEIGHT;

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
    Theme::update(cx, |theme| {
        theme.font_size = FONT_SIZE;
        // Toasts start below the title bar row, where the toolbar is.
        theme.notification.margins.top = TITLE_BAR_HEIGHT + px(8.);
    });
}
