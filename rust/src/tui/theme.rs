//! The session theme as ratatui colors. The colors come from `app::theme`,
//! so the terminal and the window show the same theme.

use ratatui::style::Color;

pub use crate::app::theme::{DEFAULT_THEME, theme_names};
use crate::app::theme::{Paint, Palette, Rgb};

fn ansi(name: &str) -> Color {
    match name {
        "black" => Color::Black,
        "red" => Color::Red,
        "green" => Color::Green,
        "yellow" => Color::Yellow,
        "blue" => Color::Blue,
        "magenta" => Color::Magenta,
        "cyan" => Color::Cyan,
        "white" => Color::Gray,
        "bright_black" => Color::DarkGray,
        "bright_red" => Color::LightRed,
        "bright_green" => Color::LightGreen,
        "bright_yellow" => Color::LightYellow,
        "bright_blue" => Color::LightBlue,
        "bright_magenta" => Color::LightMagenta,
        "bright_cyan" => Color::LightCyan,
        "bright_white" => Color::White,
        _ => Color::Reset,
    }
}

fn color(paint: Paint) -> Color {
    match paint {
        Paint::Rgb(Rgb(r, g, b)) => Color::Rgb(r, g, b),
        Paint::Ansi(name) => ansi(name),
        Paint::Missing => Color::Reset,
    }
}

/// The colors of one theme.
pub struct Theme {
    pub name: String,
    pub dark: bool,
    palette: Palette,
}

impl Theme {
    /// The named theme, or Textual's default for an unknown name.
    pub fn load(name: &str) -> Self {
        let palette = Palette::load(name);
        Self {
            name: palette.name.clone(),
            dark: palette.dark,
            palette,
        }
    }

    /// A theme variable as it shows on the screen background.
    pub fn color(&self, variable: &str) -> Color {
        self.on(variable, "background")
    }

    /// A theme variable as it shows over another variable's color, so `auto`
    /// text and colors with alpha resolve against what is really under them.
    pub fn on(&self, variable: &str, background: &str) -> Color {
        color(self.palette.on(variable, background))
    }

    /// `variable` as it shows over `under`, a color that is not a variable
    /// (a tinted background).
    pub fn on_color(&self, variable: &str, under: Color) -> Color {
        let Color::Rgb(r, g, b) = under else {
            return self.color(variable);
        };
        color(self.palette.on_rgb(variable, Rgb(r, g, b)))
    }

    /// A text-style variable (`block-cursor-text-style`): `none`, or words like `b reverse`.
    pub fn text_style(&self, variable: &str) -> ratatui::style::Modifier {
        use ratatui::style::Modifier;
        self.palette.text_style(variable).split_whitespace().fold(
            Modifier::empty(),
            |style, word| match word {
                "b" | "bold" => style | Modifier::BOLD,
                "i" | "italic" => style | Modifier::ITALIC,
                "u" | "underline" => style | Modifier::UNDERLINED,
                "r" | "reverse" => style | Modifier::REVERSED,
                "d" | "dim" => style | Modifier::DIM,
                "s" | "strike" => style | Modifier::CROSSED_OUT,
                _ => style,
            },
        )
    }

    /// `variable` with `toward` painted over it at `alpha`, as Textual's `background-tint`.
    pub fn tinted(&self, variable: &str, toward: &str, alpha: f64) -> Color {
        let Rgb(r, g, b) = self.palette.tinted(variable, toward, alpha);
        Color::Rgb(r, g, b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_ansi_theme_uses_the_terminal_palette() {
        assert_eq!(Theme::load("textual-ansi").color("primary"), Color::Blue);
        assert_eq!(
            Theme::load("catppuccin-frappe").color("text"),
            Color::Rgb(228, 228, 230)
        );
    }
}
