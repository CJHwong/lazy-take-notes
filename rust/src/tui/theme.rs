//! Textual's built-in themes, as the color variables Textual generates from them.
//!
//! `config.yaml` stores a Textual theme name, so both versions must show the same
//! colors. `scripts/gen_themes.py` writes `themes.json` from Textual itself.

use std::collections::BTreeMap;
use std::sync::OnceLock;

use ratatui::style::Color;
use serde::Deserialize;

pub const DEFAULT_THEME: &str = "textual-dark";

#[derive(Deserialize)]
struct ThemeData {
    dark: bool,
    variables: BTreeMap<String, String>,
}

fn all_themes() -> &'static BTreeMap<String, ThemeData> {
    static THEMES: OnceLock<BTreeMap<String, ThemeData>> = OnceLock::new();
    THEMES.get_or_init(|| {
        serde_json::from_str(include_str!("themes.json")).expect("themes.json parses")
    })
}

/// The theme names, sorted.
pub fn theme_names() -> Vec<String> {
    all_themes().keys().cloned().collect()
}

/// An RGB color with Textual's alpha, before it is put on a background.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Rgba {
    r: f64,
    g: f64,
    b: f64,
    a: f64,
}

impl Rgba {
    fn opaque(r: u8, g: u8, b: u8) -> Self {
        Self {
            r: f64::from(r),
            g: f64::from(g),
            b: f64::from(b),
            a: 1.0,
        }
    }

    fn brightness(self) -> f64 {
        (299.0 * self.r + 587.0 * self.g + 114.0 * self.b) / 1000.0 / 255.0
    }

    /// `Color.get_contrast_text(alpha)`: off-white on dark, off-black on light.
    fn contrast_text(self, alpha: f64) -> Self {
        let level = if self.brightness() < 0.5 { 255 } else { 0 };
        Self {
            a: alpha,
            ..Self::opaque(level, level, level)
        }
    }

    /// `Color.blend`: `int()` truncation, as Textual does.
    fn blend(self, destination: Self, factor: f64) -> Self {
        let mix = |from: f64, to: f64| (from + (to - from) * factor).trunc();
        Self {
            r: mix(self.r, destination.r),
            g: mix(self.g, destination.g),
            b: mix(self.b, destination.b),
            a: 1.0,
        }
    }

    /// This color painted over an opaque background.
    fn over(self, background: Self) -> Self {
        if self.a >= 1.0 {
            return self;
        }
        background.blend(Self { a: 1.0, ..self }, self.a)
    }

    fn to_color(self) -> Color {
        Color::Rgb(self.r as u8, self.g as u8, self.b as u8)
    }
}

/// One parsed theme value.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Value {
    Rgba(Rgba),
    /// White or black by the background, at this alpha.
    Auto(f64),
    Ansi(Color),
}

fn hex_pair(text: &str, at: usize) -> Option<u8> {
    u8::from_str_radix(text.get(at..at + 2)?, 16).ok()
}

fn percent(text: &str) -> Option<f64> {
    Some(text.trim().strip_suffix('%')?.trim().parse::<f64>().ok()? / 100.0)
}

fn parse_value(text: &str) -> Option<Value> {
    if let Some(alpha) = text.strip_prefix("auto") {
        return Some(Value::Auto(percent(alpha).unwrap_or(1.0)));
    }
    if let Some(name) = text.strip_prefix("ansi_") {
        return Some(Value::Ansi(ansi(name)));
    }
    let (hex, alpha) = match text.split_once(' ') {
        Some((hex, alpha)) => (hex, percent(alpha)),
        None => (text, None),
    };
    let hex = hex.strip_prefix('#')?;
    if hex.len() != 6 && hex.len() != 8 {
        return None;
    }
    let mut color = Rgba::opaque(hex_pair(hex, 0)?, hex_pair(hex, 2)?, hex_pair(hex, 4)?);
    if hex.len() == 8 {
        color.a = f64::from(hex_pair(hex, 6)?) / 255.0;
    }
    if let Some(alpha) = alpha {
        color.a = alpha;
    }
    Some(Value::Rgba(color))
}

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

/// The colors of one theme.
pub struct Theme {
    pub name: String,
    pub dark: bool,
    data: &'static ThemeData,
}

impl Theme {
    /// The named theme, or Textual's default for an unknown name.
    pub fn load(name: &str) -> Self {
        let themes = all_themes();
        let (name, data) = match themes.get_key_value(name) {
            Some((key, data)) => (key.clone(), data),
            None => (DEFAULT_THEME.to_string(), &themes[DEFAULT_THEME]),
        };
        Self {
            name,
            dark: data.dark,
            data,
        }
    }

    fn value(&self, variable: &str) -> Option<Value> {
        parse_value(self.data.variables.get(variable)?)
    }

    fn opaque(&self, variable: &str, depth: u8) -> Rgba {
        let fallback = if self.dark {
            Rgba::opaque(0, 0, 0)
        } else {
            Rgba::opaque(255, 255, 255)
        };
        if depth > 3 {
            return fallback;
        }
        let background = || {
            if variable == "background" {
                fallback
            } else {
                self.opaque("background", depth + 1)
            }
        };
        match self.value(variable) {
            Some(Value::Rgba(color)) => color.over(background()),
            Some(Value::Auto(alpha)) => {
                let under = background();
                under.contrast_text(alpha).over(under)
            }
            _ => fallback,
        }
    }

    /// A theme variable as it shows on the screen background.
    pub fn color(&self, variable: &str) -> Color {
        self.on(variable, "background")
    }

    /// A theme variable as it shows over another variable's color, so `auto`
    /// text and colors with alpha resolve against what is really under them.
    pub fn on(&self, variable: &str, background: &str) -> Color {
        let under = self.opaque(background, 0);
        match self.value(variable) {
            Some(Value::Ansi(color)) => color,
            Some(Value::Rgba(color)) => color.over(under).to_color(),
            Some(Value::Auto(alpha)) => under.contrast_text(alpha).over(under).to_color(),
            None => Color::Reset,
        }
    }
}

impl Theme {
    /// `variable` as it shows over `under`, a color that is not a variable
    /// (a tinted background).
    pub fn on_color(&self, variable: &str, under: Color) -> Color {
        let Color::Rgb(r, g, b) = under else {
            return self.color(variable);
        };
        let under = Rgba::opaque(r, g, b);
        match self.value(variable) {
            Some(Value::Ansi(color)) => color,
            Some(Value::Rgba(color)) => color.over(under).to_color(),
            Some(Value::Auto(alpha)) => under.contrast_text(alpha).over(under).to_color(),
            None => Color::Reset,
        }
    }

    /// A text-style variable (`block-cursor-text-style`): `none`, or words like `b reverse`.
    pub fn text_style(&self, variable: &str) -> ratatui::style::Modifier {
        use ratatui::style::Modifier;
        let words = self
            .data
            .variables
            .get(variable)
            .map_or("", |value| value.as_str());
        words
            .split_whitespace()
            .fold(Modifier::empty(), |style, word| match word {
                "b" | "bold" => style | Modifier::BOLD,
                "i" | "italic" => style | Modifier::ITALIC,
                "u" | "underline" => style | Modifier::UNDERLINED,
                "r" | "reverse" => style | Modifier::REVERSED,
                "d" | "dim" => style | Modifier::DIM,
                "s" | "strike" => style | Modifier::CROSSED_OUT,
                _ => style,
            })
    }

    /// `variable` with `toward` painted over it at `alpha`, as Textual's `background-tint`.
    pub fn tinted(&self, variable: &str, toward: &str, alpha: f64) -> Color {
        let base = self.opaque(variable, 0);
        let tint = self.opaque(toward, 0);
        base.blend(tint, alpha).to_color()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_theme_has_the_core_variables() {
        for name in theme_names() {
            let theme = Theme::load(&name);
            for variable in [
                "primary",
                "secondary",
                "accent",
                "surface",
                "background",
                "text",
                "text-muted",
                "error",
            ] {
                assert!(theme.value(variable).is_some(), "{name} lacks {variable}");
            }
        }
        assert_eq!(theme_names().len(), 20);
    }

    #[test]
    fn auto_text_is_white_on_dark_and_black_on_light_at_its_alpha() {
        let frappe = Theme::load("catppuccin-frappe");
        // Textual: Color.parse('#303446').blend(contrast_text(0.87), 0.87).
        assert_eq!(frappe.color("text"), Color::Rgb(228, 228, 230));
        let latte = Theme::load("catppuccin-latte");
        let Color::Rgb(r, _, _) = latte.color("text") else {
            panic!()
        };
        assert!(r < 60, "{r}");
    }

    #[test]
    fn alpha_colors_blend_over_the_background_and_an_unknown_name_falls_back() {
        let frappe = Theme::load("catppuccin-frappe");
        // block-cursor-blurred-background is #CA9EE64C over #303446.
        let expected = Rgba::opaque(0x30, 0x34, 0x46)
            .blend(Rgba::opaque(0xCA, 0x9E, 0xE6), f64::from(0x4C_u8) / 255.0);
        assert_eq!(
            frappe.color("block-cursor-blurred-background"),
            expected.to_color()
        );
        assert_eq!(Theme::load("no-such-theme").name, DEFAULT_THEME);
        assert_eq!(Theme::load("textual-ansi").color("primary"), Color::Blue);
    }
}
