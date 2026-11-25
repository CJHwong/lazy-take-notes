//! Textual's built-in themes, as the color variables Textual generates from them.
//!
//! `config.yaml` stores a Textual theme name, so every front end must show the
//! same colors. `scripts/gen_themes.py` writes `themes.json` from Textual itself.
//! This module resolves a variable to an opaque RGB color; each front end turns
//! that into its own color type.

use std::collections::BTreeMap;
use std::sync::OnceLock;

use serde::Deserialize;

use crate::adapters::config_file::load_raw;
use crate::adapters::paths::Paths;

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

/// The theme in the default config file. `--config` does not change it, as in Python.
pub fn saved_theme() -> String {
    let Some(paths) = Paths::for_user() else {
        return DEFAULT_THEME.into();
    };
    load_raw(&paths, None, None)
        .ok()
        .and_then(|raw| raw.get("theme")?.as_str().map(str::to_string))
        .unwrap_or_else(|| DEFAULT_THEME.into())
}

/// An opaque screen color.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rgb(pub u8, pub u8, pub u8);

/// What a variable paints.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Paint {
    Rgb(Rgb),
    /// A terminal palette color by name (`blue`, `bright_red`), from the
    /// `textual-ansi` theme. The terminal picks the real color.
    Ansi(&'static str),
    /// The theme does not have this variable.
    Missing,
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

    fn rgb(self) -> Rgb {
        Rgb(self.r as u8, self.g as u8, self.b as u8)
    }
}

/// One parsed theme value.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Value {
    Rgba(Rgba),
    /// White or black by the background, at this alpha.
    Auto(f64),
    Ansi(&'static str),
}

const ANSI_NAMES: [&str; 16] = [
    "black",
    "red",
    "green",
    "yellow",
    "blue",
    "magenta",
    "cyan",
    "white",
    "bright_black",
    "bright_red",
    "bright_green",
    "bright_yellow",
    "bright_blue",
    "bright_magenta",
    "bright_cyan",
    "bright_white",
];

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
        // An unknown name keeps the terminal's default color, as before.
        let known = ANSI_NAMES.iter().find(|known| **known == name);
        return Some(Value::Ansi(known.copied().unwrap_or("default")));
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

/// The colors of one theme.
pub struct Palette {
    pub name: String,
    pub dark: bool,
    data: &'static ThemeData,
}

impl Palette {
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

    /// The theme has this variable, and it parses.
    pub fn has(&self, variable: &str) -> bool {
        self.value(variable).is_some()
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

    /// A variable as an opaque color on the screen background. An ANSI
    /// variable gives the background's fallback.
    pub fn rgb(&self, variable: &str) -> Rgb {
        self.opaque(variable, 0).rgb()
    }

    /// A variable as it shows over another variable's color, so `auto` text
    /// and colors with alpha resolve against what is really under them.
    pub fn on(&self, variable: &str, background: &str) -> Paint {
        self.over(variable, self.opaque(background, 0))
    }

    /// A variable as it shows over `under`, a color that is not a variable
    /// (a tinted background).
    pub fn on_rgb(&self, variable: &str, under: Rgb) -> Paint {
        self.over(variable, Rgba::opaque(under.0, under.1, under.2))
    }

    fn over(&self, variable: &str, under: Rgba) -> Paint {
        match self.value(variable) {
            Some(Value::Ansi(name)) => Paint::Ansi(name),
            Some(Value::Rgba(color)) => Paint::Rgb(color.over(under).rgb()),
            Some(Value::Auto(alpha)) => Paint::Rgb(under.contrast_text(alpha).over(under).rgb()),
            None => Paint::Missing,
        }
    }

    /// A text-style variable (`block-cursor-text-style`): `none`, or words like `b reverse`.
    pub fn text_style(&self, variable: &str) -> &'static str {
        self.data
            .variables
            .get(variable)
            .map_or("", |value| value.as_str())
    }

    /// `variable` with `toward` painted over it at `alpha`, as Textual's `background-tint`.
    pub fn tinted(&self, variable: &str, toward: &str, alpha: f64) -> Rgb {
        let base = self.opaque(variable, 0);
        let tint = self.opaque(toward, 0);
        base.blend(tint, alpha).rgb()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_theme_has_the_core_variables() {
        for name in theme_names() {
            let palette = Palette::load(&name);
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
                assert!(palette.has(variable), "{name} lacks {variable}");
            }
        }
        assert_eq!(theme_names().len(), 20);
    }

    #[test]
    fn auto_text_is_white_on_dark_and_black_on_light_at_its_alpha() {
        let frappe = Palette::load("catppuccin-frappe");
        // Textual: Color.parse('#303446').blend(contrast_text(0.87), 0.87).
        assert_eq!(
            frappe.on("text", "background"),
            Paint::Rgb(Rgb(228, 228, 230))
        );
        let Paint::Rgb(Rgb(r, _, _)) = Palette::load("catppuccin-latte").on("text", "background")
        else {
            panic!()
        };
        assert!(r < 60, "{r}");
    }

    #[test]
    fn alpha_colors_blend_over_the_background_and_an_unknown_name_falls_back() {
        let frappe = Palette::load("catppuccin-frappe");
        // block-cursor-blurred-background is #CA9EE64C over #303446.
        let expected = Rgba::opaque(0x30, 0x34, 0x46)
            .blend(Rgba::opaque(0xCA, 0x9E, 0xE6), f64::from(0x4C_u8) / 255.0);
        assert_eq!(
            frappe.on("block-cursor-blurred-background", "background"),
            Paint::Rgb(expected.rgb())
        );
        assert_eq!(Palette::load("no-such-theme").name, DEFAULT_THEME);
        assert_eq!(
            Palette::load("textual-ansi").on("primary", "background"),
            Paint::Ansi("blue")
        );
    }
}
