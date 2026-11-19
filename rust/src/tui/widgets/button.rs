//! Textual's `Button` in its default style: tall top and bottom edges, a bold
//! centered label, and a background by variant.

use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;

use crate::tui::text::{center, str_width};
use crate::tui::theme::Theme;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Variant {
    Default,
    Primary,
    Warning,
    Error,
}

impl Variant {
    /// The background, the top and bottom edge colors, and the label color.
    fn colors(self) -> (&'static str, &'static str, &'static str, &'static str) {
        match self {
            Self::Default => (
                "surface",
                "surface-lighten-1",
                "surface-darken-1",
                "button-foreground",
            ),
            Self::Primary => (
                "primary",
                "primary-lighten-3",
                "primary-darken-3",
                "button-color-foreground",
            ),
            Self::Warning => (
                "warning",
                "warning-lighten-2",
                "warning-darken-3",
                "button-color-foreground",
            ),
            Self::Error => (
                "error",
                "error-lighten-2",
                "error-darken-3",
                "button-color-foreground",
            ),
        }
    }
}

/// `width: auto; min-width: 16`, with `line-pad: 1` on each side of the label.
pub fn button_width(label: &str) -> u16 {
    (str_width(label) as u16 + 2).max(16)
}

/// Draw one 3-row button. A disabled button shows its label at 60% (`text-opacity: 0.6`).
pub fn draw_button(
    frame: &mut Frame,
    area: Rect,
    label: &str,
    variant: Variant,
    focused: bool,
    theme: &Theme,
) {
    draw(frame, area, label, variant, focused, false, theme);
}

pub fn draw_disabled_button(
    frame: &mut Frame,
    area: Rect,
    label: &str,
    variant: Variant,
    theme: &Theme,
) {
    draw(frame, area, label, variant, false, true, theme);
}

fn draw(
    frame: &mut Frame,
    area: Rect,
    label: &str,
    variant: Variant,
    focused: bool,
    disabled: bool,
    theme: &Theme,
) {
    let (background, top, bottom, text) = variant.colors();
    let base = match focused {
        // `:focus` tints the background 5% toward the foreground.
        true => theme.tinted(background, "foreground", 0.05),
        false => theme.color(background),
    };
    let edge = |variable: &str, glyph: &str| {
        Line::styled(
            glyph.repeat(usize::from(area.width)),
            Style::default().fg(theme.color(variable)).bg(base),
        )
    };
    let mut label_color = theme.on_color(text, base);
    if disabled {
        label_color = crate::tui::markdown::tint(label_color, base, 0.6);
    }
    let mut label_style = Style::default()
        .fg(label_color)
        .bg(base)
        .add_modifier(Modifier::BOLD);
    if focused {
        // `$button-focus-text-style` is `b reverse`.
        label_style = label_style.add_modifier(Modifier::REVERSED);
    }
    let middle = center(
        Line::from(vec![Span::styled(format!(" {label} "), label_style)]),
        area.width,
    )
    .style(Style::default().bg(base));
    let lines = vec![edge(top, "\u{2594}"), middle, edge(bottom, "\u{2581}")];
    frame.render_widget(Paragraph::new(lines).style(Style::default().bg(base)), area);
}
