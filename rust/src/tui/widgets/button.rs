//! A one-row button: plain colored text, or `▌ Label ▐` in its variant's
//! color while focused.

use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;

use crate::tui::text::str_width;
use crate::tui::theme::Theme;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Variant {
    Default,
    Primary,
    Warning,
    Error,
}

impl Variant {
    /// The variant's theme color.
    fn color(self) -> &'static str {
        match self {
            Self::Default | Self::Primary => "primary",
            Self::Warning => "warning",
            Self::Error => "error",
        }
    }
}

/// The label and the two edge cells on each side.
pub fn button_width(label: &str) -> u16 {
    str_width(label.trim()) as u16 + 4
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
    if area.height == 0 || area.width == 0 {
        return;
    }
    // The middle row of the area, so a caller's taller slot still lines up.
    let row = Rect::new(area.x, area.y + (area.height - 1) / 2, area.width, 1);
    let background = frame
        .buffer_mut()
        .cell((row.x, row.y))
        .map_or(theme.color("background"), |cell| cell.bg);
    let background = if background == Color::Reset {
        theme.color("background")
    } else {
        background
    };
    let accent = theme.color(variant.color());
    let line = button_line(label.trim(), variant, focused, disabled, accent, background, theme);
    let pad = row.width.saturating_sub(str_width(&line_text(&line)) as u16) / 2;
    let rect = Rect::new(row.x + pad, row.y, row.width - pad, 1);
    frame.render_widget(Paragraph::new(line), rect);
}

fn button_line(
    label: &str,
    variant: Variant,
    focused: bool,
    disabled: bool,
    accent: Color,
    background: Color,
    theme: &Theme,
) -> Line<'static> {
    if focused && !disabled {
        let face = Style::default()
            .bg(accent)
            .fg(theme.on_color("button-color-foreground", accent))
            .add_modifier(Modifier::BOLD);
        let edge = Style::default().fg(accent).bg(background);
        return Line::from(vec![
            Span::styled("▌", edge),
            Span::styled(format!(" {label} "), face),
            Span::styled("▐", edge),
        ]);
    }
    let color = match (disabled, variant) {
        (true, _) => theme.on_color("foreground-disabled", background),
        (false, Variant::Default) => theme.on_color("foreground", background),
        (false, _) => accent,
    };
    let mut style = Style::default().fg(color).bg(background);
    if !disabled {
        style = style.add_modifier(Modifier::BOLD);
    }
    Line::from(Span::styled(format!("  {label}  "), style))
}

fn line_text(line: &Line<'_>) -> String {
    line.spans.iter().map(|span| span.content.as_ref()).collect()
}
