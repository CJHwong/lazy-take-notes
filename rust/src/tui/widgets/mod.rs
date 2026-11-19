//! Widgets shared by the screens.
//!
//! The look follows ratatui's idiom over the saved theme's colors: rounded
//! borders with the title in the border, the focused box's border in the
//! primary color, one-row buttons, and a `▶` marker on the selected row.

pub mod button;
pub mod input;
pub mod list;
pub mod scroll;
pub mod select;
pub mod status_bar;
pub mod text_area;

use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Borders, Paragraph};

use crate::tui::text::center;
use crate::tui::theme::Theme;

/// Textual's cursor blink: shown for 0.5 s, hidden for 0.5 s, shown again from
/// the start after each key.
#[derive(Debug)]
pub struct Blink(std::cell::Cell<std::time::Instant>);

impl Default for Blink {
    fn default() -> Self {
        Self(std::cell::Cell::new(std::time::Instant::now()))
    }
}

impl Blink {
    pub fn restart(&self) {
        self.0.set(std::time::Instant::now());
    }

    pub fn visible(&self) -> bool {
        (self.0.get().elapsed().as_millis() / 500).is_multiple_of(2)
    }
}

/// A `border: solid` box on `background`, its title at the top left. Returns the inside.
pub fn panel(
    frame: &mut Frame,
    area: Rect,
    title: &str,
    border: &str,
    background: Color,
    theme: &Theme,
) -> Rect {
    rounded(frame, area, title, theme.color(border), background, theme)
}

/// A rounded box on `background`, its title in the top border. Returns the inside.
pub fn rounded(
    frame: &mut Frame,
    area: Rect,
    title: &str,
    border: Color,
    background: Color,
    theme: &Theme,
) -> Rect {
    let mut block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .style(Style::default().bg(background))
        .border_style(Style::default().fg(border).bg(background));
    if !title.is_empty() {
        block = block.title(Line::from(vec![
            Span::styled("─", Style::default().fg(border)),
            Span::styled(
                format!(" {title} "),
                Style::default()
                    .fg(theme.color("foreground"))
                    .add_modifier(Modifier::BOLD),
            ),
        ]));
    }
    let inner = block.inner(area);
    frame.render_widget(block, area);
    inner
}

/// The border color of a box: primary while it has the focus.
pub fn border_color(focused: bool, theme: &Theme) -> Color {
    theme.color(if focused { "primary" } else { "border-blurred" })
}

/// A one-row bar: the header (`$primary`, bold, centered) or a footer.
pub fn bar(
    frame: &mut Frame,
    area: Rect,
    text: &str,
    background: &str,
    foreground: &str,
    bold: bool,
    theme: &Theme,
) {
    let mut style = Style::default()
        .bg(theme.color(background))
        .fg(theme.on(foreground, background));
    if bold {
        style = style.add_modifier(Modifier::BOLD);
    }
    frame.render_widget(Paragraph::new("").style(style), area);
    // `padding: 0 1`, then Rich's centering.
    let inner = Rect::new(
        area.x + 1,
        area.y,
        area.width.saturating_sub(2),
        area.height,
    );
    let line = center(Line::from(text.to_string()), inner.width);
    frame.render_widget(Paragraph::new(line).style(style), inner);
}

/// `area` minus `top` rows and `bottom` rows.
pub fn split_bars(area: Rect, top: u16, bottom: u16) -> (Rect, Rect, Rect) {
    let top = top.min(area.height);
    let bottom = bottom.min(area.height - top);
    let header = Rect::new(area.x, area.y, area.width, top);
    let body = Rect::new(area.x, area.y + top, area.width, area.height - top - bottom);
    let footer = Rect::new(area.x, area.bottom() - bottom, area.width, bottom);
    (header, body, footer)
}

/// A box of `width` by `height` in the middle of `area`.
pub fn centered(area: Rect, width: u16, height: u16) -> Rect {
    let width = width.min(area.width);
    let height = height.min(area.height);
    Rect::new(
        area.x + (area.width - width) / 2,
        area.y + (area.height - height) / 2,
        width,
        height,
    )
}

/// A dialog's frame: a rounded box in `border_var`, filled with
/// `background_var`, over whatever is below it. Returns the inside.
pub fn thick_box(
    frame: &mut Frame,
    area: Rect,
    border_var: &str,
    background_var: &str,
    theme: &Theme,
) -> Rect {
    if area.width < 2 || area.height < 2 {
        return Rect::default();
    }
    // A dialog covers what is under it; text styles there must not show through.
    frame.render_widget(ratatui::widgets::Clear, area);
    rounded(
        frame,
        area,
        "",
        theme.color(border_var),
        theme.color(background_var),
        theme,
    )
}

/// The styles of Textual's `tall` border: the left edge `▊`, the right edge `▎`,
/// and the top `▔` and bottom `▁` fills. The left edge is drawn reversed, so its
/// glyph shows the parent's background and only a thin line of border is left.
pub struct TallEdges {
    pub left: Style,
    pub right: Style,
    pub fill: Style,
}

pub fn tall_edges(border: Color, inner: Color, outer: Color) -> TallEdges {
    TallEdges {
        left: Style::default().fg(outer).bg(border),
        right: Style::default().fg(border).bg(outer),
        fill: Style::default().fg(border).bg(inner),
    }
}
