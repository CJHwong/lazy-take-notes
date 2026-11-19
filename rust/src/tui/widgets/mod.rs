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

/// The border color of a box: primary while it has the focus, else a quiet line.
pub fn border_color(focused: bool, theme: &Theme) -> Color {
    match focused {
        true => theme.color("primary"),
        false => quiet_line(theme),
    }
}

/// A line that shows on the background without drawing the eye: the
/// background moved 30% toward the foreground. Themes' own `border-blurred`
/// is too close to the background in most dark themes.
pub fn quiet_line(theme: &Theme) -> Color {
    theme.tinted("background", "foreground", 0.3)
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
    dialog_box(frame, area, "", border_var, background_var, theme)
}

/// A dialog's frame with `title` in its top border.
pub fn dialog_box(
    frame: &mut Frame,
    area: Rect,
    title: &str,
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
        title,
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

/// Mark each `[key] label` hint drawn in `area` as a click target named
/// `key:<key>`, from the cells already drawn there. A click on one presses the key.
pub fn hint_targets(frame: &mut Frame, area: Rect, ctx: &crate::tui::host::Ctx) {
    let buffer = frame.buffer_mut();
    for y in area.top()..area.bottom() {
        let cells: Vec<(u16, String)> = (area.left()..area.right())
            .filter_map(|x| buffer.cell((x, y)).map(|cell| (x, cell.symbol().to_string())))
            .collect();
        for (start, end, key) in hints_in_row(&cells) {
            ctx.hit(Rect::new(start, y, end - start, 1), &format!("key:{key}"), 0);
        }
    }
}

/// The `[key] label` runs in one row of cells: start column, end column, key.
/// A label ends at two spaces or at the next `[`.
fn hints_in_row(cells: &[(u16, String)]) -> Vec<(u16, u16, String)> {
    let mut found = Vec::new();
    let mut index = 0;
    while index < cells.len() {
        if cells[index].1 != "[" {
            index += 1;
            continue;
        }
        let Some(close) = (index + 1..cells.len()).find(|&at| cells[at].1 == "]") else {
            break;
        };
        let key: String = cells[index + 1..close].iter().map(|(_, symbol)| symbol.as_str()).collect();
        let mut end = close + 1;
        while end < cells.len() && cells[end].1 != "[" {
            let double_space = cells[end].1 == " " && cells.get(end + 1).is_some_and(|cell| cell.1 == " ");
            if double_space && end > close + 1 {
                break;
            }
            end += 1;
        }
        // Trailing spaces are not part of the label.
        while end > close + 1 && cells[end - 1].1 == " " {
            end -= 1;
        }
        if crate::tui::keys::hint_key(&key).is_some() {
            let last = cells.get(end).map_or(cells[end - 1].0 + 1, |cell| cell.0);
            found.push((cells[index].0, last, key));
        }
        index = end.max(close + 1);
    }
    found
}

#[cfg(test)]
mod hint_tests {
    use super::*;

    fn cells(text: &str) -> Vec<(u16, String)> {
        text.chars().enumerate().map(|(x, c)| (x as u16, c.to_string())).collect()
    }

    #[test]
    fn each_key_hint_becomes_a_target_and_arrow_hints_do_not() {
        let found = hints_in_row(&cells(" [Enter] Select  [↑/↓] Navigate  [Esc] Cancel"));
        assert_eq!(
            found,
            vec![(1, 15, "Enter".to_string()), (33, 45, "Esc".to_string())]
        );
        let found = hints_in_row(&cells("[Ctrl+S] Save"));
        assert_eq!(found, vec![(0, 13, "Ctrl+S".to_string())]);
    }
}
