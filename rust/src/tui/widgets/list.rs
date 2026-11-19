//! A one-row-per-item list with Textual's `ListView` highlight.

use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::Line;
use ratatui::widgets::Paragraph;

use crate::tui::text::wrap_line;
use crate::tui::theme::Theme;

#[derive(Default)]
pub struct ListCursor {
    pub index: Option<usize>,
    offset: usize,
    len: usize,
    /// Rows shown at the last draw, for page keys.
    page: usize,
}

impl ListCursor {
    /// New items: the highlight moves to the first one, as a rebuilt `ListView` does.
    pub fn reset(&mut self, len: usize) {
        self.len = len;
        self.offset = 0;
        self.index = (len > 0).then_some(0);
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn select(&mut self, index: usize) {
        if self.len > 0 {
            self.index = Some(index.min(self.len - 1));
        }
    }

    /// Move by a list key. Returns true when the key was a list key.
    pub fn key(&mut self, name: &str) -> bool {
        let Some(index) = self.index else {
            return matches!(name, "up" | "down" | "home" | "end" | "pageup" | "pagedown");
        };
        let last = self.len.saturating_sub(1);
        let page = self.page.max(1);
        let next = match name {
            "up" => index.saturating_sub(1),
            "down" => (index + 1).min(last),
            "home" => 0,
            "end" => last,
            "pageup" => index.saturating_sub(page),
            "pagedown" => (index + page).min(last),
            _ => return false,
        };
        self.index = Some(next);
        true
    }

    /// Draw `rows`, keeping the highlight in view.
    pub fn render(
        &mut self,
        frame: &mut Frame,
        area: Rect,
        rows: &[Line<'_>],
        focused: bool,
        background: Color,
        theme: &Theme,
    ) {
        let height = usize::from(area.height);
        self.page = height;
        self.len = rows.len();
        if let Some(index) = self.index {
            if index < self.offset {
                self.offset = index;
            } else if height > 0 && index >= self.offset + height {
                self.offset = index + 1 - height;
            }
        }
        let plain = Style::default()
            .bg(background)
            .fg(theme.on("foreground", "surface"));
        let highlight = if focused {
            Style::default()
                .bg(theme.on("block-cursor-background", "surface"))
                .fg(theme.on("block-cursor-foreground", "block-cursor-background"))
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default()
                .bg(theme.on("block-cursor-blurred-background", "surface"))
                .fg(theme.on("block-cursor-blurred-foreground", "surface"))
        };
        frame.render_widget(Paragraph::new("").style(plain), area);
        let overflow = rows.len() > height;
        let row_width = area.width.saturating_sub(u16::from(overflow));
        if overflow {
            crate::tui::widgets::scroll::draw_scrollbar(
                frame,
                area,
                self.offset,
                rows.len(),
                height,
                theme,
            );
        }
        for (row, line) in rows.iter().skip(self.offset).take(height).enumerate() {
            let index = self.offset + row;
            let rect = Rect::new(area.x, area.y + row as u16, row_width, 1);
            let style = if Some(index) == self.index {
                highlight
            } else {
                plain
            };
            // A row is a one-row `Static`: its text wraps and only the first row shows.
            // Its own markup colors stay inside the highlight, as in Textual.
            let first = wrap_line(line, row_width)
                .into_iter()
                .next()
                .unwrap_or_default();
            frame.render_widget(Paragraph::new(first).style(style), rect);
        }
    }
}
