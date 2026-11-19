//! A one-row-per-item list: the selected row carries a `▶` marker and the
//! highlight colors.

use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;

use crate::tui::text::wrap_line;
use crate::tui::theme::Theme;

#[derive(Default)]
pub struct ListCursor {
    pub index: Option<usize>,
    offset: usize,
    len: usize,
    /// Rows that cannot take the highlight (a group header).
    disabled: Vec<bool>,
    /// Scroll to the highlight at the next draw. A scroll key clears it.
    follow: bool,
    /// Items show all their wrapped rows (`height: auto`), not only the first.
    pub wrap: bool,
    /// Rows drawn at the last render; the offset counts rows.
    total_rows: usize,
    /// Rows shown at the last draw, for page keys.
    page: usize,
    /// The first row of each item at the last draw, and where the rows sat.
    starts: Vec<usize>,
    area: Rect,
}

/// Columns the `▶ ` marker takes before each row.
const MARKER: u16 = 2;

impl ListCursor {
    /// New items: the highlight moves to the first one, as a rebuilt `ListView` does.
    pub fn reset(&mut self, len: usize) {
        self.reset_with(vec![false; len]);
    }

    /// New items, some disabled: the highlight moves to the first enabled one.
    pub fn reset_with(&mut self, disabled: Vec<bool>) {
        self.len = disabled.len();
        self.disabled = disabled;
        self.offset = 0;
        self.index = self.first_enabled();
        self.follow = true;
    }

    /// New items with no highlight, as a `ListView` after `clear()` and `append()`.
    pub fn reset_unselected(&mut self, len: usize) {
        self.reset(len);
        self.index = None;
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    fn enabled(&self, index: usize) -> bool {
        !self.disabled.get(index).copied().unwrap_or(false)
    }

    pub fn first_enabled(&self) -> Option<usize> {
        (0..self.len).find(|index| self.enabled(*index))
    }

    pub fn select(&mut self, index: usize) {
        if self.len > 0 {
            self.index = Some(index.min(self.len - 1));
            self.follow = true;
        }
    }

    /// A `ListView` key. Up and down move the highlight past disabled rows;
    /// home, end and the page keys scroll the view and leave the highlight.
    /// Returns true when the key was a list key.
    pub fn key(&mut self, name: &str) -> bool {
        let page = self.page.max(1);
        let last_offset = self.total_rows.max(self.len).saturating_sub(page);
        match name {
            "up" | "down" => self.step(name == "down"),
            "home" => self.offset = 0,
            "end" => self.offset = last_offset,
            "pageup" => self.offset = self.offset.saturating_sub(page),
            "pagedown" => self.offset = (self.offset + page).min(last_offset),
            _ => return false,
        }
        if !matches!(name, "up" | "down") {
            self.follow = false;
        }
        true
    }

    /// Textual's `action_cursor_down` and `action_cursor_up`.
    fn step(&mut self, down: bool) {
        self.follow = true;
        let Some(index) = self.index else {
            if self.len > 0 {
                self.index = Some(if down { 0 } else { self.len - 1 });
            }
            return;
        };
        let next = if down {
            (index + 1..self.len).find(|at| self.enabled(*at))
        } else {
            (0..index).rev().find(|at| self.enabled(*at))
        };
        if let Some(next) = next {
            self.index = Some(next);
        }
    }

    /// Each item's lines: all its wrapped rows, or only the first in one-row mode.
    fn item_lines(&self, rows: &[Line<'_>], width: u16) -> Vec<Vec<Line<'static>>> {
        rows.iter()
            .map(|line| {
                let mut wrapped = wrap_line(line, width);
                if wrapped.is_empty() {
                    wrapped.push(Line::default());
                }
                if !self.wrap {
                    wrapped.truncate(1);
                }
                wrapped
            })
            .collect()
    }

    /// The item drawn at screen row `y` at the last draw.
    pub fn item_at(&self, y: u16) -> Option<usize> {
        if y < self.area.y || y >= self.area.bottom() {
            return None;
        }
        let row = self.offset + usize::from(y - self.area.y);
        let item = self.starts.windows(2).position(|pair| row >= pair[0] && row < pair[1])?;
        (!self.disabled.get(item).copied().unwrap_or(false)).then_some(item)
    }

    /// Each screen row of the last draw with the item it shows.
    pub fn drawn_rows(&self) -> Vec<(Rect, usize)> {
        (self.area.top()..self.area.bottom())
            .filter_map(|y| {
                let item = self.item_at(y)?;
                Some((Rect::new(self.area.x, y, self.area.width, 1), item))
            })
            .collect()
    }

    /// A click at screen row `y`: highlight the item there. Returns it, and
    /// whether it already had the highlight (a second click opens it).
    pub fn click(&mut self, y: u16) -> Option<(usize, bool)> {
        let item = self.item_at(y)?;
        let again = self.index == Some(item);
        self.select(item);
        Some((item, again))
    }

    /// Scroll by `rows` without moving the highlight, as the wheel does.
    pub fn scroll_by(&mut self, rows: isize) {
        self.follow = false;
        let last = self.total_rows.saturating_sub(self.page);
        self.offset = self.offset.saturating_add_signed(rows).min(last);
    }

    /// Scroll the least that shows the highlighted item, top first if it is taller.
    fn follow_highlight(&mut self, starts: &[usize], height: usize) {
        let Some(index) = self.index.filter(|_| self.follow) else {
            return;
        };
        let (top, bottom) = (starts[index], starts[index + 1]);
        if top < self.offset || bottom - top > height {
            self.offset = top;
        } else if bottom > self.offset + height {
            self.offset = bottom - height;
        }
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
        // The scrollbar takes a column only when the rows overflow.
        self.area = area;
        let text_width = area.width.saturating_sub(MARKER);
        let mut items = self.item_lines(rows, text_width);
        let mut total: usize = items.iter().map(Vec::len).sum();
        let overflow = total > height;
        let row_width = area.width.saturating_sub(u16::from(overflow));
        if overflow && self.wrap {
            items = self.item_lines(rows, text_width.saturating_sub(1));
            total = items.iter().map(Vec::len).sum();
        }
        self.total_rows = total;
        let mut starts = vec![0usize];
        for item in &items {
            starts.push(starts.last().copied().unwrap_or(0) + item.len());
        }
        self.offset = self.offset.min(total.saturating_sub(height));
        self.follow_highlight(&starts, height);
        self.starts = starts.clone();
        let plain = Style::default()
            .bg(background)
            .fg(theme.on("foreground", "surface"));
        let highlight = if focused {
            Style::default()
                .bg(theme.on("block-cursor-background", "surface"))
                .fg(theme.on("block-cursor-foreground", "block-cursor-background"))
                .add_modifier(theme.text_style("block-cursor-text-style"))
        } else {
            Style::default()
                .bg(theme.on("block-cursor-blurred-background", "surface"))
                .fg(theme.on("block-cursor-blurred-foreground", "surface"))
                .add_modifier(theme.text_style("block-cursor-blurred-text-style"))
        };
        frame.render_widget(Paragraph::new("").style(plain), area);
        if overflow {
            crate::tui::widgets::scroll::draw_scrollbar(
                frame,
                area,
                self.offset,
                total,
                height,
                theme,
            );
        }
        // A row is a `Static`: its own markup colors stay inside the highlight, as in Textual.
        let shown = items
            .iter()
            .enumerate()
            .flat_map(|(index, lines)| lines.iter().map(move |line| (index, line)));
        for (row, (index, line)) in shown.skip(self.offset).take(height).enumerate() {
            let rect = Rect::new(area.x, area.y + row as u16, row_width, 1);
            let selected = Some(index) == self.index;
            let first_row = starts[index] == self.offset + row;
            let (style, marker) = match (selected, first_row) {
                (true, true) => (highlight, "▶ "),
                (true, false) => (highlight, "  "),
                (false, _) => (plain, "  "),
            };
            let marker_style = style.add_modifier(ratatui::style::Modifier::BOLD);
            let mut spans = vec![Span::styled(marker, marker_style)];
            spans.extend(line.spans.iter().cloned());
            let shown_line = Line::from(spans).style(line.style);
            frame.render_widget(Paragraph::new(shown_line).style(style), rect);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn up_and_down_skip_the_disabled_rows() {
        let mut list = ListCursor::default();
        list.reset_with(vec![true, false, true, false, false]);
        assert_eq!(list.index, Some(1));
        list.key("down");
        assert_eq!(list.index, Some(3));
        list.key("up");
        assert_eq!(list.index, Some(1));
        // Nothing enabled above: the highlight stays.
        list.key("up");
        assert_eq!(list.index, Some(1));
    }

    #[test]
    fn with_no_highlight_down_takes_the_first_row_and_up_the_last() {
        let mut list = ListCursor::default();
        list.reset_unselected(4);
        list.key("down");
        assert_eq!(list.index, Some(0));
        list.reset_unselected(4);
        list.key("up");
        assert_eq!(list.index, Some(3));
    }

    #[test]
    fn home_end_and_the_page_keys_scroll_and_leave_the_highlight() {
        let theme = Theme::load("textual-dark");
        let rows: Vec<Line> = (0..20).map(|n| Line::from(format!("row {n}"))).collect();
        let mut list = ListCursor::default();
        list.reset(rows.len());
        let mut terminal =
            ratatui::Terminal::new(ratatui::backend::TestBackend::new(20, 5)).unwrap();
        let mut draw = |list: &mut ListCursor| {
            terminal
                .draw(|frame| list.render(frame, frame.area(), &rows, true, Color::Reset, &theme))
                .unwrap();
        };
        draw(&mut list);
        list.key("end");
        draw(&mut list);
        assert_eq!((list.index, list.offset), (Some(0), 15));
        list.key("pageup");
        draw(&mut list);
        assert_eq!((list.index, list.offset), (Some(0), 10));
        list.key("home");
        draw(&mut list);
        assert_eq!((list.index, list.offset), (Some(0), 0));
        list.key("pagedown");
        draw(&mut list);
        assert_eq!((list.index, list.offset), (Some(0), 5));
        // A highlight move scrolls back to it.
        list.key("down");
        draw(&mut list);
        assert_eq!((list.index, list.offset), (Some(1), 1));
    }
}
