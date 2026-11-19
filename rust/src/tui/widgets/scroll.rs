//! Scrolling text: a log of lines (Textual's `RichLog`) or rendered Markdown.
//!
//! The scrollbar column appears only when the content overflows, as with
//! Textual's `overflow-y: auto`; the text then wraps one column narrower.

use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::Style;
use ratatui::text::Line;
use ratatui::widgets::Paragraph;

use crate::tui::markdown;
use crate::tui::text::wrap_lines;
use crate::tui::theme::Theme;

enum Content {
    Lines(Vec<Line<'static>>),
    Markdown(String),
}

pub struct ScrollView {
    content: Content,
    offset: usize,
    /// Follow new lines to the bottom, as `RichLog(auto_scroll=True)` does.
    pub auto_scroll: bool,
    at_end_pending: bool,
    /// The scroll container's padding: columns each side, rows above and below.
    /// The scrollbar sits inside it, as in Textual.
    pub padding: u16,
    pub padding_y: u16,
    /// Padding of the content inside the container, inside the scrollbar.
    pub content_padding: u16,
    /// Lines that scroll above the content, outside the Markdown's padding.
    pub prefix: Vec<Line<'static>>,
    /// Rows a docked widget adds to the scroll height, its margin included.
    pub dock_rows: u16,
    /// Rows the docked widget itself covers; content draws in its margin.
    pub dock_cover: u16,
    /// The Markdown widget's own `padding: 0 N`, inside `content_padding`.
    pub markdown_padding: u16,
    /// `overflow-y: scroll`: the scrollbar column is there even when nothing overflows.
    pub always_bar: bool,
    /// Columns the vertical scrollbar takes: 1 where the CSS says `scrollbar-size: 1 1`,
    /// Textual's default 2 elsewhere.
    pub bar_width: u16,
    rows: usize,
    height: usize,
}

impl ScrollView {
    pub fn log() -> Self {
        Self {
            content: Content::Lines(Vec::new()),
            offset: 0,
            auto_scroll: true,
            at_end_pending: false,
            padding: 0,
            padding_y: 0,
            content_padding: 0,
            prefix: Vec::new(),
            dock_rows: 0,
            dock_cover: 0,
            markdown_padding: 0,
            always_bar: true,
            bar_width: 1,
            rows: 0,
            height: 0,
        }
    }

    pub fn markdown() -> Self {
        Self {
            content: Content::Markdown(String::new()),
            auto_scroll: false,
            padding: 2,
            always_bar: false,
            ..Self::log()
        }
    }

    pub fn write(&mut self, line: Line<'static>) {
        if let Content::Lines(lines) = &mut self.content {
            lines.push(line);
        }
        if self.auto_scroll {
            self.at_end_pending = true;
        }
    }

    pub fn clear(&mut self) {
        self.content = match self.content {
            Content::Lines(_) => Content::Lines(Vec::new()),
            Content::Markdown(_) => Content::Markdown(String::new()),
        };
        self.offset = 0;
    }

    /// Replace the content with plain lines, from the top (a `Static.update`).
    pub fn set_lines(&mut self, lines: Vec<Line<'static>>) {
        self.content = Content::Lines(lines);
        self.offset = 0;
    }

    pub fn set_markdown(&mut self, text: &str) {
        self.content = Content::Markdown(text.to_string());
    }

    /// True when the last render had more rows than fit.
    pub fn overflows(&self) -> bool {
        self.rows > self.height
    }

    pub fn is_at_end(&self) -> bool {
        self.offset + self.height >= self.rows
    }

    pub fn scroll_end(&mut self) {
        self.at_end_pending = true;
    }

    /// Scroll by a key. Returns true when the key was a scroll key.
    pub fn key(&mut self, name: &str) -> bool {
        let page = self.height.max(1);
        let last = self.rows.saturating_sub(self.height);
        self.offset = match name {
            "up" => self.offset.saturating_sub(1),
            "down" => (self.offset + 1).min(last),
            "pageup" => self.offset.saturating_sub(page),
            "pagedown" => (self.offset + page).min(last),
            "home" => 0,
            "end" => last,
            _ => return false,
        };
        true
    }

    /// Rows the content takes in a container `width` columns wide, without a scrollbar.
    pub fn content_rows(&self, width: u16, theme: &Theme) -> usize {
        let text_width = width
            .saturating_sub((self.padding + self.content_padding) * 2)
            .max(1);
        self.lines_at(text_width, theme, Style::default()).len() + usize::from(self.dock_rows)
    }

    /// The wrapped lines. `base` styles the Markdown's own padding, which a quote's
    /// background does not cover.
    fn lines_at(&self, width: u16, theme: &Theme, base: Style) -> Vec<Line<'static>> {
        let mut all = wrap_lines(&self.prefix, width);
        match &self.content {
            Content::Lines(lines) => all.extend(wrap_lines(lines, width)),
            Content::Markdown(text) => {
                let pad = self.markdown_padding;
                let rendered = markdown::render(text, width.saturating_sub(pad * 2).max(1), theme);
                all.extend(rendered.into_iter().map(|line| {
                    let mut spans = vec![ratatui::text::Span::styled(
                        " ".repeat(usize::from(pad)),
                        base,
                    )];
                    spans.extend(line.spans);
                    Line::from(spans).style(line.style)
                }));
            }
        }
        all
    }

    pub fn render(&mut self, frame: &mut Frame, area: Rect, style: Style, theme: &Theme) {
        frame.render_widget(Paragraph::new("").style(style), area);
        // The region inside the container's padding holds the content and the scrollbar.
        let region = Rect::new(
            area.x + self.padding,
            area.y + self.padding_y,
            area.width.saturating_sub(self.padding * 2),
            area.height.saturating_sub(self.padding_y * 2),
        );
        let height = usize::from(region.height);
        let full = region.width.saturating_sub(self.content_padding * 2).max(1);
        let dock = usize::from(self.dock_rows);
        let mut lines = self.lines_at(full, theme, style);
        let bar_column = lines.len() + dock > height || self.always_bar;
        if bar_column {
            lines = self.lines_at(full.saturating_sub(self.bar_width).max(1), theme, style);
        }
        self.rows = lines.len() + dock;
        self.height = height;
        let last = self.rows.saturating_sub(height);
        if self.at_end_pending {
            self.offset = last;
            self.at_end_pending = false;
        }
        self.offset = self.offset.min(last);
        let text_area = Rect::new(
            region.x + self.content_padding,
            region.y,
            region
                .width
                .saturating_sub(u16::from(bar_column) * self.bar_width + self.content_padding * 2),
            region.height.saturating_sub(self.dock_cover),
        );
        let shown: Vec<Line<'static>> = lines
            .into_iter()
            .skip(self.offset)
            .take(height.saturating_sub(usize::from(self.dock_cover)))
            .collect();
        frame.render_widget(Paragraph::new(shown).style(style), text_area);
        if self.rows > height {
            for column in 0..self.bar_width.min(region.width) {
                let strip = Rect::new(region.x, region.y, region.width - column, region.height);
                draw_scrollbar(frame, strip, self.offset, self.rows, height, theme);
            }
        }
    }
}

const VERTICAL_BARS: [&str; 8] = ["▁", "▂", "▃", "▄", "▅", "▆", "▇", " "];

/// Textual's `ScrollBarRender.render_bar`: the thumb in whole cells, with
/// eighth-block glyphs at its ends.
pub fn draw_scrollbar(
    frame: &mut Frame,
    area: Rect,
    offset: usize,
    rows: usize,
    height: usize,
    theme: &Theme,
) {
    let size = usize::from(area.height);
    if size == 0 || area.width == 0 {
        return;
    }
    let back = theme.color("scrollbar-background");
    let bar = theme.color("scrollbar");
    let x = area.right() - 1;
    let mut cells: Vec<(&str, Style)> = vec![(" ", Style::default().bg(back)); size];
    if height > 0 && rows > 0 && rows != size && rows > height {
        let len_bars = VERTICAL_BARS.len();
        let bar_ratio = rows as f64 / size as f64;
        let thumb_size = (height as f64 / bar_ratio).max(1.0);
        let position_ratio = offset as f64 / (rows - height) as f64;
        let position = (size as f64 - thumb_size) * position_ratio;
        let start = (position * len_bars as f64) as usize;
        let end = start + (thumb_size * len_bars as f64).ceil() as usize;
        let (start_index, start_bar) = (start / len_bars, start % len_bars);
        let (end_index, end_bar) = (end / len_bars, end % len_bars);
        for cell in cells.iter_mut().take(end_index.min(size)).skip(start_index) {
            *cell = (" ", Style::default().bg(bar));
        }
        if start_index < size && VERTICAL_BARS[len_bars - 1 - start_bar] != " " {
            cells[start_index] = (
                VERTICAL_BARS[len_bars - 1 - start_bar],
                Style::default().bg(back).fg(bar),
            );
        }
        if end_index < size && VERTICAL_BARS[len_bars - 1 - end_bar] != " " {
            cells[end_index] = (
                VERTICAL_BARS[len_bars - 1 - end_bar],
                Style::default().bg(bar).fg(back),
            );
        }
    }
    let buffer = frame.buffer_mut();
    for (row, (symbol, style)) in cells.into_iter().enumerate() {
        if let Some(cell) = buffer.cell_mut((x, area.y + row as u16)) {
            cell.set_symbol(symbol);
            cell.set_style(style);
        }
    }
}
