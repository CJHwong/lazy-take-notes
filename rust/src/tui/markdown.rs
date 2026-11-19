//! Markdown to wrapped lines, styled the way Textual's `Markdown` widget styles it.
//!
//! Vertical margins between blocks collapse to the larger one, as in Textual.

use pulldown_cmark::{Alignment, Event, HeadingLevel, Options, Parser, Tag, TagEnd};
use ratatui::layout::Alignment as LineAlignment;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};

use crate::tui::text::{center, str_width, truncate, wrap_line};
use crate::tui::theme::Theme;

const BULLETS: [&str; 5] = ["• ", "▪ ", "‣ ", "⭑ ", "◦ "];

struct ListState {
    /// The next number, for an ordered list.
    next: Option<u64>,
    /// The width of the widest number, for right-aligned numbers.
    number_width: usize,
}

struct Table {
    alignments: Vec<Alignment>,
    rows: Vec<Vec<Vec<Span<'static>>>>,
    in_header: bool,
    header_rows: usize,
}

struct Builder<'t> {
    theme: &'t Theme,
    width: u16,
    out: Vec<Line<'static>>,
    pending_gap: usize,
    /// Inline spans of the block being read.
    spans: Vec<Span<'static>>,
    styles: Vec<Style>,
    lists: Vec<ListState>,
    /// The bullet or number waiting for the item's first line.
    item_marker: Option<Span<'static>>,
    quote_depth: usize,
    code: Option<Vec<String>>,
    table: Option<Table>,
}

impl<'t> Builder<'t> {
    fn new(theme: &'t Theme, width: u16) -> Self {
        Self {
            theme,
            width,
            out: Vec::new(),
            pending_gap: 0,
            spans: Vec::new(),
            styles: vec![Style::default().fg(theme.color("foreground"))],
            lists: Vec::new(),
            item_marker: None,
            quote_depth: 0,
            code: None,
            table: None,
        }
    }

    fn style(&self) -> Style {
        *self.styles.last().expect("a base style")
    }

    fn push_style(&mut self, patch: Style) {
        let next = self.style().patch(patch);
        self.styles.push(next);
    }

    fn pop_style(&mut self) {
        if self.styles.len() > 1 {
            self.styles.pop();
        }
    }

    fn gap(&mut self, rows: usize) {
        self.pending_gap = self.pending_gap.max(rows);
    }

    fn emit(&mut self, line: Line<'static>) {
        for _ in 0..self.pending_gap {
            self.out.push(self.decorate(Line::default()));
        }
        self.pending_gap = 0;
        self.out.push(line);
    }

    /// The quote bars in front of a line.
    fn decorate(&self, line: Line<'static>) -> Line<'static> {
        if self.quote_depth == 0 {
            return line;
        }
        let bar = Style::default()
            .fg(self.theme.color("text-primary"))
            .bg(self.theme.color("boost"));
        let mut spans: Vec<Span<'static>> = (0..self.quote_depth)
            .map(|_| Span::styled("▌ ", bar))
            .collect();
        spans.extend(line.spans);
        Line::from(spans).alignment(line.alignment.unwrap_or(LineAlignment::Left))
    }

    /// The columns the list and quote prefixes take.
    fn indent(&self) -> usize {
        self.quote_depth * 2 + self.lists.iter().map(list_marker_width).sum::<usize>()
    }

    /// Wrap the collected spans as one block, with the list marker on its first row.
    fn flush_text(&mut self, alignment: LineAlignment) {
        let spans = std::mem::take(&mut self.spans);
        if spans.is_empty() && self.item_marker.is_none() {
            return;
        }
        let indent = self.indent();
        let width = usize::from(self.width).saturating_sub(indent).max(1);
        let rows = wrap_line(&Line::from(spans), u16::try_from(width).unwrap_or(u16::MAX));
        let marker = self.item_marker.take();
        let outer = indent.saturating_sub(self.quote_depth * 2);
        for (index, row) in rows.into_iter().enumerate() {
            let mut prefix: Vec<Span<'static>> = Vec::new();
            match (&marker, index) {
                (Some(marker), 0) => {
                    let pad = outer.saturating_sub(str_width(&marker.content));
                    prefix.push(Span::raw(" ".repeat(pad)));
                    prefix.push(marker.clone());
                }
                _ => prefix.push(Span::raw(" ".repeat(outer))),
            }
            prefix.extend(row.spans);
            let mut line = Line::from(prefix);
            if alignment == LineAlignment::Center {
                line = center(line, self.width);
            }
            let decorated = self.decorate(line);
            self.emit(decorated);
        }
    }

    fn heading_style(&self, level: HeadingLevel) -> Style {
        let number = level as usize;
        let color = self.theme.color(&format!("markdown-h{number}-color"));
        let mut style = Style::default().fg(color);
        let text_style = match number {
            2 => "underline",
            4 => "bold underline",
            _ => "bold",
        };
        if text_style.contains("bold") {
            style = style.add_modifier(Modifier::BOLD);
        }
        if text_style.contains("underline") {
            style = style.add_modifier(Modifier::UNDERLINED);
        }
        style
    }

    fn code_style(&self) -> Style {
        let (background, foreground) = if self.theme.dark {
            ("warning", "text-warning")
        } else {
            ("error", "text-error")
        };
        Style::default()
            .bg(tint(
                self.theme.color(background),
                self.theme.color("background"),
                if self.theme.dark { 0.1 } else { 0.05 },
            ))
            .fg(self.theme.color(foreground))
    }

    fn start(&mut self, tag: Tag<'_>) {
        match tag {
            Tag::Paragraph => {}
            Tag::Heading { level, .. } => {
                self.flush_text(LineAlignment::Left);
                let top = if matches!(level, HeadingLevel::H3 | HeadingLevel::H4) {
                    1
                } else {
                    2
                };
                self.gap(top);
                let style = self.heading_style(level);
                self.push_style(style);
            }
            Tag::BlockQuote(_) => {
                self.flush_text(LineAlignment::Left);
                self.gap(1);
                self.quote_depth += 1;
            }
            Tag::CodeBlock(_) => {
                self.flush_text(LineAlignment::Left);
                self.gap(1);
                self.code = Some(Vec::new());
            }
            Tag::List(first) => {
                self.flush_text(LineAlignment::Left);
                if self.lists.is_empty() {
                    self.gap(0);
                }
                self.lists.push(ListState {
                    next: first,
                    number_width: 0,
                });
            }
            Tag::Item => {
                self.flush_text(LineAlignment::Left);
                let depth = self.lists.len().saturating_sub(1);
                let bullet_style = Style::default().fg(self.theme.color(if self.theme.dark {
                    "text-primary"
                } else {
                    "text-secondary"
                }));
                let list = self.lists.last_mut().expect("an item is in a list");
                let marker = match list.next.as_mut() {
                    Some(number) => {
                        let text = format!("{number}.");
                        *number += 1;
                        list.number_width = list.number_width.max(text.len());
                        format!("{text:>width$} ", width = list.number_width)
                    }
                    None => BULLETS[depth % BULLETS.len()].to_string(),
                };
                self.item_marker = Some(Span::styled(marker, bullet_style));
            }
            Tag::Emphasis => self.push_style(Style::default().add_modifier(Modifier::ITALIC)),
            Tag::Strong => self.push_style(Style::default().add_modifier(Modifier::BOLD)),
            Tag::Strikethrough => {
                self.push_style(Style::default().add_modifier(Modifier::CROSSED_OUT))
            }
            Tag::Link { .. } => {
                self.push_style(Style::default().add_modifier(Modifier::UNDERLINED))
            }
            Tag::Table(alignments) => {
                self.flush_text(LineAlignment::Left);
                self.table = Some(Table {
                    alignments,
                    rows: Vec::new(),
                    in_header: false,
                    header_rows: 0,
                });
            }
            Tag::TableHead => {
                if let Some(table) = self.table.as_mut() {
                    table.in_header = true;
                    table.rows.push(Vec::new());
                }
            }
            Tag::TableRow => {
                if let Some(table) = self.table.as_mut() {
                    table.rows.push(Vec::new());
                }
            }
            Tag::TableCell => self.spans.clear(),
            _ => {}
        }
    }

    fn end(&mut self, tag: TagEnd) {
        match tag {
            TagEnd::Paragraph => {
                if self.table.is_none() {
                    self.flush_text(LineAlignment::Left);
                    if self.lists.is_empty() {
                        self.gap(1);
                    }
                }
            }
            TagEnd::Heading(level) => {
                let alignment = if level == HeadingLevel::H1 {
                    LineAlignment::Center
                } else {
                    LineAlignment::Left
                };
                self.flush_text(alignment);
                self.pop_style();
                self.gap(1);
            }
            TagEnd::BlockQuote(_) => {
                self.flush_text(LineAlignment::Left);
                self.quote_depth = self.quote_depth.saturating_sub(1);
                self.gap(1);
            }
            TagEnd::CodeBlock => self.end_code(),
            TagEnd::List(_) => {
                self.flush_text(LineAlignment::Left);
                self.lists.pop();
                if self.lists.is_empty() {
                    self.gap(1);
                }
            }
            TagEnd::Item => self.flush_text(LineAlignment::Left),
            TagEnd::Emphasis | TagEnd::Strong | TagEnd::Strikethrough | TagEnd::Link => {
                self.pop_style()
            }
            TagEnd::TableCell => {
                let cell = std::mem::take(&mut self.spans);
                if let Some(row) = self.table.as_mut().and_then(|table| table.rows.last_mut()) {
                    row.push(cell);
                }
            }
            TagEnd::TableHead => {
                if let Some(table) = self.table.as_mut() {
                    table.in_header = false;
                    table.header_rows = table.rows.len();
                }
            }
            TagEnd::Table => self.end_table(),
            _ => {}
        }
    }

    fn end_code(&mut self) {
        let Some(lines) = self.code.take() else {
            return;
        };
        // Textual: `black 10%`, or `white 30%` on a light theme.
        let background = if self.theme.dark {
            tint(Color::Rgb(0, 0, 0), self.theme.color("background"), 0.1)
        } else {
            tint(
                Color::Rgb(255, 255, 255),
                self.theme.color("background"),
                0.3,
            )
        };
        let style = Style::default()
            .fg(Color::Rgb(210, 210, 210))
            .bg(background);
        let indent = self.indent();
        let inner = usize::from(self.width).saturating_sub(indent);
        let blank = Line::from(vec![
            Span::raw(" ".repeat(indent)),
            Span::styled(" ".repeat(inner), style),
        ]);
        self.emit(blank.clone());
        for text in lines {
            let shown = truncate(&format!("  {text}"), inner);
            let pad = inner.saturating_sub(str_width(&shown));
            let line = Line::from(vec![
                Span::raw(" ".repeat(indent)),
                Span::styled(format!("{shown}{}", " ".repeat(pad)), style),
            ]);
            self.emit(line);
        }
        self.emit(blank);
        self.gap(1);
    }

    fn end_table(&mut self) {
        let Some(table) = self.table.take() else {
            return;
        };
        let columns = table.rows.iter().map(Vec::len).max().unwrap_or(0);
        if columns == 0 {
            return;
        }
        let cell_width = |cell: &Vec<Span<'static>>| {
            cell.iter()
                .map(|span| str_width(&span.content))
                .sum::<usize>()
        };
        let mut widths = vec![0; columns];
        for row in &table.rows {
            for (index, cell) in row.iter().enumerate() {
                widths[index] = widths[index].max(cell_width(cell));
            }
        }
        // Shrink the widest columns until the row fits, then cut the cells.
        let budget = usize::from(self.width).saturating_sub(self.indent() + 3 * columns);
        while widths.iter().sum::<usize>() > budget && widths.iter().any(|w| *w > 3) {
            let widest = widths
                .iter()
                .enumerate()
                .max_by_key(|(_, w)| **w)
                .map(|(i, _)| i)
                .unwrap_or(0);
            widths[widest] -= 1;
        }
        let keyline = Style::default().fg(tint(
            self.theme.color("foreground"),
            self.theme.color("background"),
            0.2,
        ));
        let header = Style::default()
            .fg(self.theme.color("primary"))
            .add_modifier(Modifier::BOLD);
        for (row_index, row) in table.rows.iter().enumerate() {
            let is_header = row_index < table.header_rows;
            let mut spans = vec![Span::raw(" ".repeat(self.indent()))];
            for (index, width) in widths.iter().enumerate() {
                if index > 0 {
                    spans.push(Span::styled("│", keyline));
                }
                let text: String = row
                    .get(index)
                    .map(|cell| cell.iter().map(|s| s.content.as_ref()).collect())
                    .unwrap_or_default();
                let shown = if str_width(&text) > *width {
                    format!("{}…", truncate(&text, width.saturating_sub(1)))
                } else {
                    text
                };
                let pad = width.saturating_sub(str_width(&shown));
                let aligned = match table.alignments.get(index) {
                    Some(Alignment::Right) => format!("{}{shown}", " ".repeat(pad)),
                    Some(Alignment::Center) => format!(
                        "{}{shown}{}",
                        " ".repeat(pad / 2),
                        " ".repeat(pad - pad / 2)
                    ),
                    _ => format!("{shown}{}", " ".repeat(pad)),
                };
                let style = if is_header { header } else { self.style() };
                spans.push(Span::styled(format!(" {aligned} "), style));
            }
            self.emit(Line::from(spans));
            if is_header && row_index + 1 == table.header_rows {
                let rule: Vec<String> = widths.iter().map(|w| "─".repeat(w + 2)).collect();
                self.emit(Line::from(vec![
                    Span::raw(" ".repeat(self.indent())),
                    Span::styled(rule.join("┼"), keyline),
                ]));
            }
        }
        self.gap(1);
    }

    fn text(&mut self, text: &str) {
        if let Some(code) = self.code.as_mut() {
            code.extend(text.trim_end_matches('\n').split('\n').map(str::to_string));
            return;
        }
        let style = self.style();
        self.spans.push(Span::styled(text.to_string(), style));
    }

    fn rule(&mut self) {
        self.flush_text(LineAlignment::Left);
        self.gap(1);
        let width = usize::from(self.width).saturating_sub(self.indent());
        let line = Line::from(Span::styled(
            "─".repeat(width),
            Style::default().fg(self.theme.color("secondary")),
        ));
        // `padding-top: 1`, then the bottom border.
        self.emit(Line::default());
        self.emit(line);
        self.gap(1);
    }
}

fn list_marker_width(list: &ListState) -> usize {
    match list.next {
        Some(_) => list.number_width.max(2) + 1,
        None => 2,
    }
}

/// `color` at `alpha` over `under`.
fn tint(color: Color, under: Color, alpha: f64) -> Color {
    match (color, under) {
        (Color::Rgb(r, g, b), Color::Rgb(ur, ug, ub)) => {
            let mix = |from: u8, to: u8| {
                (f64::from(from) + (f64::from(to) - f64::from(from)) * alpha) as u8
            };
            Color::Rgb(mix(ur, r), mix(ug, g), mix(ub, b))
        }
        (color, _) => color,
    }
}

/// Render `markdown` into lines `width` columns wide.
pub fn render(markdown: &str, width: u16, theme: &Theme) -> Vec<Line<'static>> {
    let options = Options::ENABLE_TABLES | Options::ENABLE_STRIKETHROUGH;
    let mut builder = Builder::new(theme, width);
    for event in Parser::new_ext(markdown, options) {
        match event {
            Event::Start(tag) => builder.start(tag),
            Event::End(tag) => builder.end(tag),
            Event::Text(text) => builder.text(&text),
            Event::Code(code) => {
                let style = builder.style().patch(builder.code_style());
                builder.spans.push(Span::styled(code.to_string(), style));
            }
            Event::SoftBreak => builder.text(" "),
            Event::HardBreak => builder.flush_text(LineAlignment::Left),
            Event::Rule => builder.rule(),
            Event::Html(html) | Event::InlineHtml(html) => builder.text(&html),
            Event::TaskListMarker(done) => builder.text(if done { "☑ " } else { "☐ " }),
            _ => {}
        }
    }
    builder.flush_text(LineAlignment::Left);
    builder.out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plain(lines: &[Line<'_>]) -> Vec<String> {
        lines
            .iter()
            .map(|line| {
                line.spans
                    .iter()
                    .map(|s| s.content.as_ref())
                    .collect::<String>()
                    .trim_end()
                    .to_string()
            })
            .collect()
    }

    #[test]
    fn blocks_keep_textual_margins_and_lists_hang() {
        let theme = Theme::load("textual-dark");
        let lines = render(
            "## Topics\n\n- one two three four\n- five\n\nAfter.",
            14,
            &theme,
        );
        assert_eq!(
            plain(&lines),
            [
                "",
                "",
                "Topics",
                "",
                "• one two",
                "  three four",
                "• five",
                "",
                "After."
            ]
        );
        let numbered = render("1. a\n2. b", 20, &theme);
        assert_eq!(plain(&numbered), ["1. a", "2. b"]);
        let cjk = render("### 已結案議題\n\n本集由主持人", 8, &theme);
        assert_eq!(plain(&cjk), ["", "已結案議", "題", "", "本集由主", "持人"]);
    }
}
