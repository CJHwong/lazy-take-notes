//! Markdown to wrapped lines, styled the way Textual's `Markdown` widget styles it.
//!
//! Vertical margins between blocks collapse to the larger one, as in Textual.

use pulldown_cmark::{Event, HeadingLevel, Options, Parser, Tag, TagEnd};
use ratatui::layout::Alignment as LineAlignment;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};

use crate::tui::text::{center, str_width, truncate, wrap_line};
use crate::tui::theme::Theme;

const BULLETS: [&str; 5] = ["• ", "▪ ", "‣ ", "⭑ ", "◦ "];

struct ListState {
    /// The next number, for an ordered list.
    next: Option<u64>,
    /// The width of every marker: Textual right-justifies `"{n}. "` to the
    /// length of the last one plus 1.
    marker_width: usize,
}

struct Table {
    rows: Vec<Vec<Vec<Span<'static>>>>,
    in_header: bool,
    header_rows: usize,
}

struct Builder<'t> {
    theme: &'t Theme,
    width: u16,
    out: Vec<Line<'static>>,
    pending_gap: usize,
    /// The quote depth the pending gap is drawn at.
    gap_depth: usize,
    /// Inline spans of the block being read.
    spans: Vec<Span<'static>>,
    styles: Vec<Style>,
    lists: Vec<ListState>,
    /// Items per list, from a first pass, and how many lists have started.
    list_items: Vec<usize>,
    lists_seen: usize,
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
            gap_depth: 0,
            spans: Vec::new(),
            styles: vec![Style::default().fg(theme.color("foreground"))],
            lists: Vec::new(),
            list_items: Vec::new(),
            lists_seen: 0,
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
        // A margin belongs to the outermost block that asked for it: the gap
        // before a quote has no bar, the gap between its paragraphs has one.
        self.gap_depth = match self.pending_gap {
            0 => self.quote_depth,
            _ => self.gap_depth.min(self.quote_depth),
        };
        self.pending_gap = self.pending_gap.max(rows);
    }

    fn emit(&mut self, line: Line<'static>) {
        for _ in 0..self.pending_gap {
            self.out
                .push(self.decorate_at(Line::default(), self.gap_depth));
        }
        self.pending_gap = 0;
        self.out.push(line);
    }

    /// The quote bars in front of a line.
    fn decorate(&self, line: Line<'static>) -> Line<'static> {
        self.decorate_at(line, self.quote_depth)
    }

    fn decorate_at(&self, line: Line<'static>, depth: usize) -> Line<'static> {
        if depth == 0 {
            return line;
        }
        // `background: $boost` on the whole row; `border-left: outer $text-primary 50%`
        // (`$text-secondary` in a light theme), its color taken over the parent.
        let boost = self.theme.color("boost");
        let bar_color = match self.theme.dark {
            true => tint(
                self.theme.color("text-primary"),
                self.theme.color("background"),
                0.5,
            ),
            false => self.theme.color("text-secondary"),
        };
        let bar = Style::default().fg(bar_color).bg(boost);
        let mut spans: Vec<Span<'static>> = (0..depth).map(|_| Span::styled("▌ ", bar)).collect();
        spans.extend(line.spans);
        // The quote's background fills the row to the right edge.
        let used: usize = spans.iter().map(|span| str_width(&span.content)).sum();
        let rest = usize::from(self.width).saturating_sub(used);
        spans.push(Span::styled(" ".repeat(rest), Style::default().bg(boost)));
        Line::from(spans)
            .style(line.style.bg(boost))
            .alignment(line.alignment.unwrap_or(LineAlignment::Left))
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
        let code_background = tint(
            self.theme.color(background),
            self.theme.color("background"),
            if self.theme.dark { 0.1 } else { 0.05 },
        );
        // `color: $text-warning 95%` (or `$text-error 95%`) over the code's background.
        Style::default().bg(code_background).fg(tint(
            self.theme.color(foreground),
            code_background,
            0.95,
        ))
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
                let items = self.list_items.get(self.lists_seen).copied().unwrap_or(1);
                self.lists_seen += 1;
                let marker_width = match first {
                    Some(start) => format!("{}. ", start + items.max(1) as u64 - 1).len() + 1,
                    None => 2,
                };
                self.lists.push(ListState {
                    next: first,
                    marker_width,
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
                        let text = format!("{number}. ");
                        *number += 1;
                        format!("{text:>width$}", width = list.marker_width)
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
            Tag::Table(_) => {
                self.flush_text(LineAlignment::Left);
                self.table = Some(Table {
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

    /// Textual's `MarkdownTableContent`: a grid with `keyline: thin`, columns
    /// sized by `resolve_columns`, cells padded `0 1` and wrapped.
    fn end_table(&mut self) {
        let Some(table) = self.table.take() else {
            return;
        };
        let columns = table.rows.iter().map(Vec::len).max().unwrap_or(0);
        if columns == 0 {
            return;
        }
        let cell_line = |row: &Vec<Vec<Span<'static>>>, index: usize| -> Line<'static> {
            Line::from(row.get(index).cloned().unwrap_or_default())
        };
        let mut natural = vec![0usize; columns];
        let mut minimums = vec![1usize; columns];
        for row in &table.rows {
            for index in 0..columns {
                let line = cell_line(row, index);
                let text: String = line
                    .spans
                    .iter()
                    .map(|span| span.content.as_ref())
                    .collect();
                natural[index] = natural[index].max(str_width(&text) + 2);
                let longest_word = text.split_whitespace().map(str_width).max().unwrap_or(0);
                minimums[index] = minimums[index].max(longest_word + 2);
            }
        }
        let indent = self.indent();
        // The outer keylines take one column on each side.
        let grid = usize::from(self.width).saturating_sub(indent + 2);
        let widths = resolve_columns(&natural, &minimums, grid, 1);
        let keyline = Style::default().fg(tint(
            self.theme.color("foreground"),
            self.theme.color("background"),
            0.2,
        ));
        let header_style = Style::default()
            .fg(self.theme.color("primary"))
            .add_modifier(Modifier::BOLD);
        let rule = |left: &str, middle: &str, right: &str| {
            let parts: Vec<String> = widths.iter().map(|width| "─".repeat(*width)).collect();
            Line::from(vec![
                Span::raw(" ".repeat(indent)),
                Span::styled(format!("{left}{}{right}", parts.join(middle)), keyline),
            ])
        };
        self.emit(rule("┌", "┬", "┐"));
        for (row_index, row) in table.rows.iter().enumerate() {
            let is_header = row_index < table.header_rows;
            let cells: Vec<Vec<Line<'static>>> = widths
                .iter()
                .enumerate()
                .map(|(index, width)| {
                    let mut line = cell_line(row, index);
                    if is_header {
                        line = Line::from(
                            line.spans
                                .into_iter()
                                .map(|span| Span::styled(span.content, header_style))
                                .collect::<Vec<_>>(),
                        );
                    }
                    wrap_line(&line, (*width as u16).saturating_sub(2).max(1))
                })
                .collect();
            let height = cells.iter().map(Vec::len).max().unwrap_or(1).max(1);
            for line_index in 0..height {
                let mut spans = vec![Span::raw(" ".repeat(indent)), Span::styled("│", keyline)];
                for (index, width) in widths.iter().enumerate() {
                    // A header sits at the bottom of its row (`content-align: left bottom`).
                    let lines = &cells[index];
                    let shift = if is_header { height - lines.len() } else { 0 };
                    let line = line_index
                        .checked_sub(shift)
                        .and_then(|at| lines.get(at))
                        .cloned()
                        .unwrap_or_default();
                    let used = line.width();
                    spans.push(Span::raw(" "));
                    spans.extend(line.spans);
                    spans.push(Span::raw(" ".repeat(width.saturating_sub(used + 1))));
                    spans.push(Span::styled("│", keyline));
                }
                self.emit(Line::from(spans));
            }
            if row_index + 1 < table.rows.len() {
                self.emit(rule("├", "┼", "┤"));
            }
        }
        self.emit(rule("└", "┴", "┘"));
        self.gap(1);
    }

    fn text(&mut self, text: &str) {
        if let Some(code) = self.code.as_mut() {
            code.extend(text.trim_end_matches('\n').split('\n').map(str::to_string));
            return;
        }
        let style = self.style();
        // Textual's `re.sub(r"\s+", " ", ...)` on each inline text token.
        let collapsed = text.split_whitespace().collect::<Vec<_>>().join(" ");
        let edges = (
            text.starts_with(char::is_whitespace) && !collapsed.is_empty(),
            text.ends_with(char::is_whitespace) && !collapsed.is_empty(),
        );
        let text = match (collapsed.is_empty() && !text.is_empty(), edges) {
            (true, _) => " ".to_string(),
            (false, (lead, trail)) => format!(
                "{}{collapsed}{}",
                if lead { " " } else { "" },
                if trail { " " } else { "" }
            ),
        };
        self.spans.push(Span::styled(text, style));
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
    list.marker_width
}

/// `color` at `alpha` over `under`.
pub(crate) fn tint(color: Color, under: Color, alpha: f64) -> Color {
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
    builder.list_items = list_item_counts(markdown, options);
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
    // The last block's bottom margin is part of the Markdown's height in Textual.
    if !builder.out.is_empty() {
        for _ in 0..builder.pending_gap {
            let line = builder.decorate_at(Line::default(), builder.gap_depth);
            builder.out.push(line);
        }
    }
    builder.out
}

/// The number of items in each list, in the order the lists start. A marker's
/// width depends on the last number, which a single pass has not seen yet.
fn list_item_counts(markdown: &str, options: Options) -> Vec<usize> {
    let mut counts = Vec::new();
    let mut open: Vec<usize> = Vec::new();
    for event in Parser::new_ext(markdown, options) {
        match event {
            Event::Start(Tag::List(_)) => {
                counts.push(0);
                open.push(counts.len() - 1);
            }
            Event::End(TagEnd::List(_)) => {
                open.pop();
            }
            Event::Start(Tag::Item) => {
                if let Some(at) = open.last() {
                    counts[*at] += 1;
                }
            }
            _ => {}
        }
    }
    counts
}

/// Textual's `resolve` for a grid with `expand` and `shrink`: grow the columns
/// in proportion to fill `total`, or take the whole excess from each column in
/// turn, down to its minimum, then in proportion if still too wide.
pub fn resolve_columns(
    natural: &[usize],
    minimums: &[usize],
    total: usize,
    gutter: usize,
) -> Vec<usize> {
    let count = natural.len();
    let space = total as f64 - (gutter * count.saturating_sub(1)) as f64;
    let mut widths: Vec<f64> = natural.iter().map(|w| *w as f64).collect();
    let mut used: f64 = widths.iter().sum();
    if used < space && used > 0.0 {
        let remaining = space - used;
        widths = widths.iter().map(|w| w + w / used * remaining).collect();
    }
    let mut excess = used - space;
    if excess > 0.0 {
        for (index, minimum) in minimums.iter().enumerate() {
            let width = widths[index];
            let updated = (*minimum as f64).max(width - excess);
            widths[index] = updated;
            used = used - width + updated;
            excess = used - space;
            if excess <= 0.0 {
                break;
            }
        }
        used = widths.iter().sum();
        excess = used - space;
        if excess > 0.0 {
            widths = widths.iter().map(|w| w - w / used * excess).collect();
        }
    }
    // Offsets are the floors of the running sums, gutters included.
    let mut offsets = vec![0usize];
    let mut sum = 0.0;
    for width in &widths {
        sum += width;
        offsets.push((sum + 1e-9).floor() as usize);
        sum += gutter as f64;
        offsets.push((sum + 1e-9).floor() as usize);
    }
    (0..count)
        .map(|index| offsets[index * 2 + 1] - offsets[index * 2])
        .collect()
}

#[cfg(test)]
mod tests {

    #[test]
    fn table_columns_shrink_and_grow_as_textual_resolves_them() {
        // The help table: natural widths 34 and 49 in 52 columns, one gutter.
        assert_eq!(resolve_columns(&[34, 49], &[15, 16], 52, 1), [15, 36]);
        // A narrow table grows in proportion to fill the width.
        assert_eq!(resolve_columns(&[10, 20], &[3, 3], 61, 1), [20, 40]);
    }

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
                "After.",
                ""
            ]
        );
        // Textual: `"{n}. ".rjust(len("{last}. ") + 1)`, so a lead space, two from 10.
        let numbered = render("1. a\n2. b", 20, &theme);
        assert_eq!(plain(&numbered), [" 1. a", " 2. b", ""]);
        let long: String = (1..=11).map(|n| format!("{n}. x\n")).collect();
        let long = plain(&render(&long, 20, &theme));
        assert_eq!((long[0].as_str(), long[10].as_str()), ("  1. x", " 11. x"));
        let cjk = render("### 已結案議題\n\n本集由主持人", 8, &theme);
        assert_eq!(
            plain(&cjk),
            ["", "已結案議", "題", "", "本集由主", "持人", ""]
        );
    }

    #[test]
    fn the_height_counts_the_last_block_margin_as_textual_does() {
        // Textual 40 columns wide: this document is 17 rows, the last paragraph's margin included.
        let theme = Theme::load("textual-dark");
        let markdown = "# Title\n\nPara one\n\n- a\n- b\n\n> quote\n\n```\ncode\n```\n\nEnd";
        assert_eq!(render(markdown, 36, &theme).len(), 17);
        assert!(render("", 36, &theme).is_empty());
    }
}
