//! Textual's `Select`: the current choice in a `tall` box, and a list that
//! opens over the screen below it.

use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::Style;
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;

use crate::tui::text::str_width;
use crate::tui::theme::Theme;
use crate::tui::widgets::list::ListCursor;
use crate::tui::widgets::tall_edges;

/// `SelectOverlay { max-height: 12 }`, borders included.
const OVERLAY_MAX_HEIGHT: u16 = 12;

/// What a key did.
#[derive(Debug, PartialEq, Eq)]
pub enum SelectKey {
    Used,
    Unused,
    /// The value changed.
    Changed,
}

#[derive(Default)]
pub struct Select {
    /// (label, value) pairs.
    options: Vec<(String, String)>,
    index: usize,
    pub expanded: bool,
    list: ListCursor,
}

impl Select {
    /// `allow_blank=False`: the value is one of the options, else the first.
    pub fn new(options: Vec<(String, String)>, value: &str) -> Self {
        let mut select = Self::default();
        select.set_options(options);
        select.set_value(value);
        select
    }

    pub fn value(&self) -> String {
        self.options
            .get(self.index)
            .map(|(_, value)| value.clone())
            .unwrap_or_default()
    }

    pub fn set_value(&mut self, value: &str) {
        if let Some(index) = self.options.iter().position(|(_, v)| v == value) {
            self.index = index;
        }
    }

    pub fn options(&self) -> &[(String, String)] {
        &self.options
    }

    /// New options; the value stays when it is still one of them.
    pub fn set_options(&mut self, options: Vec<(String, String)>) {
        let value = self.value();
        self.options = options;
        self.index = 0;
        self.set_value(&value);
        self.list.reset(self.options.len());
    }

    pub fn key(&mut self, name: &str) -> SelectKey {
        if !self.expanded {
            return match name {
                "enter" | "down" | "space" | "up" => {
                    self.expanded = true;
                    self.list.reset(self.options.len());
                    self.list.select(self.index);
                    SelectKey::Used
                }
                _ => SelectKey::Unused,
            };
        }
        match name {
            "escape" => {
                self.expanded = false;
                SelectKey::Used
            }
            "enter" => {
                self.expanded = false;
                match self.list.index {
                    Some(index) if index != self.index => {
                        self.index = index;
                        SelectKey::Changed
                    }
                    _ => SelectKey::Used,
                }
            }
            _ => {
                self.list.key(name);
                SelectKey::Used
            }
        }
    }

    /// The closed box, 3 rows. `outer` is the theme variable behind the box's edges.
    pub fn render(&self, frame: &mut Frame, area: Rect, focused: bool, outer: &str, theme: &Theme) {
        if area.height < 3 || area.width < 6 {
            return;
        }
        let border = theme.color(if focused { "border" } else { "border-blurred" });
        let inner = match focused {
            true => theme.tinted("surface", "foreground", 0.05),
            false => theme.color("surface"),
        };
        let edges = tall_edges(border, inner, theme.color(outer));
        let width = usize::from(area.width - 2);
        let label = self
            .options
            .get(self.index)
            .map(|(label, _)| label.as_str())
            .unwrap_or("");
        // `padding: 0 2`; the arrow takes 2 columns (`padding: 0 0 0 1`).
        let text_width = width.saturating_sub(6);
        let shown = crate::tui::text::truncate(label, text_width);
        let pad = text_width.saturating_sub(str_width(&shown));
        let text = Style::default()
            .fg(theme.on_color("foreground", inner))
            .bg(inner);
        // `color: $foreground 50%`.
        let foreground = theme.on_color("foreground", inner);
        let arrow = Style::default()
            .fg(crate::tui::markdown::tint(foreground, inner, 0.5))
            .bg(inner);
        let arrow_glyph = if self.expanded { "▲" } else { "▼" };
        let rows = [
            Line::from(vec![
                Span::styled("▊", edges.left),
                Span::styled("▔".repeat(width), edges.fill),
                Span::styled("▎", edges.right),
            ]),
            Line::from(vec![
                Span::styled("▊", edges.left),
                Span::styled("  ", text),
                Span::styled(shown, text),
                Span::styled(" ".repeat(pad), text),
                Span::styled(" ", text),
                Span::styled(arrow_glyph, arrow),
                Span::styled("  ", text),
                Span::styled("▎", edges.right),
            ]),
            Line::from(vec![
                Span::styled("▊", edges.left),
                Span::styled("▁".repeat(width), edges.fill),
                Span::styled("▎", edges.right),
            ]),
        ];
        for (offset, line) in rows.into_iter().enumerate() {
            let rect = Rect::new(area.x, area.y + offset as u16, area.width, 1);
            frame.render_widget(Paragraph::new(line), rect);
        }
    }

    /// The open list, drawn over whatever is below the box at `below`.
    /// `screen` bounds it, as `constrain: none inside`.
    pub fn render_overlay(
        &mut self,
        frame: &mut Frame,
        below: Rect,
        screen: Rect,
        outer: &str,
        theme: &Theme,
    ) {
        if !self.expanded || below.width < 4 {
            return;
        }
        let height = (self.options.len() as u16 + 2).min(OVERLAY_MAX_HEIGHT);
        let mut y = below.y;
        if y + height > screen.bottom() {
            y = screen.bottom().saturating_sub(height);
        }
        let area = Rect::new(below.x, y, below.width, height);
        // `SelectOverlay`: a focused `tall` border on the select's background,
        // `padding: 0 1`, and each option with `padding: 0 1`.
        let border = theme.color("border");
        let inner = theme.tinted("surface", "foreground", 0.05);
        let edges = tall_edges(border, inner, theme.color(outer));
        let width = usize::from(area.width - 2);
        let row = |glyph: &str| {
            Line::from(vec![
                Span::styled("▊", edges.left),
                Span::styled(glyph.repeat(width), edges.fill),
                Span::styled("▎", edges.right),
            ])
        };
        frame.render_widget(
            Paragraph::new(row("▔")),
            Rect::new(area.x, area.y, area.width, 1),
        );
        frame.render_widget(
            Paragraph::new(row("▁")),
            Rect::new(area.x, area.bottom() - 1, area.width, 1),
        );
        for y in area.y + 1..area.bottom() - 1 {
            frame.render_widget(
                Paragraph::new(Line::from(vec![
                    Span::styled("▊", edges.left),
                    Span::styled(" ".repeat(width), Style::default().bg(inner)),
                    Span::styled("▎", edges.right),
                ])),
                Rect::new(area.x, y, area.width, 1),
            );
        }
        let rows: Vec<Line<'static>> = self
            .options
            .iter()
            .map(|(label, _)| Line::from(format!(" {label}")))
            .collect();
        let list_area = Rect::new(
            area.x + 2,
            area.y + 1,
            area.width.saturating_sub(4),
            area.height - 2,
        );
        self.list
            .render(frame, list_area, &rows, true, inner, theme);
    }

    /// Rows the open list takes, borders included.
    pub fn overlay_height(&self) -> u16 {
        (self.options.len() as u16 + 2).min(OVERLAY_MAX_HEIGHT)
    }
}

/// Textual's `Switch` with `height: 1; border: none`: `padding: 0 2` around a
/// 4-column slider. The thumb sits left when off, right when on.
pub fn draw_switch(frame: &mut Frame, area: Rect, on: bool, focused: bool, theme: &Theme) {
    let surface = match focused {
        true => theme.tinted("surface", "foreground", 0.05),
        false => theme.color("surface"),
    };
    let track = theme.color("panel-darken-2");
    let thumb = theme.color(if on { "success" } else { "panel" });
    let (left, right) = if on { (track, thumb) } else { (thumb, track) };
    let line = Line::from(vec![
        Span::styled("  ", Style::default().bg(surface)),
        Span::styled("  ", Style::default().bg(left)),
        Span::styled("  ", Style::default().bg(right)),
        Span::styled("  ", Style::default().bg(surface)),
    ]);
    frame.render_widget(
        Paragraph::new(line),
        Rect::new(area.x, area.y, 8.min(area.width), 1),
    );
}

/// Settings' `Button { height: 1; border: none; padding: 0 1; min-width: 8 }`:
/// the label with one more column on each side (`line-pad`).
pub fn compact_button_width(label: &str) -> u16 {
    (str_width(label) as u16 + 4).max(8)
}

pub fn draw_compact_button(
    frame: &mut Frame,
    area: Rect,
    label: &str,
    focused: bool,
    disabled: bool,
    theme: &Theme,
) {
    let mut base = match focused {
        true => theme.tinted("surface-lighten-2", "foreground", 0.05),
        false => theme.color("surface-lighten-2"),
    };
    let mut color = theme.on_color("text", base);
    if disabled {
        // Measured from Textual: a disabled button shows at about half strength
        // over the page, and its label at 40% over that.
        base = theme.tinted("background", "surface-lighten-2", 0.49);
        color = crate::tui::markdown::tint(color, base, 0.395);
    }
    let mut style = Style::default()
        .fg(color)
        .bg(base)
        .add_modifier(ratatui::style::Modifier::BOLD);
    if focused {
        style = style.add_modifier(ratatui::style::Modifier::REVERSED);
    }
    let line = crate::tui::text::center(
        Line::from(Span::styled(format!(" {label} "), style)),
        area.width,
    )
    .style(Style::default().bg(base));
    frame.render_widget(Paragraph::new(line).style(Style::default().bg(base)), area);
}
