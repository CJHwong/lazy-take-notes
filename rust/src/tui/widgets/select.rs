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

    /// The closed box, 3 rows: a rounded border around the value and `▾`.
    /// `outer` is the theme variable behind the box.
    pub fn render(&self, frame: &mut Frame, area: Rect, focused: bool, outer: &str, theme: &Theme) {
        if area.height < 3 || area.width < 6 {
            return;
        }
        let background = theme.color(outer);
        let inner = crate::tui::widgets::rounded(
            frame,
            area,
            "",
            crate::tui::widgets::border_color(focused || self.expanded, theme),
            background,
            theme,
        );
        let label = self
            .options
            .get(self.index)
            .map(|(label, _)| label.as_str())
            .unwrap_or("");
        let text_width = usize::from(inner.width).saturating_sub(4);
        let shown = crate::tui::text::truncate(label, text_width);
        let pad = text_width.saturating_sub(str_width(&shown));
        let text = Style::default()
            .fg(theme.on_color("foreground", background))
            .bg(background);
        let arrow = Style::default().fg(theme.color("primary")).bg(background);
        let line = Line::from(vec![
            Span::styled(" ", text),
            Span::styled(shown, text),
            Span::styled(" ".repeat(pad), text),
            Span::styled(if self.expanded { " ▴ " } else { " ▾ " }, arrow),
        ]);
        frame.render_widget(
            Paragraph::new(line),
            Rect::new(inner.x, inner.y, inner.width, 1),
        );
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
        frame.render_widget(ratatui::widgets::Clear, area);
        let background = theme.color("surface");
        let inner = crate::tui::widgets::rounded(
            frame,
            area,
            "",
            theme.color("primary"),
            background,
            theme,
        );
        let rows: Vec<Line<'static>> = self
            .options
            .iter()
            .map(|(label, _)| Line::from(label.clone()))
            .collect();
        let _ = outer;
        self.list
            .render(frame, inner, &rows, true, background, theme);
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
    use crate::tui::widgets::button::{Variant, draw_button, draw_disabled_button};
    match disabled {
        true => draw_disabled_button(frame, area, label, Variant::Default, theme),
        false => draw_button(frame, area, label, Variant::Default, focused, theme),
    }
}
