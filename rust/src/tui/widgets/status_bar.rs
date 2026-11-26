//! The bottom bar: the session's status line, drawn in the terminal.

use std::time::Instant;

use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;

pub use crate::app::session::status_line::{StatusLine as StatusBar, rms_to_char};
use crate::tui::text::str_width;
use crate::tui::theme::Theme;

impl StatusBar {
    /// The bar's rows as plain text. `width` is the bar's content width, inside its padding.
    pub fn text(&self, width: u16, now: Instant) -> Vec<String> {
        self.lines(width, now)
            .iter()
            .map(|line| {
                line.spans
                    .iter()
                    .map(|span| span.content.as_ref())
                    .collect()
            })
            .collect()
    }

    pub fn height(&self) -> u16 {
        if self.quick_action_hints.is_empty() {
            1
        } else {
            2
        }
    }

    fn lines(&self, width: u16, now: Instant) -> Vec<Line<'static>> {
        let content_width = usize::from(width).saturating_sub(2);
        let alert = Style::default()
            .fg(Color::White)
            .bg(Color::Red)
            .add_modifier(Modifier::BOLD);
        let mut spans = Vec::new();
        let mut used = 0;
        for (index, (text, is_alert)) in self.parts(now).into_iter().enumerate() {
            if index > 0 {
                spans.push(Span::raw(" │ "));
                used += 3;
            }
            used += str_width(&text);
            spans.push(if is_alert {
                Span::styled(text, alert)
            } else {
                Span::raw(text)
            });
        }
        let hints = &self.keybinding_hints;
        if !hints.is_empty() {
            let gap = content_width as i64 - used as i64 - str_width(hints) as i64;
            if gap >= 2 {
                spans.push(Span::raw(format!("{}{hints}", " ".repeat(gap as usize))));
            }
        }
        let bottom = Line::from(spans);
        if self.quick_action_hints.is_empty() {
            return vec![bottom];
        }
        let quick = &self.quick_action_hints;
        let gap = content_width.saturating_sub(str_width(quick));
        vec![Line::from(format!("{}{quick}", " ".repeat(gap))), bottom]
    }

    pub fn render(&self, frame: &mut Frame, area: Rect, theme: &Theme) {
        let style = Style::default()
            .bg(theme.color("surface"))
            .fg(theme.on("text", "surface"));
        frame.render_widget(Paragraph::new("").style(style), area);
        let inner = Rect::new(
            area.x + 1,
            area.y,
            area.width.saturating_sub(2),
            area.height,
        );
        let lines = self.lines(inner.width, Instant::now());
        frame.render_widget(Paragraph::new(lines).style(style), inner);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_view_bar_reads_like_python() {
        let mut bar = StatusBar::default();
        bar.mode_label = "View".into();
        bar.keybinding_hints = "[c] copy  [o] open  [Tab] switch  [q] back".into();
        bar.set_stopped(true);
        let text = bar.text(100, Instant::now());
        assert_eq!(text.len(), 1);
        assert!(
            text[0].starts_with("View │ ■ Stopped │ buf 0/15 │ 00:00:00 "),
            "{}",
            text[0]
        );
        assert!(text[0].ends_with("[c] copy  [o] open  [Tab] switch  [q] back"));
        assert_eq!(str_width(&text[0]), 98);
    }
}
