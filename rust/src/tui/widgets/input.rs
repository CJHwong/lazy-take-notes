//! A one-line text input with Textual's editing keys and its `tall` border.

use crossterm::event::KeyEvent;
use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::Style;
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;

use crate::tui::keys::typed_char;
use crate::tui::text::str_width;
use crate::tui::theme::Theme;

#[derive(Default)]
pub struct Input {
    chars: Vec<char>,
    /// The cursor, as a character index.
    cursor: usize,
    pub placeholder: String,
    pub max_length: Option<usize>,
    /// The theme variable of the parent's background, under the border's outer side.
    /// None means the screen background.
    pub parent_background: Option<&'static str>,
}

impl Input {
    pub fn new(placeholder: &str) -> Self {
        Self {
            placeholder: placeholder.into(),
            ..Self::default()
        }
    }

    pub fn value(&self) -> String {
        self.chars.iter().collect()
    }

    pub fn set_value(&mut self, value: &str) {
        self.chars = value.chars().collect();
        self.cursor = self.chars.len();
    }

    fn insert(&mut self, text: &str) {
        for character in text.chars().filter(|c| !c.is_control()) {
            if self.max_length.is_some_and(|max| self.chars.len() >= max) {
                break;
            }
            self.chars.insert(self.cursor, character);
            self.cursor += 1;
        }
    }

    pub fn paste(&mut self, text: &str) {
        self.insert(text.lines().next().unwrap_or(""));
    }

    fn word_start(&self) -> usize {
        let mut at = self.cursor;
        while at > 0 && self.chars[at - 1] == ' ' {
            at -= 1;
        }
        while at > 0 && self.chars[at - 1] != ' ' {
            at -= 1;
        }
        at
    }

    fn word_end(&self) -> usize {
        let mut at = self.cursor;
        while at < self.chars.len() && self.chars[at] == ' ' {
            at += 1;
        }
        while at < self.chars.len() && self.chars[at] != ' ' {
            at += 1;
        }
        at
    }

    /// Handle an editing key. Returns true when the input used it.
    pub fn key(&mut self, key: &KeyEvent, name: &str) -> bool {
        match name {
            "left" => self.cursor = self.cursor.saturating_sub(1),
            "right" => self.cursor = (self.cursor + 1).min(self.chars.len()),
            "ctrl+left" => self.cursor = self.word_start(),
            "ctrl+right" => self.cursor = self.word_end(),
            "home" | "ctrl+a" => self.cursor = 0,
            "end" | "ctrl+e" => self.cursor = self.chars.len(),
            "backspace" => {
                if self.cursor > 0 {
                    self.cursor -= 1;
                    self.chars.remove(self.cursor);
                }
            }
            "delete" | "ctrl+d" => {
                if self.cursor < self.chars.len() {
                    self.chars.remove(self.cursor);
                }
            }
            "ctrl+w" => {
                let start = self.word_start();
                self.chars.drain(start..self.cursor);
                self.cursor = start;
            }
            "ctrl+u" => {
                self.chars.drain(..self.cursor);
                self.cursor = 0;
            }
            "ctrl+k" => {
                self.chars.truncate(self.cursor);
            }
            _ => match typed_char(key) {
                Some(character) => self.insert(&character.to_string()),
                None => return false,
            },
        }
        true
    }

    /// Draw the input in its 3-row `tall` border.
    pub fn render(&self, frame: &mut Frame, area: Rect, focused: bool, theme: &Theme) {
        if area.height < 3 {
            self.render_row(frame, area, focused, theme);
            return;
        }
        let border = theme.color(if focused { "border" } else { "border-blurred" });
        let inner = self.background(focused, theme);
        let outer = theme.color(self.parent_background.unwrap_or("background"));
        let width = usize::from(area.width.saturating_sub(2));
        let edges = crate::tui::widgets::tall_edges(border, inner, outer);
        let rows = [
            ("▔", edges.fill),
            (" ", Style::default().bg(inner)),
            ("▁", edges.fill),
        ];
        for (offset, (fill, style)) in rows.into_iter().enumerate() {
            let line = Line::from(vec![
                Span::styled("▊", edges.left),
                Span::styled(fill.repeat(width), style),
                Span::styled("▎", edges.right),
            ]);
            frame.render_widget(
                Paragraph::new(line),
                Rect::new(area.x, area.y + offset as u16, area.width, 1),
            );
        }
        // `padding: 0 2` inside the border.
        let text_area = Rect::new(area.x + 3, area.y + 1, area.width.saturating_sub(6), 1);
        self.render_row(frame, text_area, focused, theme);
    }

    /// The value (or placeholder) and the cursor, scrolled to keep the cursor shown.
    /// `$surface`, tinted 5% toward the foreground while focused (`:focus`).
    fn background(&self, focused: bool, theme: &Theme) -> ratatui::style::Color {
        match focused {
            true => theme.tinted("surface", "foreground", 0.05),
            false => theme.color("surface"),
        }
    }

    pub fn render_row(&self, frame: &mut Frame, area: Rect, focused: bool, theme: &Theme) {
        let surface = self.background(focused, theme);
        let text_style = Style::default()
            .fg(theme.on_color("foreground", surface))
            .bg(surface);
        let placeholder_color = theme.on_color("text-disabled", surface);
        let cursor_style = Style::default()
            .fg(theme.on("input-cursor-foreground", "surface"))
            .bg(theme.on("input-cursor-background", "surface"));
        let width = usize::from(area.width);
        if self.chars.is_empty() {
            let mut spans = Vec::new();
            let mut placeholder = self.placeholder.chars();
            if focused {
                let first = placeholder
                    .next()
                    .map_or(" ".to_string(), |c| c.to_string());
                spans.push(Span::styled(first, cursor_style));
            }
            spans.push(Span::styled(
                placeholder.collect::<String>(),
                Style::default().fg(placeholder_color).bg(surface),
            ));
            frame.render_widget(Paragraph::new(Line::from(spans)).style(text_style), area);
            return;
        }
        // Start far enough left that the cursor fits.
        let mut start = 0;
        let cursor_column =
            |from: usize| str_width(&self.chars[from..self.cursor].iter().collect::<String>());
        while start < self.cursor && cursor_column(start) + 1 > width {
            start += 1;
        }
        let before: String = self.chars[start..self.cursor].iter().collect();
        let at: String = self
            .chars
            .get(self.cursor)
            .map_or(" ".into(), |c| c.to_string());
        let after: String = self.chars.iter().skip(self.cursor + 1).collect();
        let mut spans = vec![Span::styled(before, text_style)];
        spans.push(Span::styled(
            at,
            if focused { cursor_style } else { text_style },
        ));
        spans.push(Span::styled(after, text_style));
        frame.render_widget(Paragraph::new(Line::from(spans)).style(text_style), area);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossterm::event::{KeyCode, KeyModifiers};

    fn type_text(input: &mut Input, text: &str) {
        for character in text.chars() {
            let key = KeyEvent::new(KeyCode::Char(character), KeyModifiers::NONE);
            input.key(&key, &character.to_string());
        }
    }

    #[test]
    fn editing_keys_work_like_textual() {
        let none = KeyEvent::new(KeyCode::Null, KeyModifiers::NONE);
        let mut input = Input::new("Filter...");
        type_text(&mut input, "hello world");
        input.key(&none, "ctrl+w");
        assert_eq!(input.value(), "hello ");
        input.key(&none, "home");
        type_text(&mut input, "會議 ");
        assert_eq!(input.value(), "會議 hello ");
        input.key(&none, "ctrl+k");
        assert_eq!(input.value(), "會議 ");
        input.key(&none, "ctrl+u");
        input.paste("a\nb");
        assert_eq!(input.value(), "a");
        assert!(!input.key(&KeyEvent::new(KeyCode::Up, KeyModifiers::NONE), "up"));
    }
}
