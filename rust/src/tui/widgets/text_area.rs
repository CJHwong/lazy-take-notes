//! A multi-line text box with Textual's `TextArea` editing keys, soft wrap and
//! the `tall` border. No selection, undo or clipboard keys.

use crossterm::event::KeyEvent;
use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::Style;
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;
use unicode_width::UnicodeWidthChar;

use crate::tui::keys::typed_char;
use crate::tui::theme::Theme;

pub struct TextArea {
    lines: Vec<Vec<char>>,
    row: usize,
    column: usize,
    /// The first visual row shown.
    scroll: usize,
    /// Rows the box showed last time; page keys move by this.
    page: usize,
    pub read_only: bool,
    blink: crate::tui::widgets::Blink,
}

impl Default for TextArea {
    fn default() -> Self {
        Self {
            lines: vec![Vec::new()],
            row: 0,
            column: 0,
            scroll: 0,
            page: 1,
            read_only: false,
            blink: crate::tui::widgets::Blink::default(),
        }
    }
}

fn char_width(character: char) -> usize {
    character.width().unwrap_or(0)
}

/// Where each visual row of `line` starts, wrapping at spaces within `width` columns.
fn row_starts(line: &[char], width: usize) -> Vec<usize> {
    let width = width.max(1);
    let mut starts = vec![0];
    let mut start = 0;
    let mut used = 0;
    let mut last_space: Option<usize> = None;
    let mut index = 0;
    while index < line.len() {
        let character = line[index];
        let cells = char_width(character);
        if character != ' ' && used + cells > width && index > start {
            // Break after the last space on this row, else fold the word.
            let next = match last_space {
                Some(space) if space + 1 > start => space + 1,
                _ => index,
            };
            starts.push(next);
            start = next;
            used = line[start..index].iter().map(|c| char_width(*c)).sum();
            last_space = None;
            continue;
        }
        if character == ' ' {
            last_space = Some(index);
        }
        used += cells;
        index += 1;
    }
    starts
}

impl TextArea {
    pub fn text(&self) -> String {
        self.lines
            .iter()
            .map(|line| line.iter().collect::<String>())
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Replace the text; the cursor goes to the start, as `TextArea.text = ...` does.
    pub fn set_text(&mut self, text: &str) {
        self.lines = text
            .split('\n')
            .map(|line| line.chars().collect())
            .collect();
        self.row = 0;
        self.column = 0;
        self.scroll = 0;
    }

    pub fn line_count(&self) -> usize {
        self.lines.len()
    }

    fn line(&self) -> &Vec<char> {
        &self.lines[self.row]
    }

    fn insert(&mut self, text: &str) {
        let text = text.replace("\r\n", "\n").replace('\r', "\n");
        for character in text.chars() {
            if character == '\n' {
                self.newline();
            } else if !character.is_control() || character == '\t' {
                self.lines[self.row].insert(self.column, character);
                self.column += 1;
            }
        }
    }

    fn newline(&mut self) {
        let rest = self.lines[self.row].split_off(self.column);
        self.lines.insert(self.row + 1, rest);
        self.row += 1;
        self.column = 0;
    }

    pub fn paste(&mut self, text: &str) {
        self.blink.restart();
        if !self.read_only {
            self.insert(text);
        }
    }

    fn word_left(&self) -> (usize, usize) {
        if self.column == 0 {
            return match self.row {
                0 => (0, 0),
                row => (row - 1, self.lines[row - 1].len()),
            };
        }
        let line = self.line();
        let mut at = self.column;
        while at > 0 && line[at - 1] == ' ' {
            at -= 1;
        }
        while at > 0 && line[at - 1] != ' ' {
            at -= 1;
        }
        (self.row, at)
    }

    fn word_right(&self) -> (usize, usize) {
        let line = self.line();
        if self.column == line.len() {
            return match self.row + 1 < self.lines.len() {
                true => (self.row + 1, 0),
                false => (self.row, self.column),
            };
        }
        let mut at = self.column;
        while at < line.len() && line[at] == ' ' {
            at += 1;
        }
        while at < line.len() && line[at] != ' ' {
            at += 1;
        }
        (self.row, at)
    }

    /// Remove the text between two positions, the first one before the second.
    fn delete_range(&mut self, from: (usize, usize), to: (usize, usize)) {
        if from == to {
            return;
        }
        let tail = self.lines[to.0].split_off(to.1);
        self.lines[from.0].truncate(from.1);
        self.lines[from.0].extend(tail);
        self.lines.drain(from.0 + 1..=to.0);
        (self.row, self.column) = from;
    }

    fn move_vertically(&mut self, rows: isize) {
        let last = self.lines.len() as isize - 1;
        self.row = (self.row as isize + rows).clamp(0, last) as usize;
        self.column = self.column.min(self.line().len());
    }

    fn move_key(&mut self, name: &str) -> bool {
        let end = self.line().len();
        match name {
            "up" => self.move_vertically(-1),
            "down" => self.move_vertically(1),
            "pageup" => self.move_vertically(-(self.page as isize)),
            "pagedown" => self.move_vertically(self.page as isize),
            "left" if self.column == 0 && self.row > 0 => {
                self.row -= 1;
                self.column = self.line().len();
            }
            "left" => self.column = self.column.saturating_sub(1),
            "right" if self.column == end && self.row + 1 < self.lines.len() => {
                (self.row, self.column) = (self.row + 1, 0);
            }
            "right" => self.column = (self.column + 1).min(end),
            "ctrl+left" => (self.row, self.column) = self.word_left(),
            "ctrl+right" => (self.row, self.column) = self.word_right(),
            "home" | "ctrl+a" => self.column = 0,
            "end" | "ctrl+e" => self.column = end,
            _ => return false,
        }
        true
    }

    fn edit_key(&mut self, key: &KeyEvent, name: &str) -> bool {
        let here = (self.row, self.column);
        let end = (self.row, self.line().len());
        let previous = match (self.row, self.column) {
            (0, 0) => here,
            (row, 0) => (row - 1, self.lines[row - 1].len()),
            (row, column) => (row, column - 1),
        };
        let next = match here {
            _ if self.column < self.line().len() => (self.row, self.column + 1),
            _ if self.row + 1 < self.lines.len() => (self.row + 1, 0),
            _ => here,
        };
        match name {
            "enter" => self.newline(),
            "backspace" => self.delete_range(previous, here),
            "delete" | "ctrl+d" => self.delete_range(here, next),
            "ctrl+w" => self.delete_range(self.word_left(), here),
            "ctrl+f" => self.delete_range(here, self.word_right()),
            "ctrl+u" => self.delete_range((self.row, 0), here),
            "ctrl+k" if here == end => self.delete_range(here, next),
            "ctrl+k" => self.delete_range(here, end),
            "ctrl+shift+k" => self.delete_line(),
            _ => match typed_char(key) {
                Some(character) => self.insert(&character.to_string()),
                None => return false,
            },
        }
        true
    }

    fn delete_line(&mut self) {
        if self.lines.len() == 1 {
            self.lines[0].clear();
        } else {
            self.lines.remove(self.row);
            self.row = self.row.min(self.lines.len() - 1);
        }
        self.column = self.column.min(self.line().len());
    }

    /// Textual's `check_consume_key`: an editable box takes printable keys
    /// before the screen's priority bindings. A read-only box takes none.
    pub fn consumes(&self, key: &KeyEvent) -> bool {
        !self.read_only && typed_char(key).is_some_and(|c| !c.is_control())
    }

    /// Handle a key. True when the box used it. A read-only box only moves the cursor.
    pub fn key(&mut self, key: &KeyEvent, name: &str) -> bool {
        self.blink.restart();
        if self.move_key(name) {
            return true;
        }
        !self.read_only && self.edit_key(key, name)
    }

    /// Draw the box: a `tall` border in `border_var` with `title`, and `padding: 0 1`.
    pub fn render(
        &mut self,
        frame: &mut Frame,
        area: Rect,
        focused: bool,
        title: &str,
        border_var: &str,
        theme: &Theme,
    ) {
        if area.height < 3 || area.width < 4 {
            return;
        }
        // "primary" means the normal look: primary while focused, quiet otherwise.
        // Any other color (an error) shows as it is.
        let border = match border_var {
            "primary" => crate::tui::widgets::border_color(focused, theme),
            other => theme.color(other),
        };
        let inner =
            crate::tui::widgets::rounded(frame, area, title, border, theme.color("surface"), theme);
        let inner = Rect::new(
            inner.x + 1,
            inner.y,
            inner.width.saturating_sub(2),
            inner.height,
        );
        self.render_text(frame, inner, focused, theme);
    }

    fn render_text(&mut self, frame: &mut Frame, area: Rect, focused: bool, theme: &Theme) {
        let width = usize::from(area.width);
        let height = usize::from(area.height);
        self.page = height.max(1);
        let text_style = Style::default()
            .fg(theme.on("foreground", "surface"))
            .bg(theme.color("surface"));
        // Textual marks the cursor line of an editable area, focused or not.
        let line_style = match self.read_only {
            false => text_style.bg(theme.on("boost", "surface")),
            true => text_style,
        };
        let cursor_style = Style::default()
            .fg(theme.color("input-cursor-foreground"))
            .bg(theme.color("input-cursor-background"));
        // (logical row, start, end) for every visual row.
        let mut rows: Vec<(usize, usize, usize)> = Vec::new();
        let mut cursor_visual = 0;
        for (index, line) in self.lines.iter().enumerate() {
            let starts = row_starts(line, width);
            for (part, start) in starts.iter().enumerate() {
                let end = starts.get(part + 1).copied().unwrap_or(line.len());
                let last = part + 1 == starts.len();
                let holds_cursor = self.column >= *start && (self.column < end || last);
                if index == self.row && holds_cursor {
                    cursor_visual = rows.len();
                }
                rows.push((index, *start, end));
            }
        }
        if cursor_visual < self.scroll {
            self.scroll = cursor_visual;
        } else if cursor_visual >= self.scroll + height {
            self.scroll = cursor_visual + 1 - height;
        }
        let shown = rows.iter().enumerate().skip(self.scroll).take(height);
        let lines: Vec<Line<'static>> = shown
            .map(|(visual, (row, start, end))| {
                let style = if *row == self.row {
                    line_style
                } else {
                    text_style
                };
                let chars = &self.lines[*row][*start..*end];
                if !(focused && self.blink.visible() && visual == cursor_visual) {
                    let text: String = chars.iter().collect();
                    let pad = width.saturating_sub(text.chars().map(char_width).sum());
                    return Line::from(vec![
                        Span::styled(text, style),
                        Span::styled(" ".repeat(pad), style),
                    ]);
                }
                let at = self.column - start;
                let before: String = chars[..at].iter().collect();
                let under = chars.get(at).map_or(" ".to_string(), |c| c.to_string());
                let after: String = chars.iter().skip(at + 1).collect();
                let used: usize = chars.iter().map(|c| char_width(*c)).sum::<usize>()
                    + usize::from(at == chars.len());
                Line::from(vec![
                    Span::styled(before, style),
                    Span::styled(under, cursor_style),
                    Span::styled(after, style),
                    Span::styled(" ".repeat(width.saturating_sub(used)), style),
                ])
            })
            .collect();
        frame.render_widget(Paragraph::new(lines).style(text_style), area);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossterm::event::{KeyCode, KeyModifiers};

    fn press(area: &mut TextArea, name: &str) -> bool {
        let code = match name {
            "enter" => KeyCode::Enter,
            name if name.chars().count() == 1 => KeyCode::Char(name.chars().next().unwrap()),
            _ => KeyCode::Null,
        };
        area.key(&KeyEvent::new(code, KeyModifiers::NONE), name)
    }

    fn type_text(area: &mut TextArea, text: &str) {
        for character in text.chars() {
            press(area, &character.to_string());
        }
    }

    #[test]
    fn typing_and_editing_keys_work_like_textual() {
        let mut area = TextArea::default();
        type_text(&mut area, "aqz");
        press(&mut area, "enter");
        type_text(&mut area, "會議 notes");
        assert_eq!(area.text(), "aqz\n會議 notes");
        press(&mut area, "ctrl+w");
        assert_eq!(area.text(), "aqz\n會議 ");
        press(&mut area, "ctrl+u");
        press(&mut area, "backspace");
        assert_eq!(area.text(), "aqz");
        press(&mut area, "home");
        press(&mut area, "ctrl+k");
        assert_eq!(area.text(), "");
        area.paste("one\r\ntwo");
        assert_eq!(area.text(), "one\ntwo");
        assert!(!press(&mut area, "tab"), "tab moves the focus");
        assert!(!press(&mut area, "escape"));
    }

    #[test]
    fn a_read_only_box_takes_keys_but_keeps_its_text() {
        let mut area = TextArea::default();
        type_text(&mut area, "kept");
        area.read_only = true;
        assert!(
            !press(&mut area, "q"),
            "the screen's q quits from a read-only box"
        );
        area.paste("x");
        assert_eq!(area.text(), "kept");
    }

    #[test]
    fn rows_wrap_at_spaces_and_fold_long_words() {
        let line: Vec<char> = "one two three".chars().collect();
        assert_eq!(row_starts(&line, 8), [0, 8]);
        let cjk: Vec<char> = "會議記錄".chars().collect();
        assert_eq!(row_starts(&cjk, 4), [0, 2]);
    }
}
