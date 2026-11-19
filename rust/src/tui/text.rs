//! Line wrapping by display width, as Rich wraps: at spaces, and a word wider
//! than the line is folded at the edge. CJK text has no spaces, so it folds.
//! Wide characters count two columns.

use ratatui::style::Style;
use ratatui::text::{Line, Span};
use unicode_width::UnicodeWidthChar;

fn char_width(character: char) -> usize {
    character.width().unwrap_or(0)
}

pub fn str_width(text: &str) -> usize {
    text.chars().map(char_width).sum()
}

/// A word with the spaces after it, as styled characters.
type Word = Vec<(char, Style)>;

fn words(line: &Line<'_>) -> Vec<Word> {
    let mut all = Vec::new();
    let mut word: Word = Vec::new();
    let mut in_spaces = false;
    for span in &line.spans {
        let style = line.style.patch(span.style);
        for character in span.content.chars() {
            let space = character == ' ';
            if !space && in_spaces {
                all.push(std::mem::take(&mut word));
            }
            in_spaces = space;
            word.push((character, style));
        }
    }
    if !word.is_empty() {
        all.push(word);
    }
    all
}

/// Builds wrapped rows of styled characters.
struct Rows {
    width: usize,
    rows: Vec<Word>,
    used: usize,
}

impl Rows {
    fn new_row(&mut self) {
        self.rows.push(Vec::new());
        self.used = 0;
    }

    fn push_char(&mut self, character: char, style: Style) {
        let width = char_width(character);
        if self.used + width > self.width && self.used > 0 {
            self.new_row();
        }
        self.rows
            .last_mut()
            .expect("a row")
            .push((character, style));
        self.used += width;
    }

    fn push_word(&mut self, word: Word) {
        let text_width: usize = word
            .iter()
            .filter(|(c, _)| *c != ' ')
            .map(|(c, _)| char_width(*c))
            .sum();
        if self.used > 0 && self.used + text_width > self.width {
            self.new_row();
        }
        for (character, style) in word {
            // Spaces at the end of a full row are dropped, not wrapped.
            if character == ' ' && self.used >= self.width {
                continue;
            }
            self.push_char(character, style);
        }
    }
}

fn into_line(row: Word, base: &Line<'_>) -> Line<'static> {
    let mut spans: Vec<Span<'static>> = Vec::new();
    for (character, style) in row {
        match spans.last_mut() {
            Some(span) if span.style == style => span.content.to_mut().push(character),
            _ => spans.push(Span::styled(character.to_string(), style)),
        }
    }
    let mut line = Line::from(spans);
    line.alignment = base.alignment;
    line
}

/// One line wrapped to `width` columns. An empty line stays one row.
pub fn wrap_line(line: &Line<'_>, width: u16) -> Vec<Line<'static>> {
    let mut rows = Rows {
        width: usize::from(width.max(1)),
        rows: vec![Vec::new()],
        used: 0,
    };
    for word in words(line) {
        rows.push_word(word);
    }
    rows.rows
        .into_iter()
        .map(|row| into_line(row, line))
        .collect()
}

pub fn wrap_lines(lines: &[Line<'_>], width: u16) -> Vec<Line<'static>> {
    lines
        .iter()
        .flat_map(|line| wrap_line(line, width))
        .collect()
}

pub fn wrapped_height(lines: &[Line<'_>], width: u16) -> u16 {
    u16::try_from(wrap_lines(lines, width).len()).unwrap_or(u16::MAX)
}

/// `line` centered in `width` columns as Rich centers: `(width - len) // 2` spaces first.
pub fn center(line: Line<'static>, width: u16) -> Line<'static> {
    let pad = usize::from(width).saturating_sub(line.width()) / 2;
    let mut spans = vec![Span::raw(" ".repeat(pad))];
    spans.extend(line.spans);
    Line::from(spans).style(line.style)
}

/// Cut `text` to at most `width` columns.
pub fn truncate(text: &str, width: usize) -> String {
    let mut used = 0;
    let mut out = String::new();
    for character in text.chars() {
        let char_cols = char_width(character);
        if used + char_cols > width {
            break;
        }
        used += char_cols;
        out.push(character);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plain(lines: &[Line<'_>]) -> Vec<String> {
        lines
            .iter()
            .map(|line| line.spans.iter().map(|s| s.content.as_ref()).collect())
            .collect()
    }

    #[test]
    fn words_wrap_at_spaces_and_a_long_word_folds() {
        let lines = wrap_line(&Line::from("the quick brown fox"), 10);
        assert_eq!(plain(&lines), ["the quick ", "brown fox"]);
        let lines = wrap_line(&Line::from("abcdefghijkl xy"), 5);
        assert_eq!(plain(&lines), ["abcde", "fghij", "kl xy"]);
    }

    #[test]
    fn cjk_folds_by_display_width() {
        // Expected values from rich.text.Text.wrap.
        let lines = wrap_line(&Line::from("[00:00:02] 大家腦中想到的幾乎都是美國"), 16);
        assert_eq!(
            plain(&lines),
            ["[00:00:02] ", "大家腦中想到的幾", "乎都是美國"]
        );
        for line in &lines {
            assert!(line.width() <= 16, "{line:?}");
        }
        assert_eq!(wrap_line(&Line::from(""), 5).len(), 1);
    }

    #[test]
    fn truncate_counts_columns() {
        assert_eq!(truncate("中文ab", 5), "中文a");
        assert_eq!(str_width("中a"), 3);
    }
}
