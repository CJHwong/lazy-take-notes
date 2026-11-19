//! The subset of Python's `str.format` that templates use: `{name}`, `{{` and `}}`.
//!
//! Templates written for the Python version must render the same text, and a broken
//! template must fail with the same message. A field with a conversion (`!r`), a
//! format spec (`:>5`), an attribute (`.x`) or an index (`[0]`) is a format error
//! here. No built-in or known user template uses one.

use std::fmt;

/// Python's exception class and message for a failed format.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FormatError {
    Key(String),
    Value(String),
    Index(String),
}

impl FormatError {
    /// The Python exception class name, as the digest error shows it.
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Key(_) => "KeyError",
            Self::Value(_) => "ValueError",
            Self::Index(_) => "IndexError",
        }
    }
}

impl fmt::Display for FormatError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Key(name) => f.write_str(&py_repr(name)),
            Self::Value(message) | Self::Index(message) => f.write_str(message),
        }
    }
}

/// One piece of a parsed format string.
#[derive(Debug, PartialEq, Eq)]
enum Piece<'a> {
    Literal(&'a str),
    /// The whole text between the braces.
    Field(&'a str),
}

/// Split a format string the way Python's `MarkupIterator` does, stopping at the first error.
///
/// Returns the pieces before the error too: the validator collects the field
/// names that come before a malformed brace.
fn parse(template: &str) -> (Vec<Piece<'_>>, Option<FormatError>) {
    let bytes = template.as_bytes();
    let mut pieces = Vec::new();
    let mut literal_start = 0;
    let mut index = 0;
    while index < bytes.len() {
        let brace = bytes[index];
        if brace != b'{' && brace != b'}' {
            index += 1;
            continue;
        }
        let doubled = bytes.get(index + 1) == Some(&brace);
        if doubled {
            pieces.push(Piece::Literal(&template[literal_start..=index]));
            index += 2;
            literal_start = index;
            continue;
        }
        if brace == b'}' {
            return (pieces, Some(single_brace('}')));
        }
        if index + 1 == bytes.len() {
            return (pieces, Some(single_brace('{')));
        }
        let Some(close) = matching_close(bytes, index + 1) else {
            return (
                pieces,
                Some(FormatError::Value(
                    "expected '}' before end of string".into(),
                )),
            );
        };
        if literal_start < index {
            pieces.push(Piece::Literal(&template[literal_start..index]));
        }
        pieces.push(Piece::Field(&template[index + 1..close]));
        index = close + 1;
        literal_start = index;
    }
    if literal_start < bytes.len() {
        pieces.push(Piece::Literal(&template[literal_start..]));
    }
    (pieces, None)
}

fn single_brace(brace: char) -> FormatError {
    FormatError::Value(format!("Single '{brace}' encountered in format string"))
}

/// Index of the `}` that closes a field opened just before `start`, counting nested braces.
fn matching_close(bytes: &[u8], start: usize) -> Option<usize> {
    let mut depth = 1;
    for (offset, byte) in bytes[start..].iter().enumerate() {
        match byte {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(start + offset);
                }
            }
            _ => {}
        }
    }
    None
}

/// The field name of a field's text: everything before `!` or `:`.
fn field_name(field: &str) -> &str {
    field.find(['!', ':']).map_or(field, |end| &field[..end])
}

/// Root names a format string refers to, in order, as `string.Formatter().parse` sees them.
/// Parsing stops silently at a malformed brace; `format` reports that error.
pub fn field_names(template: &str) -> Vec<String> {
    let (pieces, _) = parse(template);
    let mut names = Vec::new();
    for piece in pieces {
        let Piece::Field(field) = piece else { continue };
        let name = field_name(field);
        let root = name.split(['.', '[']).next().unwrap_or_default();
        if !root.is_empty() && !names.iter().any(|known| known == root) {
            names.push(root.to_string());
        }
    }
    names
}

/// Render `template` with named values, like Python's `template.format(**values)`.
pub fn format(template: &str, values: &[(&str, &str)]) -> Result<String, FormatError> {
    let (pieces, error) = parse(template);
    let mut out = String::with_capacity(template.len());
    let mut auto_index = 0;
    for piece in pieces {
        match piece {
            Piece::Literal(text) => out.push_str(text),
            Piece::Field(field) => out.push_str(lookup(field, values, &mut auto_index)?),
        }
    }
    match error {
        Some(error) => Err(error),
        None => Ok(out),
    }
}

fn lookup<'v>(
    field: &str,
    values: &[(&str, &'v str)],
    auto_index: &mut usize,
) -> Result<&'v str, FormatError> {
    let name = field_name(field);
    if name.is_empty() || name.bytes().all(|byte| byte.is_ascii_digit()) {
        let index = if name.is_empty() {
            *auto_index
        } else {
            name.parse().unwrap_or(usize::MAX)
        };
        *auto_index += 1;
        return Err(FormatError::Index(format!(
            "Replacement index {index} out of range for positional args tuple"
        )));
    }
    if name.len() != field.len() || name.contains(['.', '[']) {
        return Err(FormatError::Value(format!(
            "unsupported format field {}",
            py_repr(field)
        )));
    }
    values
        .iter()
        .find(|(key, _)| *key == name)
        .map(|(_, value)| *value)
        .ok_or_else(|| FormatError::Key(name.to_string()))
}

/// Python's `repr()` of a str: single quotes unless the text holds one and no double quote.
pub fn py_repr(text: &str) -> String {
    let quote = if text.contains('\'') && !text.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut out = String::with_capacity(text.len() + 2);
    out.push(quote);
    for character in text.chars() {
        match character {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c if (c as u32) < 0x20 || (0x7f..0xa0).contains(&(c as u32)) => {
                out.push_str(&format!("\\x{:02x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const VALUES: &[(&str, &str)] = &[("line_count", "3"), ("new_lines", "a\nb")];

    #[test]
    fn fills_named_fields_and_unescapes_doubled_braces() {
        let out = format("{{x}} {line_count} lines:\n{new_lines}}}", VALUES).unwrap();
        assert_eq!(out, "{x} 3 lines:\na\nb}");
    }

    #[test]
    fn a_value_is_not_parsed_again() {
        assert_eq!(
            format("{new_lines}", &[("new_lines", "{line_count}")]).unwrap(),
            "{line_count}"
        );
    }

    #[test]
    fn errors_match_python() {
        let cases = [
            ("{bogus}", "KeyError", "'bogus'"),
            (
                "{line_count} {",
                "ValueError",
                "Single '{' encountered in format string",
            ),
            (
                "a } b",
                "ValueError",
                "Single '}' encountered in format string",
            ),
            (
                "{line_count} {unclosed",
                "ValueError",
                "expected '}' before end of string",
            ),
            (
                "{}",
                "IndexError",
                "Replacement index 0 out of range for positional args tuple",
            ),
            (
                "{2}",
                "IndexError",
                "Replacement index 2 out of range for positional args tuple",
            ),
        ];
        for (template, kind, message) in cases {
            let error = format(template, VALUES).unwrap_err();
            assert_eq!(
                (error.kind(), error.to_string().as_str()),
                (kind, message),
                "{template}"
            );
        }
    }

    #[test]
    fn the_first_error_in_reading_order_wins() {
        assert_eq!(
            format("{bogus} {", VALUES).unwrap_err(),
            FormatError::Key("bogus".into())
        );
    }

    #[test]
    fn conversion_and_spec_are_rejected() {
        assert!(matches!(
            format("{line_count!r}", VALUES),
            Err(FormatError::Value(_))
        ));
        assert!(matches!(
            format("{line_count:>5}", VALUES),
            Err(FormatError::Value(_))
        ));
        assert!(matches!(
            format("{line_count.real}", VALUES),
            Err(FormatError::Value(_))
        ));
    }

    #[test]
    fn field_names_keep_the_names_before_a_parse_error() {
        assert_eq!(field_names("{line_count} {unclosed"), ["line_count"]);
        assert_eq!(field_names("{a.b} {c[0]} {d!r} {} {a}"), ["a", "c", "d"]);
    }

    #[test]
    fn non_ascii_text_passes_through() {
        assert_eq!(format("新的 {line_count} 行", VALUES).unwrap(), "新的 3 行");
    }

    #[test]
    fn repr_picks_the_quote_like_python() {
        assert_eq!(py_repr("plain"), "'plain'");
        assert_eq!(py_repr("O'Brien"), "\"O'Brien\"");
        assert_eq!(py_repr("say \"hi\""), "'say \"hi\"'");
        assert_eq!(py_repr("both ' \""), "'both \\' \"'");
        assert_eq!(py_repr("tab\there\\"), "'tab\\there\\\\'");
        assert_eq!(py_repr("貨代"), "'貨代'");
    }
}
