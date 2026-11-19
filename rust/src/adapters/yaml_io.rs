//! YAML that the Python version (PyYAML, YAML 1.1) and this one (YAML 1.2) read the same way.
//!
//! The two specs disagree on plain scalars: PyYAML reads `yes`, `on` and `No` as
//! booleans, and `1e3` as a string. The emitter quotes every string that either
//! spec would read as something else. On the way in, `yaml11_bool` reads the
//! YAML 1.1 words for the boolean settings a user may have written by hand.

use yaml_serde::{Mapping, Value};

const INDENT: usize = 2;

/// Block-style YAML in the layout PyYAML writes: insertion order, lists flush with their key.
pub fn to_yaml(value: &Value) -> String {
    let mut out = String::new();
    match value {
        Value::Mapping(map) if !map.is_empty() => write_mapping(&mut out, map, 0),
        Value::Sequence(items) if !items.is_empty() => write_sequence(&mut out, items, 0),
        other => {
            out.push_str(&inline(other));
            out.push('\n');
        }
    }
    out
}

fn write_mapping(out: &mut String, map: &Mapping, indent: usize) {
    for (index, (key, value)) in map.iter().enumerate() {
        // A list item's first key sits on the dash line, so it gets no indent.
        if index > 0 || !out.ends_with("- ") {
            out.push_str(&" ".repeat(indent));
        }
        out.push_str(&inline(key));
        out.push(':');
        write_value(out, value, indent, indent + INDENT);
    }
}

fn write_sequence(out: &mut String, items: &[Value], indent: usize) {
    for item in items {
        out.push_str(&" ".repeat(indent));
        out.push('-');
        match item {
            Value::Mapping(map) if !map.is_empty() => {
                out.push(' ');
                write_mapping(out, map, indent + INDENT);
            }
            other => write_value(out, other, indent, indent + INDENT),
        }
    }
}

/// The value after `key:` or `-`. `indent` is the parent's column; `child` the nested one.
fn write_value(out: &mut String, value: &Value, indent: usize, child: usize) {
    match value {
        Value::Mapping(map) if !map.is_empty() => {
            out.push('\n');
            write_mapping(out, map, child);
        }
        // PyYAML puts a list under a key at the key's own column.
        Value::Sequence(items) if !items.is_empty() => {
            out.push('\n');
            write_sequence(
                out,
                items,
                if out_is_list_item(out) { child } else { indent },
            );
        }
        Value::String(text) if needs_block(text) => write_block(out, text, child),
        Value::Tagged(tagged) => write_value(out, &tagged.value, indent, child),
        other => {
            out.push(' ');
            out.push_str(&inline(other));
            out.push('\n');
        }
    }
}

/// True when the value being written is a list item (`- `) rather than a key.
fn out_is_list_item(out: &str) -> bool {
    out.trim_end_matches('\n').ends_with('-')
}

fn write_block(out: &mut String, text: &str, indent: usize) {
    let body = text.trim_end_matches('\n');
    let trailing = text.len() - body.len();
    let chomp = match trailing {
        0 => "-",
        1 => "",
        _ => "+",
    };
    let indicator = if body.starts_with(' ') {
        INDENT.to_string()
    } else {
        String::new()
    };
    out.push_str(&format!(" |{indicator}{chomp}\n"));
    let pad = " ".repeat(indent);
    for line in body.split('\n') {
        if !line.is_empty() {
            out.push_str(&pad);
            out.push_str(line);
        }
        out.push('\n');
    }
    // "|+" keeps the extra newlines as empty lines.
    for _ in 1..trailing {
        out.push('\n');
    }
}

/// A multi-line string goes in a literal block, unless it holds a character a block cannot carry.
fn needs_block(text: &str) -> bool {
    text.contains('\n') && !text.chars().any(|c| c != '\n' && needs_escape(c))
}

fn needs_escape(c: char) -> bool {
    (c.is_control() && c != '\t') || c == '\u{feff}'
}

fn inline(value: &Value) -> String {
    match value {
        Value::Null => "null".into(),
        Value::Bool(flag) => flag.to_string(),
        Value::Number(number) => number_text(number),
        Value::String(text) => scalar(text),
        Value::Sequence(_) => "[]".into(),
        Value::Mapping(_) => "{}".into(),
        Value::Tagged(tagged) => inline(&tagged.value),
    }
}

/// Floats keep a decimal point, as Python writes them, so they read back as floats.
fn number_text(number: &yaml_serde::Number) -> String {
    if number.is_i64() || number.is_u64() {
        return number.to_string();
    }
    let float = number.as_f64().unwrap_or_default();
    if float.is_nan() {
        return ".nan".into();
    }
    if float.is_infinite() {
        return if float > 0.0 {
            ".inf".into()
        } else {
            "-.inf".into()
        };
    }
    let text = format!("{float:?}");
    if text.contains(['.', 'e', 'E']) {
        text
    } else {
        format!("{text}.0")
    }
}

/// A string on one line: plain when both specs read it as this string, else quoted.
fn scalar(text: &str) -> String {
    if text.contains('\n') || text.chars().any(needs_escape) {
        return double_quoted(text);
    }
    if is_plain_safe(text) {
        return text.to_string();
    }
    format!("'{}'", text.replace('\'', "''"))
}

fn double_quoted(text: &str) -> String {
    let mut out = String::from("\"");
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            c if needs_escape(c) => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// Words YAML 1.1 reads as a boolean or null, in any case PyYAML accepts.
const SPECIAL_WORDS: &[&str] = &[
    "y", "n", "yes", "no", "true", "false", "on", "off", "null", "~",
];

fn is_plain_safe(text: &str) -> bool {
    let Some(first) = text.chars().next() else {
        return false;
    };
    if SPECIAL_WORDS.contains(&text.to_lowercase().as_str()) || text == "=" || text == "<<" {
        return false;
    }
    // Leading digits, signs and dots may resolve to a number, a date or a time in one of the specs.
    if first.is_ascii_digit() || "-+.?:,[]{}#&*!|>'\"%@`".contains(first) {
        return false;
    }
    if text.starts_with(' ') || text.ends_with(' ') || text.ends_with(':') || text.contains('\t') {
        return false;
    }
    !text.contains(": ") && !text.contains(" #")
}

/// A boolean in the YAML 1.1 words PyYAML accepts, for settings a user wrote by hand.
pub fn yaml11_bool(word: &str) -> Option<bool> {
    match word.to_lowercase().as_str() {
        "y" | "yes" | "true" | "on" => Some(true),
        "n" | "no" | "false" | "off" => Some(false),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn round_trip(value: &Value) -> Value {
        yaml_serde::from_str(&to_yaml(value)).expect("emitted YAML parses")
    }

    fn map(pairs: &[(&str, Value)]) -> Value {
        Value::Mapping(
            pairs
                .iter()
                .map(|(k, v)| (Value::String((*k).into()), v.clone()))
                .collect(),
        )
    }

    #[test]
    fn tricky_strings_survive_a_round_trip() {
        let tricky = [
            "yes",
            "No",
            "on",
            "OFF",
            "y",
            "null",
            "~",
            "",
            " lead",
            "trail ",
            "1.0",
            "1e3",
            "0o17",
            "12:30",
            "-dash",
            "a: b",
            "x #y",
            "key:",
            "'quoted'",
            "\"dq\"",
            "中文 文字",
            "tab\there",
            "bell\u{7}",
            "=",
            "<<",
            "- item",
            "hf://ggerganov/whisper.cpp/ggml-large-v3-turbo-q8_0.bin",
            "~/Documents/LazyTakeNotes",
        ];
        for text in tricky {
            let value = map(&[("v", Value::String(text.into()))]);
            assert_eq!(round_trip(&value), value, "{text:?} -> {}", to_yaml(&value));
        }
    }

    #[test]
    fn multi_line_strings_use_a_literal_block() {
        for text in [
            "a\nb",
            "a\nb\n",
            "a\nb\n\n",
            "  indented\nnext",
            "line\n\n  gap\n",
            "末行\n",
        ] {
            let value = map(&[("t", Value::String(text.into()))]);
            let yaml = to_yaml(&value);
            assert!(yaml.starts_with("t: |"), "{yaml}");
            assert_eq!(round_trip(&value), value, "{text:?} -> {yaml}");
        }
    }

    #[test]
    fn layout_matches_pyyaml() {
        let value: Value = yaml_serde::from_str(
            "digest:\n  model: gpt-oss:20b\n  min_interval: 60.0\nhints:\n- a\n- b\nactions:\n- label: x\n  body: |-\n    l1\n    l2\nempty: []\nnone: null\n",
        )
        .unwrap();
        assert_eq!(
            to_yaml(&value),
            "digest:\n  model: gpt-oss:20b\n  min_interval: 60.0\nhints:\n- a\n- b\nactions:\n- label: x\n  body: |-\n    l1\n    l2\nempty: []\nnone: null\n"
        );
    }

    #[test]
    fn nested_lists_and_maps_round_trip() {
        let value: Value =
            yaml_serde::from_str("a:\n- - 1\n  - 2\n- k: v\n  n:\n  - x\nb:\n  c:\n    d: 1.5\n")
                .unwrap();
        assert_eq!(round_trip(&value), value, "{}", to_yaml(&value));
    }

    #[test]
    fn yaml11_bool_reads_the_pyyaml_words() {
        for (word, expected) in [
            ("yes", Some(true)),
            ("On", Some(true)),
            ("no", Some(false)),
            ("OFF", Some(false)),
            ("maybe", None),
        ] {
            assert_eq!(yaml11_bool(word), expected, "{word}");
        }
    }
}
