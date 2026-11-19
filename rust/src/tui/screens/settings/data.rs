//! What the settings form reads from config.yaml and what Save writes back.

use std::path::Path;

use yaml_serde::{Mapping, Value};

use super::form::Page;
use super::models::{LanguageEditor, ModelChooser};
use super::pages::DEFAULT_DEVICE;
use crate::adapters::config_file::{deep_merge, load_raw};
use crate::adapters::paths::Paths;
use crate::config::{
    InfraConfig, build_infra_config, defaults, drop_removed_language_models,
    language_models_for_file,
};

/// Keys the form fully owns. A blank or removed entry in the form deletes the key.
pub const FORM_OWNED: [&[&str]; 2] = [&["transcription", "models"], &["digest", "max_lines"]];

/// The config as the form opens it.
pub struct Loaded {
    /// The defaults with the user's file merged on top.
    pub raw: Value,
    /// The provider settings from the user's file alone.
    pub infra: InfraConfig,
    /// Why the file could not be read; the form then shows the defaults.
    pub error: String,
}

pub fn load(paths: &Paths, config_path: Option<&Path>) -> Loaded {
    match load_raw(paths, config_path, None) {
        Ok(user) => {
            let mut raw = defaults();
            deep_merge(&mut raw, &user);
            drop_removed_language_models(&mut raw);
            Loaded {
                raw,
                infra: build_infra_config(&user).unwrap_or_default(),
                error: String::new(),
            }
        }
        // Settings is where the user fixes a broken file, so it must still open.
        Err(e) => Loaded {
            raw: defaults(),
            infra: InfraConfig::default(),
            error: e.to_string(),
        },
    }
}

/// Find the page that holds `id`.
fn page_of<'p>(pages: &'p [Page; 5], id: &str) -> &'p Page {
    pages
        .iter()
        .find(|page| page.item(id).is_some())
        .unwrap_or(&pages[0])
}

fn input(pages: &[Page; 5], id: &str) -> String {
    page_of(pages, id).input(id)
}

fn switch(pages: &[Page; 5], id: &str) -> bool {
    page_of(pages, id).switch(id)
}

fn select(pages: &[Page; 5], id: &str) -> String {
    page_of(pages, id).select(id)
}

/// `float(text)` in Python, or 0.0.
pub fn to_float(text: &str) -> f64 {
    parse_float(text).unwrap_or(0.0)
}

/// `int(text)` in Python, or 0.
pub fn to_int(text: &str) -> i64 {
    parse_int(text).unwrap_or(0)
}

/// Python allows `_` between digits: `1_000`.
fn without_digit_underscores(text: &str) -> Option<String> {
    let chars: Vec<char> = text.chars().collect();
    for (index, c) in chars.iter().enumerate() {
        if *c == '_' {
            let before = index.checked_sub(1).and_then(|i| chars.get(i));
            let after = chars.get(index + 1);
            if !before.is_some_and(char::is_ascii_digit) || !after.is_some_and(char::is_ascii_digit)
            {
                return None;
            }
        }
    }
    Some(text.replace('_', ""))
}

fn parse_float(text: &str) -> Option<f64> {
    let text = without_digit_underscores(text.trim())?;
    let lower = text.to_ascii_lowercase();
    let bare = lower.trim_start_matches(['+', '-']);
    // Rust also reads "infinity" and "nan" as Python does, but not "1." alone.
    if bare.is_empty() {
        return None;
    }
    text.parse::<f64>().ok().or_else(|| {
        text.strip_suffix('.')
            .and_then(|whole| whole.parse::<f64>().ok())
    })
}

fn parse_int(text: &str) -> Option<i64> {
    without_digit_underscores(text.trim())?.parse::<i64>().ok()
}

/// The custom path when set, else the chosen row, else the saved model.
fn model_value(pages: &mut [Page; 5], raw: &Value) -> String {
    let custom = input(pages, "cfg-trans-custom-path").trim().to_string();
    if !custom.is_empty() {
        return custom;
    }
    let chosen = pages[0]
        .custom::<ModelChooser>("models-chooser")
        .map(|chooser| chooser.selected_uri())
        .unwrap_or_default();
    let value = match chosen.is_empty() {
        true => raw
            .get("transcription")
            .and_then(|t| t.get("model"))
            .map(|model| super::pages::python_str(Some(model)))
            .unwrap_or_default(),
        false => chosen,
    };
    // A saved custom path the user cleared must not come back from the file.
    match Path::new(&value).is_absolute() {
        true => String::new(),
        false => value,
    }
}

/// `_model_error`: no model, or a custom file that does not exist.
pub fn model_error(model: &str) -> Option<String> {
    if model.is_empty() {
        return Some("Choose a speech model, or enter the path of a model file.".into());
    }
    if Path::new(model).is_absolute() && !Path::new(model).is_file() {
        return Some(format!("Model file not found: {model}"));
    }
    None
}

fn map(pairs: Vec<(&str, Value)>) -> Value {
    Value::Mapping(
        pairs
            .into_iter()
            .map(|(key, value)| (Value::from(key), value))
            .collect::<Mapping>(),
    )
}

/// Python's `str.splitlines()` line breaks.
fn split_lines(text: &str) -> Vec<&str> {
    const BREAKS: [char; 10] = [
        '\n', '\r', '\x0b', '\x0c', '\x1c', '\x1d', '\x1e', '\u{85}', '\u{2028}', '\u{2029}',
    ];
    let mut lines = Vec::new();
    let mut rest = text;
    while let Some(at) = rest.find(BREAKS) {
        lines.push(&rest[..at]);
        let skip = if rest[at..].starts_with("\r\n") {
            2
        } else {
            rest[at..].chars().next().map_or(1, char::len_utf8)
        };
        rest = &rest[at + skip..];
    }
    if !rest.is_empty() {
        lines.push(rest);
    }
    lines
}

/// Read every field and build the config data, as `_collect_form_data`.
pub fn collect(pages: &mut [Page; 5], raw: &Value, clear_api_key: bool) -> Value {
    let diarization = raw.get("diarization");
    let saved = |key: &str| {
        diarization
            .and_then(|d| d.get(key))
            .map(|v| super::pages::python_str(Some(v)))
            .unwrap_or_default()
    };
    let mut openai = vec![(
        "base_url",
        Value::from(input(pages, "cfg-openai-base-url").trim()),
    )];
    let api_key = input(pages, "cfg-openai-api-key").trim().to_string();
    if clear_api_key {
        openai.push(("api_key", Value::Null));
    } else if !api_key.is_empty() {
        openai.push(("api_key", Value::from(api_key)));
    }
    let language = pages[0]
        .custom::<LanguageEditor>("models-language")
        .map(|editor| editor.values())
        .unwrap_or_default();
    let device = select(pages, "cfg-audio-device");
    let device = match device == DEFAULT_DEVICE {
        true => Value::Null,
        false => Value::from(device),
    };
    let model = model_value(pages, raw);
    let float = |id: &str| Value::from(to_float(&input(pages, id)));
    let int = |id: &str| Value::from(to_int(&input(pages, id)));
    let mut digest = vec![
        (
            "model",
            Value::from(input(pages, "cfg-digest-model").trim()),
        ),
        ("min_lines", int("cfg-digest-min-lines")),
        ("min_interval", float("cfg-digest-min-interval")),
        ("compact_token_threshold", int("cfg-digest-compact")),
    ];
    let max_lines = input(pages, "cfg-digest-max-lines").trim().to_string();
    if !max_lines.is_empty() {
        digest.push(("max_lines", Value::from(to_int(&max_lines))));
    }
    let hints_text = page_of(pages, "cfg-recognition-hints").text_area("cfg-recognition-hints");
    let hints: Vec<Value> = split_lines(hints_text.trim())
        .into_iter()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(Value::from)
        .collect();
    map(vec![
        (
            "llm_provider",
            Value::from(select(pages, "cfg-llm-provider")),
        ),
        (
            "transcription_provider",
            Value::from(select(pages, "cfg-trans-provider")),
        ),
        ("theme", Value::from(select(pages, "cfg-theme"))),
        (
            "show_builtin_templates",
            Value::from(switch(pages, "cfg-show-builtin-templates")),
        ),
        (
            "ollama",
            map(vec![(
                "host",
                Value::from(input(pages, "cfg-ollama-host").trim()),
            )]),
        ),
        ("openai", map(openai)),
        (
            "transcription",
            map(vec![
                ("model", Value::from(model)),
                (
                    "models",
                    Value::Mapping(language_models_for_file(&language)),
                ),
                ("input_device", device),
                ("chunk_duration", float("cfg-trans-chunk")),
                ("overlap", float("cfg-trans-overlap")),
                ("silence_threshold", float("cfg-trans-silence")),
                ("pause_duration", float("cfg-trans-pause")),
            ]),
        ),
        ("digest", map(digest)),
        (
            "diarization",
            map(vec![
                (
                    "enabled",
                    Value::from(switch(pages, "cfg-diarization-enabled")),
                ),
                ("binary", Value::from(saved("binary"))),
                ("model", Value::from(saved("model"))),
                (
                    "device",
                    Value::from(select(pages, "cfg-diarization-device")),
                ),
                ("interval_s", float("cfg-diarization-interval")),
            ]),
        ),
        (
            "interactive",
            map(vec![(
                "model",
                Value::from(input(pages, "cfg-interactive-model").trim()),
            )]),
        ),
        (
            "output",
            map(vec![
                (
                    "directory",
                    Value::from(input(pages, "cfg-output-dir").trim()),
                ),
                (
                    "save_audio",
                    Value::from(switch(pages, "cfg-output-save-audio")),
                ),
                (
                    "save_notes_history",
                    Value::from(switch(pages, "cfg-output-save-notes-history")),
                ),
                (
                    "save_context",
                    Value::from(switch(pages, "cfg-output-save-context")),
                ),
                (
                    "save_debug_log",
                    Value::from(switch(pages, "cfg-output-save-debug-log")),
                ),
                (
                    "auto_label",
                    Value::from(switch(pages, "cfg-output-auto-label")),
                ),
            ]),
        ),
        ("recognition_hints", Value::Sequence(hints)),
    ])
}

enum Number {
    Float(f64),
    Int(i64),
}

/// `_numeric_errors`: a field that is not a number, or out of range. Blank passes.
pub fn numeric_errors(pages: &[Page; 5]) -> Vec<String> {
    let checks: [(&str, &str, Number, Option<Number>); 9] = [
        (
            "cfg-trans-chunk",
            "Chunk duration",
            Number::Float(0.0),
            None,
        ),
        ("cfg-trans-overlap", "Overlap", Number::Float(0.0), None),
        (
            "cfg-trans-silence",
            "Silence threshold",
            Number::Float(0.0),
            Some(Number::Float(1.0)),
        ),
        (
            "cfg-trans-pause",
            "Pause duration",
            Number::Float(0.0),
            None,
        ),
        ("cfg-digest-min-lines", "After lines", Number::Int(1), None),
        (
            "cfg-digest-min-interval",
            "Wait seconds",
            Number::Float(0.0),
            None,
        ),
        (
            "cfg-digest-compact",
            "Compaction threshold",
            Number::Int(1),
            None,
        ),
        ("cfg-digest-max-lines", "Force after", Number::Int(1), None),
        (
            "cfg-diarization-interval",
            "Update every",
            Number::Float(0.0),
            None,
        ),
    ];
    let shown = |number: &Number| match number {
        Number::Float(value) => crate::adapters::pyrepr::float_repr(*value),
        Number::Int(value) => value.to_string(),
    };
    let mut errors = Vec::new();
    for (id, label, minimum, maximum) in checks {
        let text = input(pages, id).trim().to_string();
        if text.is_empty() {
            continue;
        }
        let value = match minimum {
            Number::Float(_) => parse_float(&text),
            Number::Int(_) => parse_int(&text).map(|v| v as f64),
        };
        let Some(value) = value else {
            errors.push(format!("{label} must be a number."));
            continue;
        };
        let low = match minimum {
            Number::Float(v) => v,
            Number::Int(v) => v as f64,
        };
        if value < low {
            errors.push(format!("{label} must be at least {}.", shown(&minimum)));
        }
        if let Some(maximum) = maximum {
            let high = match maximum {
                Number::Float(v) => v,
                Number::Int(v) => v as f64,
            };
            if value > high {
                errors.push(format!("{label} must be at most {}.", shown(&maximum)));
            }
        }
    }
    errors
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numbers_read_as_python_reads_them() {
        assert_eq!(to_float(" 2.5 "), 2.5);
        assert_eq!(to_float("1_000"), 1000.0);
        assert_eq!(to_float("1__0"), 0.0);
        assert_eq!(to_float("x"), 0.0);
        assert_eq!(to_float("1."), 1.0);
        assert_eq!(to_float("1e3"), 1000.0);
        assert_eq!(to_int("+5"), 5);
        assert_eq!(to_int("2.0"), 0);
        assert_eq!(to_int("1_0"), 10);
    }

    #[test]
    fn lines_split_where_python_splits_them() {
        assert_eq!(split_lines("a\r\nb\rc\u{2028}d"), vec!["a", "b", "c", "d"]);
        assert_eq!(split_lines("a\n\nb\n"), vec!["a", "", "b"]);
    }
}
