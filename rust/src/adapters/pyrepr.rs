//! Python's `repr()` of a decoded JSON value. The SDK error texts embed it.

use serde_json::Value;

use crate::usecases::pyformat::py_repr;

pub fn py_repr_json(value: &Value) -> String {
    match value {
        Value::Null => "None".into(),
        Value::Bool(true) => "True".into(),
        Value::Bool(false) => "False".into(),
        Value::Number(number) => number.to_string(),
        Value::String(text) => py_repr(text),
        Value::Array(items) => format!(
            "[{}]",
            items
                .iter()
                .map(py_repr_json)
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Value::Object(map) => {
            let pairs: Vec<String> = map
                .iter()
                .map(|(k, v)| format!("{}: {}", py_repr(k), py_repr_json(v)))
                .collect();
            format!("{{{}}}", pairs.join(", "))
        }
    }
}

/// Python's `str()`: a string as is, anything else as its repr.
pub fn py_str_json(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        other => py_repr_json(other),
    }
}

/// Python's `json.dumps(value)` with its defaults: `", "` and `": "` between
/// items, and every non-ASCII character escaped as `\uXXXX`.
pub fn py_dumps<T: serde::Serialize + ?Sized>(value: &T) -> serde_json::Result<String> {
    let mut out = Vec::new();
    let mut serializer = serde_json::Serializer::with_formatter(&mut out, PythonDefaults);
    value.serialize(&mut serializer)?;
    Ok(String::from_utf8(out).expect("the formatter writes ASCII"))
}

struct PythonDefaults;

impl serde_json::ser::Formatter for PythonDefaults {
    fn begin_array_value<W: ?Sized + std::io::Write>(
        &mut self,
        writer: &mut W,
        first: bool,
    ) -> std::io::Result<()> {
        if first {
            Ok(())
        } else {
            writer.write_all(b", ")
        }
    }

    fn begin_object_key<W: ?Sized + std::io::Write>(
        &mut self,
        writer: &mut W,
        first: bool,
    ) -> std::io::Result<()> {
        if first {
            Ok(())
        } else {
            writer.write_all(b", ")
        }
    }

    fn begin_object_value<W: ?Sized + std::io::Write>(
        &mut self,
        writer: &mut W,
    ) -> std::io::Result<()> {
        writer.write_all(b": ")
    }

    fn write_string_fragment<W: ?Sized + std::io::Write>(
        &mut self,
        writer: &mut W,
        fragment: &str,
    ) -> std::io::Result<()> {
        for character in fragment.chars() {
            if character.is_ascii() {
                writer.write_all(&[character as u8])?;
                continue;
            }
            let mut units = [0u16; 2];
            for unit in character.encode_utf16(&mut units) {
                write!(writer, "\\u{unit:04x}")?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn matches_python_repr() {
        let body = json!({"error": {"message": "Incorrect API key provided", "param": null, "ok": true, "n": [1, 2.5]}});
        assert_eq!(
            py_repr_json(&body),
            "{'error': {'message': 'Incorrect API key provided', 'param': None, 'ok': True, 'n': [1, 2.5]}}"
        );
        assert_eq!(
            py_repr_json(&json!({"error": "model 'x' not found"})),
            r#"{'error': "model 'x' not found"}"#
        );
        assert_eq!(py_str_json(&json!("plain")), "plain");
    }
}
