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
