//! The app settings: defaults, the user's file merged on top, and the provider settings.

use indexmap::IndexMap;
use serde::Deserialize;
use yaml_serde::{Mapping, Value};

use crate::adapters::config_file::deep_merge;
use crate::adapters::model_catalog::expand_model_alias;
use crate::adapters::yaml_io::yaml11_bool;
use crate::domain::config::AppConfig;

const DEFAULTS: &str = r#"
transcription:
  model: hf://ggerganov/whisper.cpp/ggml-large-v3-turbo-q8_0.bin
  models:
    zh: hf://alan314159/Breeze-ASR-25-whispercpp/ggml-model-q8_0.bin
    zh-min-nan: hf://phate334/Breeze-ASR-26-GGML/ggml-model-q8_0.bin
  chunk_duration: 25.0
  overlap: 1.0
  silence_threshold: 0.01
  pause_duration: 1.5
  input_device: null
digest:
  model: gpt-oss:20b
  min_lines: 15
  min_interval: 60.0
  compact_token_threshold: 100000
interactive:
  model: gpt-oss:20b
output:
  # Absolute, so a session recorded from any folder is found from any other.
  directory: ~/Documents/LazyTakeNotes
  save_audio: true
  save_notes_history: true
  save_context: true
  save_debug_log: false
  auto_label: true
recognition_hints: []
"#;

/// Boolean settings a user may have written in YAML 1.1 words (`yes`, `off`).
const BOOL_SETTINGS: &[&[&str]] = &[
    &["output", "save_audio"],
    &["output", "save_notes_history"],
    &["output", "save_context"],
    &["output", "save_debug_log"],
    &["output", "auto_label"],
    &["diarization", "enabled"],
    &["show_builtin_templates"],
];

pub fn defaults() -> Value {
    yaml_serde::from_str(DEFAULTS).expect("the defaults are valid YAML")
}

/// The built-in language overrides, by locale.
pub fn default_language_models() -> IndexMap<String, String> {
    let defaults = defaults();
    let models = defaults["transcription"]["models"]
        .as_mapping()
        .cloned()
        .unwrap_or_default();
    models
        .into_iter()
        .filter_map(|(k, v)| Some((k.as_str()?.to_string(), v.as_str()?.to_string())))
        .collect()
}

/// Merge the user's settings over the defaults, then validate.
///
/// A language override set to null removes a built-in one. Short model aliases
/// become `hf://` URIs, so the user always sees where a model comes from.
pub fn build_app_config(raw: &Value) -> Result<AppConfig, String> {
    let mut merged = defaults();
    deep_merge(&mut merged, raw);
    drop_removed_language_models(&mut merged);
    expand_transcription_aliases(&mut merged);
    read_yaml11_bools(&mut merged);
    yaml_serde::from_value(merged).map_err(|e| e.to_string())
}

/// Remove the language overrides set to null. The built-in overrides merge into
/// every config, so the file sets one to null to remove it.
pub fn drop_removed_language_models(merged: &mut Value) {
    if let Some(models) = merged
        .get_mut("transcription")
        .and_then(|t| t.get_mut("models"))
        .and_then(Value::as_mapping_mut)
    {
        models.retain(|_, name| !name.is_null());
    }
}

/// The language overrides to save. A removed built-in one is saved as null, so it stays removed.
pub fn language_models_for_file(models: &IndexMap<String, String>) -> Mapping {
    let mut out: Mapping = models
        .iter()
        .map(|(k, v)| (Value::from(k.as_str()), Value::from(v.as_str())))
        .collect();
    for locale in default_language_models().keys() {
        if !models.contains_key(locale) {
            out.insert(Value::from(locale.as_str()), Value::Null);
        }
    }
    out
}

fn expand_transcription_aliases(merged: &mut Value) {
    let Some(transcription) = merged.get_mut("transcription") else {
        return;
    };
    if let Some(Value::String(model)) = transcription.get_mut("model") {
        *model = expand_model_alias(model);
    }
    if let Some(models) = transcription
        .get_mut("models")
        .and_then(Value::as_mapping_mut)
    {
        for (_, name) in models.iter_mut() {
            if let Value::String(text) = name {
                *text = expand_model_alias(text);
            }
        }
    }
}

fn read_yaml11_bools(merged: &mut Value) {
    for path in BOOL_SETTINGS {
        let mut node = Some(&mut *merged);
        for key in *path {
            node = node.and_then(|n| n.get_mut(*key));
        }
        if let Some(value) = node
            && let Some(flag) = value.as_str().and_then(yaml11_bool)
        {
            *value = Value::Bool(flag);
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, serde::Serialize)]
#[serde(default)]
pub struct OllamaConfig {
    pub host: String,
}

impl Default for OllamaConfig {
    fn default() -> Self {
        Self {
            host: "http://localhost:11434".into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, serde::Serialize)]
#[serde(default)]
pub struct OpenAiConfig {
    /// None reads `OPENAI_API_KEY` from the environment.
    pub api_key: Option<String>,
    pub base_url: String,
}

impl Default for OpenAiConfig {
    fn default() -> Self {
        Self {
            api_key: None,
            base_url: "https://api.openai.com/v1".into(),
        }
    }
}

pub const DEFAULT_THEME: &str = "textual-dark";

/// Provider settings, outside the domain. Unknown keys are ignored here and kept in the file.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, serde::Serialize)]
#[serde(default)]
pub struct InfraConfig {
    pub llm_provider: String,
    pub transcription_provider: String,
    pub theme: String,
    pub show_builtin_templates: bool,
    pub ollama: OllamaConfig,
    pub openai: OpenAiConfig,
    /// Every other key, for plugin providers: Python's `InfraConfig` keeps them
    /// (`extra='allow'`) and a plugin reads its own from `model_extra`.
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

impl Default for InfraConfig {
    fn default() -> Self {
        Self {
            llm_provider: "ollama".into(),
            transcription_provider: "whisper-cpp".into(),
            theme: DEFAULT_THEME.into(),
            show_builtin_templates: true,
            ollama: OllamaConfig::default(),
            openai: OpenAiConfig::default(),
            extra: serde_json::Map::new(),
        }
    }
}

pub fn build_infra_config(raw: &Value) -> Result<InfraConfig, String> {
    let mut raw = raw.clone();
    read_yaml11_bools(&mut raw);
    yaml_serde::from_value(raw).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn yaml(text: &str) -> Value {
        yaml_serde::from_str(text).unwrap()
    }

    #[test]
    fn an_empty_file_gives_the_defaults() {
        let config = build_app_config(&yaml("{}")).unwrap();
        assert_eq!(config.digest.min_lines, 15);
        assert_eq!(config.output.directory, "~/Documents/LazyTakeNotes");
        assert_eq!(config.transcription.models.len(), 2);
        assert_eq!(config.diarization.interval_s, 60.0);
    }

    #[test]
    fn user_values_merge_and_null_removes_a_built_in_language() {
        let raw = yaml(
            "transcription:\n  models:\n    ja: large-v3\n    zh: null\ndigest:\n  min_lines: 3\n",
        );
        let config = build_app_config(&raw).unwrap();
        let locales: Vec<&str> = config
            .transcription
            .models
            .keys()
            .map(String::as_str)
            .collect();
        assert_eq!(locales, ["zh-min-nan", "ja"]);
        assert_eq!(
            config.transcription.models["ja"],
            "hf://ggerganov/whisper.cpp/ggml-large-v3.bin"
        );
        assert_eq!(
            (config.digest.min_lines, config.digest.min_interval),
            (3, 60.0)
        );
    }

    #[test]
    fn yaml_1_1_booleans_are_read() {
        let config =
            build_app_config(&yaml("output:\n  save_audio: no\n  auto_label: 'off'\n")).unwrap();
        assert!(!config.output.save_audio && !config.output.auto_label);
        let infra = build_infra_config(&yaml("show_builtin_templates: No\n")).unwrap();
        assert!(!infra.show_builtin_templates);
    }

    #[test]
    fn a_removed_built_in_language_is_saved_as_null() {
        let models: IndexMap<String, String> = [
            ("ja".to_string(), "x".to_string()),
            ("zh".to_string(), "y".to_string()),
        ]
        .into();
        let saved = language_models_for_file(&models);
        assert_eq!(
            Value::Mapping(saved),
            yaml("ja: x\nzh: y\nzh-min-nan: null\n")
        );
    }

    #[test]
    fn infra_defaults_and_plugin_keys() {
        let infra = build_infra_config(&yaml(
            "llm_provider: openai\nmy_plugin: {a: 1}\nopenai:\n  base_url: http://x\n",
        ))
        .unwrap();
        assert_eq!(infra.llm_provider, "openai");
        assert_eq!(
            infra.openai,
            OpenAiConfig {
                api_key: None,
                base_url: "http://x".into()
            }
        );
        assert_eq!(infra.ollama.host, "http://localhost:11434");
    }
}
