//! Configuration schema. Defaults live in the config loader, not here.

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TranscriptionConfig {
    pub model: String,
    #[serde(default)]
    pub models: IndexMap<String, String>,
    #[serde(default)]
    pub input_device: Option<String>,
    pub chunk_duration: f64,
    pub overlap: f64,
    pub silence_threshold: f64,
    pub pause_duration: f64,
}

impl TranscriptionConfig {
    /// The model for a locale: the full key, then the primary subtag, then the default.
    pub fn model_for_locale(&self, locale: &str) -> &str {
        let key = locale.to_lowercase();
        if let Some(model) = self.models.get(&key) {
            return model;
        }
        let prefix = key.split('-').next().unwrap_or_default();
        self.models.get(prefix).map_or(&self.model, |model| model)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DigestConfig {
    pub model: String,
    pub min_lines: usize,
    pub min_interval: f64,
    pub compact_token_threshold: u64,
    /// Force-trigger threshold. None means twice `min_lines`.
    #[serde(default)]
    pub max_lines: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InteractiveConfig {
    pub model: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OutputConfig {
    pub directory: String,
    pub save_audio: bool,
    pub save_notes_history: bool,
    pub save_context: bool,
    pub save_debug_log: bool,
    pub auto_label: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct DiarizationConfig {
    pub enabled: bool,
    pub binary: String,
    pub model: String,
    pub device: String,
    pub interval_s: f64,
}

impl Default for DiarizationConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            binary: String::new(),
            model: String::new(),
            device: "auto".into(),
            interval_s: 60.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AppConfig {
    pub transcription: TranscriptionConfig,
    pub digest: DigestConfig,
    pub interactive: InteractiveConfig,
    pub output: OutputConfig,
    #[serde(default)]
    pub diarization: DiarizationConfig,
    #[serde(default)]
    pub recognition_hints: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn transcription(models: &[(&str, &str)]) -> TranscriptionConfig {
        TranscriptionConfig {
            model: "large-v3-turbo-q8_0".into(),
            models: models
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
            input_device: None,
            chunk_duration: 25.0,
            overlap: 1.0,
            silence_threshold: 0.01,
            pause_duration: 1.5,
        }
    }

    #[test]
    fn model_for_locale_prefers_the_full_key_then_the_subtag_then_the_default() {
        let config = transcription(&[("zh-tw", "breeze"), ("zh", "zh-generic")]);
        assert_eq!(config.model_for_locale("zh-TW"), "breeze");
        assert_eq!(config.model_for_locale("zh-CN"), "zh-generic");
        assert_eq!(config.model_for_locale("en"), "large-v3-turbo-q8_0");
    }

    #[test]
    fn diarization_defaults_when_absent() {
        let config: DiarizationConfig = serde_json::from_str("{}").unwrap();
        assert_eq!(config, DiarizationConfig::default());
        assert_eq!(config.device, "auto");
    }
}
