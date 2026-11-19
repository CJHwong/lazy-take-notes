//! The composition root: config in, the concrete objects a session uses out.

use std::path::{Path, PathBuf};
use std::sync::{Arc, LazyLock};

use regex::Regex;
use serde_json::Value as JsonValue;
use yaml_serde::{Mapping, Value};

use crate::adapters::audio::coreaudio_tap::CoreAudioTapSource;
use crate::adapters::audio::microphone::{MicrophoneSource, play_test_tone};
use crate::adapters::audio::mixed::{MAX_MIC_GAIN, MixTiming, MixedAudioSource};
use crate::adapters::config_file::load_raw;
use crate::adapters::llm::ollama::OllamaLlmClient;
use crate::adapters::llm::openai::OpenAiCompatClient;
use crate::adapters::paths::Paths;
use crate::adapters::pyrepr::py_repr_json;
use crate::config::{InfraConfig, build_app_config, build_infra_config};
use crate::domain::config::AppConfig;
use crate::ports::llm::LlmClient;

const LLM_PROVIDERS: [&str; 2] = ["ollama", "openai"];
const TRANSCRIPTION_PROVIDERS: [&str; 1] = ["whisper-cpp"];

pub struct Loaded {
    pub config: AppConfig,
    pub infra: InfraConfig,
    pub paths: Paths,
}

/// Read the config, with `--output-dir` merged on top. The error is the text after "Error: ".
pub fn load_config(config_path: Option<&Path>, output_dir: Option<&str>) -> Result<Loaded, String> {
    let paths = Paths::for_user().ok_or("cannot find the home folder")?;
    let overrides = output_dir.map(|dir| {
        let mut output = Mapping::new();
        output.insert("directory".into(), dir.into());
        let mut top = Mapping::new();
        top.insert("output".into(), Value::Mapping(output));
        Value::Mapping(top)
    });
    let raw = load_raw(&paths, config_path, overrides.as_ref()).map_err(|e| e.to_string())?;
    Ok(Loaded {
        config: build_app_config(&raw)?,
        infra: build_infra_config(&raw)?,
        paths,
    })
}

/// The base output folder, with `~` expanded.
pub fn resolve_base_dir(paths: &Paths, output_dir: Option<&str>, config: &AppConfig) -> PathBuf {
    paths.expand_user(output_dir.unwrap_or(&config.output.directory))
}

static UNSAFE_LABEL: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"[^\w\-]").expect("valid regex"));

/// Create a timestamped session folder under `base_dir`.
pub fn make_session_dir(base_dir: &Path, label: Option<&str>) -> std::io::Result<PathBuf> {
    let stamp = chrono::Local::now().format("%Y-%m-%d_%H%M%S").to_string();
    let name = match label.filter(|label| !label.is_empty()) {
        Some(label) => format!("{stamp}_{}", UNSAFE_LABEL.replace_all(label, "_")),
        None => stamp,
    };
    let session_dir = base_dir.join(name);
    std::fs::create_dir_all(&session_dir)?;
    Ok(session_dir)
}

fn unknown(provider: &str, available: &[&str]) -> String {
    let quoted = py_repr_json(&JsonValue::String(provider.into()));
    format!(
        "Unknown provider {quoted}. Available: {}",
        available.join(", ")
    )
}

pub fn build_llm_client(infra: &InfraConfig) -> Result<Arc<dyn LlmClient>, String> {
    match infra.llm_provider.as_str() {
        "openai" => Ok(Arc::new(OpenAiCompatClient::new(
            infra.openai.api_key.clone(),
            &infra.openai.base_url,
        ))),
        "ollama" => Ok(Arc::new(OllamaLlmClient::new(&infra.ollama.host))),
        other => Err(unknown(other, &LLM_PROVIDERS)),
    }
}

/// Only whisper.cpp is built in; provider plugins are not ported.
pub fn check_transcription_provider(infra: &InfraConfig) -> Result<(), String> {
    if TRANSCRIPTION_PROVIDERS.contains(&infra.transcription_provider.as_str()) {
        return Ok(());
    }
    Err(unknown(
        &infra.transcription_provider,
        &TRANSCRIPTION_PROVIDERS,
    ))
}

/// The microphone and the system audio, mixed. macOS only for now.
pub fn build_mixed_source(config: &AppConfig) -> Arc<MixedAudioSource> {
    let mic = Arc::new(MicrophoneSource::new(
        config.transcription.input_device.clone(),
    ));
    let system = Arc::new(CoreAudioTapSource::new(false));
    // The Core Audio tap sends exact zeros when nothing plays, and it captures
    // this tone even with the output muted.
    Arc::new(MixedAudioSource::new(
        mic,
        system,
        config.transcription.silence_threshold,
        MAX_MIC_GAIN,
        Some(Box::new(play_test_tone)),
        MixTiming::default(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_label_keeps_word_characters_and_dashes() {
        let dir = tempfile::tempdir().unwrap();
        let made = make_session_dir(dir.path(), Some("sprint-review 週會/1")).unwrap();
        let name = made.file_name().unwrap().to_str().unwrap();
        assert!(name.ends_with("_sprint-review_週會_1"), "{name}");
        assert!(made.is_dir());
    }

    #[test]
    fn an_unknown_provider_lists_the_built_in_ones() {
        assert_eq!(
            unknown("claude", &LLM_PROVIDERS),
            "Unknown provider 'claude'. Available: ollama, openai"
        );
    }
}
