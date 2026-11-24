//! Speaker diarization through the native `nemo-speech` program.
//!
//! The runtime and the model are files, not libraries: a 7 MB binary and a
//! 107 MB GGUF. Paths come from the config, then `LTN_NEMO_BINARY` and
//! `LTN_NEMO_MODEL`, then the model cache. A session never downloads them.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::Deserialize;
use tracing::info;

use crate::adapters::assets::store::CachedModelStore;
use crate::adapters::model_catalog::{NEMOTRON_ID, RUNTIME_ID};
use crate::adapters::paths::Paths;
use crate::domain::model_asset::AssetStatus;
use crate::domain::speaker_turn::SpeakerTurn;
use crate::ports::diarizer::Diarizer;

pub const DEFAULT_PRESET: &str = "v3-offline";

pub struct NemoSpeechDiarizer {
    binary: String,
    model: String,
    device: String,
    preset: String,
    store: Option<Arc<CachedModelStore>>,
}

#[derive(Deserialize)]
struct Output {
    #[serde(default)]
    segments: Vec<SpeakerTurn>,
}

impl NemoSpeechDiarizer {
    /// An empty `binary` or `model` falls back to its environment variable, then the cache.
    pub fn new(
        binary: &str,
        model: &str,
        device: &str,
        store: Option<Arc<CachedModelStore>>,
    ) -> Self {
        let or_env = |value: &str, name: &str| {
            if value.is_empty() {
                std::env::var(name).unwrap_or_default()
            } else {
                value.to_string()
            }
        };
        Self {
            binary: or_env(binary, "LTN_NEMO_BINARY"),
            model: or_env(model, "LTN_NEMO_MODEL"),
            device: device.to_string(),
            preset: DEFAULT_PRESET.to_string(),
            store,
        }
    }

    /// The binary and model paths, or the error with its fix. Preflight calls this.
    pub fn resolve_paths(&self) -> Result<(PathBuf, PathBuf), String> {
        let binary = self.resolve_one(
            &self.binary,
            RUNTIME_ID,
            "nemo-speech binary",
            "LTN_NEMO_BINARY",
        )?;
        let model = self.resolve_one(
            &self.model,
            NEMOTRON_ID,
            "diarization model",
            "LTN_NEMO_MODEL",
        )?;
        Ok((binary, model))
    }

    fn resolve_one(
        &self,
        explicit: &str,
        asset_id: &str,
        label: &str,
        env_var: &str,
    ) -> Result<PathBuf, String> {
        if !explicit.is_empty() {
            let path = match Paths::for_user() {
                Some(paths) => paths.expand_user(explicit),
                None => PathBuf::from(explicit),
            };
            if path.is_file() {
                return Ok(path);
            }
            return Err(format!(
                "{label} not found at {}. Set {env_var} to a valid file or clear the override.",
                path.display()
            ));
        }
        let Some(store) = &self.store else {
            return Err(format!(
                "{label} not found. Set {env_var} to the file path."
            ));
        };
        let state = store.status(asset_id).map_err(|e| e.to_string())?;
        match (state.status, state.path) {
            (AssetStatus::Ready, Some(path)) if !path.is_empty() => Ok(PathBuf::from(path)),
            _ => Err(format!(
                "{label} is not installed. Open Settings > Models to install it."
            )),
        }
    }
}

/// Collect a pipe on a thread, so a full pipe never blocks the child.
fn drain(pipe: Option<impl Read + Send + 'static>) -> std::thread::JoinHandle<String> {
    std::thread::spawn(move || {
        let mut text = String::new();
        if let Some(mut pipe) = pipe {
            let mut bytes = Vec::new();
            let _ = pipe.read_to_end(&mut bytes);
            text = String::from_utf8_lossy(&bytes).into_owned();
        }
        text
    })
}

impl Diarizer for NemoSpeechDiarizer {
    fn diarize(&self, audio_path: &Path, timeout: Duration) -> Result<Vec<SpeakerTurn>, String> {
        let (binary, model) = self.resolve_paths()?;
        if !audio_path.is_file() {
            return Err(format!("audio file not found: {}", audio_path.display()));
        }
        let workdir = tempfile::tempdir().map_err(|e| e.to_string())?;
        let output = workdir.path().join("turns.json");
        info!(
            target: "ltn.diarize",
            "diarizing {} (device={}, preset={})",
            audio_path.display(),
            self.device,
            self.preset
        );
        let mut child = Command::new(&binary)
            .arg("diarize")
            .arg(audio_path)
            .arg("--model")
            .arg(&model)
            .args(["--device", &self.device, "--preset", &self.preset])
            .args(["--format", "json", "--output"])
            .arg(&output)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("could not start {}: {e}", binary.display()))?;
        let stdout = drain(child.stdout.take());
        let stderr = drain(child.stderr.take());
        let deadline = Instant::now() + timeout;
        let status = loop {
            if let Some(status) = child.try_wait().map_err(|e| e.to_string())? {
                break status;
            }
            if Instant::now() >= deadline {
                let _ = child.kill();
                let _ = child.wait();
                return Err(format!(
                    "diarization timed out after {:.0}s",
                    timeout.as_secs_f64()
                ));
            }
            std::thread::sleep(Duration::from_millis(50));
        };
        let (stdout, stderr) = (
            stdout.join().unwrap_or_default(),
            stderr.join().unwrap_or_default(),
        );
        if !status.success() {
            let detail = if stderr.is_empty() { stdout } else { stderr };
            let code = status.code().unwrap_or(-1);
            return Err(format!(
                "nemo-speech failed with code {code}: {}",
                detail.trim()
            ));
        }
        let text = std::fs::read_to_string(&output)
            .map_err(|_| "nemo-speech produced no output file".to_string())?;
        let parsed: Output = serde_json::from_str(&text).map_err(|e| e.to_string())?;
        Ok(parsed.segments)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;

    /// A stand-in binary: a shell script that acts like nemo-speech.
    fn fake_binary(dir: &Path, body: &str) -> PathBuf {
        let path = dir.join("nemo-speech");
        std::fs::write(&path, format!("#!/bin/sh\n{body}\n")).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
        path
    }

    fn setup(body: &str) -> (tempfile::TempDir, NemoSpeechDiarizer, PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let binary = fake_binary(dir.path(), body);
        let model = dir.path().join("model.gguf");
        let audio = dir.path().join("recording.wav");
        std::fs::write(&model, "m").unwrap();
        std::fs::write(&audio, "a").unwrap();
        let diarizer = NemoSpeechDiarizer::new(
            binary.to_str().unwrap(),
            model.to_str().unwrap(),
            "cpu",
            None,
        );
        (dir, diarizer, audio)
    }

    #[test]
    fn reads_the_turns_the_binary_writes() {
        // $12 is the value after --output.
        let (_dir, diarizer, audio) = setup(
            r#"echo '{"segments":[{"start":0.5,"end":2.0,"speaker":1},{"start":2.0,"end":3,"speaker":0}]}' > "${12}""#,
        );
        let turns = diarizer.diarize(&audio, Duration::from_secs(5)).unwrap();
        assert_eq!(turns.len(), 2);
        assert_eq!(
            (turns[0].start, turns[0].end, turns[0].speaker),
            (0.5, 2.0, 1)
        );
    }

    #[test]
    fn a_failure_reports_the_code_and_stderr() {
        let (_dir, diarizer, audio) = setup("echo 'bad model' >&2; exit 3");
        let error = diarizer
            .diarize(&audio, Duration::from_secs(5))
            .unwrap_err();
        assert_eq!(error, "nemo-speech failed with code 3: bad model");
    }

    #[test]
    fn a_stalled_binary_is_killed_at_the_timeout() {
        let (_dir, diarizer, audio) = setup("sleep 30");
        let started = Instant::now();
        let error = diarizer
            .diarize(&audio, Duration::from_millis(300))
            .unwrap_err();
        assert_eq!(error, "diarization timed out after 0s");
        assert!(started.elapsed() < Duration::from_secs(5));
    }

    #[test]
    fn missing_paths_name_the_fix() {
        let diarizer = NemoSpeechDiarizer::new("/nowhere/nemo", "/nowhere/m", "auto", None);
        assert_eq!(
            diarizer.resolve_paths().unwrap_err(),
            "nemo-speech binary not found at /nowhere/nemo. Set LTN_NEMO_BINARY to a valid file or clear the override."
        );
    }
}
