//! Settings without any screen (the logic of Python's `ConfigApp`): load and
//! save config.yaml, the unsaved-changes check, the model assets, the AI test
//! and model list, and the audio probe. A front end owns the fields and lends
//! them through `FormValues`; it shows the state this runtime keeps.

pub mod form;
pub mod probe;
pub mod rows;

#[cfg(test)]
mod tests;

use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use indexmap::IndexMap;
use tracing::{error, info};
use yaml_serde::Value;

use self::form::{DEFAULT_DEVICE, model_error, numeric_errors};
pub use self::form::{FORM_OWNED, FormValues, Loaded, collect, load};
use self::probe::{ProbeEvent, Source};
use crate::adapters::assets::download::DownloadError;
use crate::adapters::assets::store::{CachedModelStore, StoreError};
use crate::adapters::config_file::{config_file_path, write_config};
use crate::adapters::model_catalog::{NEMOTRON_ID, RUNTIME_ID};
use crate::adapters::paths::Paths;
use crate::app::download_progress::{describe, megabytes};
use crate::app::preflight::{Check, Status};
use crate::app::session::Severity;
use crate::app::tasks::{Payload, Tasks};
use crate::config::{InfraConfig, build_app_config, build_infra_config};
use crate::domain::model_asset::{
    AssetGroup, AssetProgress, AssetState, AssetStatus, RemovalEstimate,
};

const MEGABYTE: u64 = 1024 * 1024;
/// The speaker runtime and model, installed and removed together.
pub const SPEAKER_IDS: [&str; 2] = [RUNTIME_ID, NEMOTRON_ID];
/// The id a Remove of the speaker pair goes by.
pub const SPEAKER: &str = "speaker";
/// The front end's default notice time.
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(5);
/// Bars each level meter keeps.
const METER_BARS: usize = 12;

/// What Settings needs from its front end.
pub trait SettingsIo {
    /// Show `text` to the user for `timeout`.
    fn notify(&mut self, text: String, severity: Severity, timeout: Duration);
    /// Background work whose results come back to `task_result`.
    fn tasks(&self) -> Tasks;
}

impl dyn SettingsIo + '_ {
    fn info(&mut self, text: impl Into<String>, timeout: Duration) {
        self.notify(text.into(), Severity::Information, timeout);
    }

    fn warn(&mut self, text: impl Into<String>, timeout: Duration) {
        self.notify(text.into(), Severity::Warning, timeout);
    }

    fn error(&mut self, text: impl Into<String>, timeout: Duration) {
        self.notify(text.into(), Severity::Error, timeout);
    }
}

/// What the front end does after a task result, besides showing the state.
#[derive(Debug, PartialEq, Eq)]
pub enum Step {
    None,
    /// config.yaml was read again: put its values back into the fields, then
    /// call `mark_saved_state`.
    Reloaded,
}

/// The raw editor closed. The front end posts it to `task_result`.
pub struct Edited(pub Result<(), String>);

/// Results from threads and tasks.
enum Task {
    Loaded(Result<Vec<AssetState>, String>),
    Progress(String, AssetProgress),
    Installed(String),
    Cancelled(String),
    Failed(String, String),
    Removed(String, u64),
    OllamaText(String),
    PullFailed(String, String),
    PullDone(String),
    AiTested(Check),
    Probe(ProbeEvent),
}

/// One speech model row.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelRow {
    pub status: AssetStatus,
    pub progress: Option<AssetProgress>,
    /// Bytes kept after a cancel; the status then reads "Paused".
    pub kept: u64,
    pub paused: bool,
}

impl ModelRow {
    fn missing() -> Self {
        Self {
            status: AssetStatus::Missing,
            progress: None,
            kept: 0,
            paused: false,
        }
    }
}

/// The speaker assets row: its status and the buttons it shows.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SpeakerStatus {
    /// Empty until the first check.
    pub status: String,
    pub install: bool,
    pub remove: bool,
    pub cancel: bool,
}

/// What the probe found out about a source's permission.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Permission {
    pub text: &'static str,
    /// The theme color the text takes, if any.
    pub color: Option<&'static str>,
}

/// The probe's view of one source.
#[derive(Debug, Clone, PartialEq)]
pub struct Meter {
    /// The newest level last.
    pub levels: VecDeque<f64>,
    /// False until the first level arrives; the meter then shows its start.
    pub live: bool,
    pub permission: Option<Permission>,
}

impl Meter {
    fn new() -> Self {
        Self {
            levels: VecDeque::from(vec![0.0; METER_BARS]),
            live: false,
            permission: None,
        }
    }
}

pub struct SettingsRuntime {
    paths: Paths,
    config_path: Option<PathBuf>,
    store: Arc<CachedModelStore>,
    pub loaded: Loaded,
    baseline: Option<Value>,
    /// Save worked at least once.
    pub saved: bool,
    /// The stored API key is removed on save.
    pub clear_api_key: bool,
    states: HashMap<String, AssetState>,
    downloads: HashMap<String, Arc<AtomicBool>>,
    speaker_progress: IndexMap<String, AssetProgress>,
    /// The speech model rows, by asset id.
    pub rows: HashMap<&'static str, ModelRow>,
    pub speaker: SpeakerStatus,
    /// The installed AI models line, once something set it.
    pub models_text: Option<String>,
    probe_stop: Option<Arc<AtomicBool>>,
    /// Microphone, then system audio.
    pub meters: [Meter; 2],
    /// The asset a Remove question is about; the speaker pair is `SPEAKER`.
    removing: Option<String>,
}

impl SettingsRuntime {
    pub fn new(paths: Paths, config_path: Option<PathBuf>, store: Arc<CachedModelStore>) -> Self {
        let loaded = load(&paths, config_path.as_deref());
        let rows = store
            .catalog()
            .iter()
            .filter(|asset| asset.group == AssetGroup::Transcription)
            .map(|asset| (asset.id, ModelRow::missing()))
            .collect();
        Self {
            paths,
            config_path,
            store,
            loaded,
            baseline: None,
            saved: false,
            clear_api_key: false,
            states: HashMap::new(),
            downloads: HashMap::new(),
            speaker_progress: IndexMap::new(),
            rows,
            speaker: SpeakerStatus::default(),
            models_text: None,
            probe_stop: None,
            meters: [Meter::new(), Meter::new()],
            removing: None,
        }
    }

    pub fn store(&self) -> &CachedModelStore {
        &self.store
    }

    /// The file Save writes and the raw editor opens.
    pub fn config_file(&self) -> PathBuf {
        self.config_path
            .clone()
            .unwrap_or_else(|| config_file_path(&self.paths))
    }

    /// The theme the loaded config names.
    pub fn theme(&self) -> &str {
        &self.loaded.infra.theme
    }

    /// The config data the form holds now.
    pub fn collect(&self, form: &mut dyn FormValues) -> Value {
        collect(form, &self.loaded.raw, self.clear_api_key)
    }

    /// Take the form as it is now as the saved state.
    pub fn mark_saved_state(&mut self, form: &mut dyn FormValues) {
        self.baseline = Some(self.collect(form));
    }

    pub fn has_unsaved_changes(&self, form: &mut dyn FormValues) -> bool {
        let Some(baseline) = &self.baseline else {
            return false;
        };
        self.collect(form) != *baseline
    }

    /// The form opened: take its state, report a broken file, read the model cache.
    /// `fix_hint` tells the user how this front end opens the file.
    pub fn mounted(&mut self, form: &mut dyn FormValues, io: &mut dyn SettingsIo, fix_hint: &str) {
        self.mark_saved_state(form);
        if !self.loaded.error.is_empty() {
            io.error(
                format!(
                    "config.yaml cannot be read ({}). Showing defaults. {fix_hint}",
                    self.loaded.error
                ),
                Duration::from_secs(20),
            );
        }
        let store = self.store.clone();
        io.tasks().spawn_thread_or(
            "models-load",
            move || Task::Loaded(store.list_assets().map_err(|e| e.to_string())),
            |message| Task::Loaded(Err(message)),
        );
    }

    // --- Save -----------------------------------------------------------

    /// Check the form and write it. None when nothing was written; else whether
    /// the file holds an API key, which the key row shows by.
    pub fn save(&mut self, form: &mut dyn FormValues, io: &mut dyn SettingsIo) -> Option<bool> {
        let errors = numeric_errors(form);
        if !errors.is_empty() {
            io.error(errors.join("\n"), Duration::from_secs(10));
            return None;
        }
        let data = self.collect(form);
        let model = data["transcription"]["model"]
            .as_str()
            .unwrap_or("")
            .to_string();
        if let Some(message) = model_error(&model) {
            io.error(message, Duration::from_secs(8));
            return None;
        }
        let diarization = data["diarization"]["enabled"].as_bool().unwrap_or(false);
        let save_audio = data["output"]["save_audio"].as_bool().unwrap_or(true);
        if diarization && !save_audio {
            io.error(
                "Speaker labels need Save audio turned on.",
                Duration::from_secs(8),
            );
            return None;
        }
        if let Err(e) = build_app_config(&data).and_then(|_| build_infra_config(&data).map(|_| ()))
        {
            io.error(format!("Validation error: {e}"), Duration::from_secs(8));
            return None;
        }
        let path = match write_config(&self.paths, &data, self.config_path.as_deref(), &FORM_OWNED)
        {
            Ok(path) => path,
            Err(e) => {
                error!(target: "ltn.config", "Cannot save config: {e}");
                io.error(format!("Cannot save config: {e}"), Duration::from_secs(10));
                return None;
            }
        };
        let has_key = data["openai"]["api_key"]
            .as_str()
            .is_some_and(|key| !key.is_empty());
        self.saved = true;
        self.clear_api_key = false;
        self.mark_saved_state(form);
        io.info(format!("Saved to {}", path.display()), DEFAULT_TIMEOUT);
        Some(has_key)
    }

    /// Remove the stored key on the next save. The front end empties the key field.
    pub fn clear_key(&mut self, io: &mut dyn SettingsIo) {
        self.clear_api_key = true;
        io.info("The stored key is removed on save.", Duration::from_secs(4));
    }

    /// Stop the downloads and the probe, whose results would go to a screen
    /// that is gone. The `.part` files stay. Returns `saved`.
    pub fn leave(&mut self) -> bool {
        for cancel in self.downloads.values() {
            cancel.store(true, Ordering::SeqCst);
        }
        if let Some(stop) = &self.probe_stop {
            stop.store(true, Ordering::SeqCst);
        }
        self.saved
    }

    // --- AI -------------------------------------------------------------

    fn form_infra(
        &self,
        form: &mut dyn FormValues,
        io: &mut dyn SettingsIo,
    ) -> Option<(InfraConfig, Value)> {
        let data = self.collect(form);
        match build_infra_config(&data) {
            Ok(infra) => Some((infra, data)),
            Err(e) => {
                io.error(format!("Invalid provider config: {e}"), DEFAULT_TIMEOUT);
                None
            }
        }
    }

    /// Try the AI the form names: connection, models, one short reply.
    pub fn test_ai(&mut self, form: &mut dyn FormValues, io: &mut dyn SettingsIo) {
        let Some((infra, data)) = self.form_infra(form, io) else {
            return;
        };
        let client = match crate::app::container::build_llm_client(&infra) {
            Ok(client) => client,
            Err(e) => {
                io.error(e, Duration::from_secs(8));
                return;
            }
        };
        let models = vec![
            data["digest"]["model"].as_str().unwrap_or("").to_string(),
            data["interactive"]["model"]
                .as_str()
                .unwrap_or("")
                .to_string(),
        ];
        io.info(
            "Testing the AI: connection, models, one short reply...",
            Duration::from_secs(4),
        );
        let provider = infra.llm_provider.clone();
        io.tasks().spawn_or(
            async move {
                let mut check = Check::new("ai", "AI", true);
                crate::app::preflight::check_ai(&mut check, &*client, &provider, &models, &|_| {})
                    .await;
                Task::AiTested(check)
            },
            |message| {
                let mut check = Check::new("ai", "AI", true);
                check.set(Status::Fail, format!("panic: {message}"), "");
                Task::AiTested(check)
            },
        );
    }

    fn ai_tested(&mut self, check: Check, io: &mut dyn SettingsIo) {
        if check.status == Status::Pass {
            io.info(
                format!("AI works: {}", check.detail),
                Duration::from_secs(6),
            );
            return;
        }
        let text = match check.fix.is_empty() {
            true => check.detail.clone(),
            false => format!("{}. {}", check.detail.trim_end_matches('.'), check.fix),
        };
        io.error(text, Duration::from_secs(12));
    }

    /// List the models the AI provider has installed.
    pub fn refresh_models(&mut self, form: &mut dyn FormValues, io: &mut dyn SettingsIo) {
        let Some((infra, _)) = self.form_infra(form, io) else {
            return;
        };
        self.models_text = Some("Loading...".into());
        io.tasks().spawn_or(
            async move {
                let text = match infra.llm_provider.as_str() {
                    "ollama" => {
                        let client =
                            crate::adapters::llm::ollama::OllamaLlmClient::new(&infra.ollama.host);
                        match client.list_models().await {
                            Ok(models) if models.is_empty() => "No models installed.".to_string(),
                            Ok(models) => models
                                .into_iter()
                                .map(|(name, _)| name)
                                .collect::<Vec<_>>()
                                .join(", "),
                            Err(e) => format!("Cannot list models: {e}"),
                        }
                    }
                    _ => match crate::app::container::build_llm_client(&infra) {
                        Ok(_) => "This provider does not list models.".to_string(),
                        Err(e) => format!("Cannot list models: {e}"),
                    },
                };
                Task::OllamaText(text)
            },
            |message| Task::OllamaText(format!("Cannot list models: {message}")),
        );
    }

    /// Pull the model named in the field `input_id` into Ollama.
    pub fn pull_model(
        &mut self,
        input_id: &str,
        form: &mut dyn FormValues,
        io: &mut dyn SettingsIo,
    ) {
        let model = form.input(input_id).trim().to_string();
        if model.is_empty() {
            io.warn("Enter a model name first.", Duration::from_secs(4));
            return;
        }
        let Some((infra, _)) = self.form_infra(form, io) else {
            return;
        };
        self.models_text = Some(format!("Pulling {model}..."));
        let tasks = io.tasks();
        let sender = tasks.clone();
        let failed_model = model.clone();
        tasks.spawn_or(
            async move {
                if infra.llm_provider != "ollama" {
                    let error = match crate::app::container::build_llm_client(&infra) {
                        Ok(_) => "This provider cannot pull models.".to_string(),
                        Err(e) => e,
                    };
                    return Task::PullFailed(model, error);
                }
                let client = crate::adapters::llm::ollama::OllamaLlmClient::new(&infra.ollama.host);
                let name = model.clone();
                let mut report = |event: crate::adapters::llm::ollama::PullProgress| {
                    let total = event.total.unwrap_or(0);
                    let completed = event.completed.unwrap_or(0);
                    // Ollama names a layer by its hash; the bytes say how far the pull is.
                    // A layer of a few KB would read "0 MB of 0 MB", so it keeps the status.
                    let text = match total >= MEGABYTE {
                        true => describe(&AssetProgress {
                            downloaded: completed,
                            total,
                            bytes_per_second: 0.0,
                        }),
                        false => event.status.clone(),
                    };
                    sender.post(Task::OllamaText(format!("Pulling {name}: {text}")));
                };
                match client.pull_model(&model, &mut report).await {
                    Ok(()) => Task::PullDone(model),
                    Err(e) => Task::PullFailed(model, e.to_string()),
                }
            },
            move |message| Task::PullFailed(failed_model, message),
        );
    }

    // --- Raw editor -----------------------------------------------------

    /// The file to open in an editor, created from the loaded config when it
    /// does not exist. None after an error notice.
    pub fn prepare_raw_edit(&self, io: &mut dyn SettingsIo) -> Option<PathBuf> {
        let path = self.config_file();
        if path.exists() {
            return Some(path);
        }
        let written = path
            .parent()
            .map_or(Ok(()), std::fs::create_dir_all)
            .and_then(|()| {
                std::fs::write(&path, crate::adapters::yaml_io::to_yaml(&self.loaded.raw))
            });
        match written {
            Ok(()) => Some(path),
            Err(e) => {
                error!(target: "ltn.config", "Cannot create {}: {e}", path.display());
                io.error(
                    format!("Cannot create {}: {e}", path.display()),
                    DEFAULT_TIMEOUT,
                );
                None
            }
        }
    }

    /// After the editor: read config.yaml again. True when it was read.
    fn after_edit(&mut self, result: Result<(), String>, io: &mut dyn SettingsIo) -> bool {
        if let Err(e) = result {
            io.error(format!("Cannot open editor: {e}"), DEFAULT_TIMEOUT);
            return false;
        }
        let path = self.config_file();
        let loaded = load(&self.paths, self.config_path.as_deref());
        if !loaded.error.is_empty() {
            // A hand edit can leave YAML that does not parse or is not a mapping.
            error!(target: "ltn.config", "Cannot reload {}: {}", path.display(), loaded.error);
            let name = path
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_default();
            io.error(
                format!("Cannot reload {name}: {}", loaded.error),
                Duration::from_secs(12),
            );
            return false;
        }
        self.loaded = loaded;
        true
    }

    // --- Models ---------------------------------------------------------

    /// The catalog label of an asset, or its id.
    pub fn label(&self, asset_id: &str) -> String {
        self.store
            .catalog()
            .iter()
            .find(|asset| asset.id == asset_id)
            .map_or(asset_id.to_string(), |asset| asset.label.to_string())
    }

    fn asset_size(&self, asset_id: &str) -> u64 {
        self.store
            .catalog()
            .iter()
            .find(|asset| asset.id == asset_id)
            .map_or(0, |asset| asset.size)
    }

    fn apply_state(&mut self, state: AssetState) {
        if state.status != AssetStatus::Downloading {
            self.speaker_progress.shift_remove(state.asset.id);
        }
        if state.asset.group == AssetGroup::Transcription
            && let Some(row) = self.rows.get_mut(state.asset.id)
        {
            row.status = state.status;
            row.paused = false;
            if state.status != AssetStatus::Downloading {
                row.progress = None;
            }
        }
        self.states.insert(state.asset.id.to_string(), state);
    }

    fn refresh(&mut self, asset_id: &str) {
        match self.store.status(asset_id) {
            Ok(state) => self.apply_state(state),
            Err(e) => error!(target: "ltn.config", "Cannot read the state of {asset_id}: {e}"),
        }
        self.update_speaker_status();
    }

    fn update_speaker_status(&mut self) {
        let mut speaker = SpeakerStatus::default();
        if !self.speaker_progress.is_empty() {
            let done: u64 = self.speaker_progress.values().map(|p| p.downloaded).sum();
            let total: u64 = self.speaker_progress.values().map(|p| p.total).sum();
            speaker.status = format!("Downloading {} / {}", megabytes(done), megabytes(total));
            speaker.cancel = true;
        } else {
            let runtime = self.states.get(RUNTIME_ID).map(|s| s.status);
            let model = self.states.get(NEMOTRON_ID).map(|s| s.status);
            let ready = match (runtime, model) {
                (None, _) | (_, None) => {
                    speaker.status = "Checking...".into();
                    false
                }
                (Some(AssetStatus::Busy), _) | (_, Some(AssetStatus::Busy)) => {
                    speaker.status = "Downloading elsewhere".into();
                    false
                }
                (Some(AssetStatus::Ready), Some(AssetStatus::Ready)) => {
                    let size = self.disk_size(RUNTIME_ID) + self.disk_size(NEMOTRON_ID);
                    speaker.status = format!("Ready ({})", megabytes(size));
                    true
                }
                _ => {
                    speaker.status = "Missing".into();
                    false
                }
            };
            speaker.install = !ready;
            speaker.remove = ready;
        }
        self.speaker = speaker;
    }

    fn disk_size(&self, asset_id: &str) -> u64 {
        // The catalog size is the download; a runtime archive takes more space once extracted.
        match self.states.get(asset_id) {
            Some(state) if state.installed_bytes > 0 => state.installed_bytes,
            _ => self.asset_size(asset_id),
        }
    }

    pub fn download_all_missing(&mut self, io: &mut dyn SettingsIo) {
        let ids: Vec<&'static str> = self
            .store
            .catalog()
            .iter()
            .filter(|asset| {
                self.states.get(asset.id).is_none_or(|state| {
                    matches!(state.status, AssetStatus::Missing | AssetStatus::Corrupt)
                })
            })
            .map(|asset| asset.id)
            .collect();
        for asset_id in ids {
            self.start_download(asset_id, io);
        }
    }

    pub fn install_speaker(&mut self, io: &mut dyn SettingsIo) {
        for asset_id in SPEAKER_IDS {
            self.start_download(asset_id, io);
        }
    }

    pub fn cancel_speaker(&mut self) {
        for asset_id in SPEAKER_IDS {
            self.cancel_download(asset_id);
        }
    }

    pub fn start_download(&mut self, asset_id: &str, io: &mut dyn SettingsIo) {
        if self.downloads.contains_key(asset_id) {
            return;
        }
        // Registered before the thread starts, so a second press never starts a second download.
        let cancel = Arc::new(AtomicBool::new(false));
        self.downloads.insert(asset_id.to_string(), cancel.clone());
        if SPEAKER_IDS.contains(&asset_id) {
            let total = self.asset_size(asset_id);
            self.speaker_progress.insert(
                asset_id.to_string(),
                AssetProgress {
                    downloaded: 0,
                    total,
                    bytes_per_second: 0.0,
                },
            );
            self.update_speaker_status();
        } else if let Some(row) = self.rows.get_mut(asset_id) {
            row.kept = 0;
            row.status = AssetStatus::Downloading;
            row.progress = None;
            row.paused = false;
        }
        let store = self.store.clone();
        let tasks = io.tasks();
        let sender = tasks.clone();
        let id = asset_id.to_string();
        let failed_id = id.clone();
        tasks.spawn_thread_or(
            &format!("model-{id}"),
            move || {
                let mut report =
                    |progress: AssetProgress| sender.post(Task::Progress(id.clone(), progress));
                match store.install(&id, &mut report, Some(&cancel)) {
                    Ok(_) => Task::Installed(id),
                    Err(StoreError::Download(DownloadError::Cancelled(_))) => {
                        info!(target: "ltn.config", "download of {id} cancelled");
                        Task::Cancelled(id)
                    }
                    Err(e) => {
                        error!(target: "ltn.config", "download of {id} failed: {e}");
                        Task::Failed(id, e.to_string())
                    }
                }
            },
            move |message| Task::Failed(failed_id, message),
        );
    }

    pub fn cancel_download(&mut self, asset_id: &str) {
        if let Some(cancel) = self.downloads.get(asset_id) {
            cancel.store(true, Ordering::SeqCst);
        }
    }

    /// True while a download of `asset_id` runs.
    pub fn downloading(&self, asset_id: &str) -> bool {
        self.downloads.contains_key(asset_id)
    }

    /// What Remove of `asset_id` (or `SPEAKER`) would free, to ask the user.
    /// Returns the label and the estimate; `answer_remove` takes the answer.
    pub fn ask_remove(
        &mut self,
        asset_id: &str,
        io: &mut dyn SettingsIo,
    ) -> Option<(String, RemovalEstimate)> {
        let asked = match asset_id {
            SPEAKER => {
                let mut estimates = Vec::new();
                for id in SPEAKER_IDS {
                    match self.store.removal(id) {
                        Ok(estimate) => estimates.push(estimate),
                        Err(e) => {
                            io.error(format!("{}: {e}", self.label(id)), Duration::from_secs(10));
                            return None;
                        }
                    }
                }
                (
                    "Speaker runtime and model".to_string(),
                    combined(&estimates),
                )
            }
            id => match self.store.removal(id) {
                Ok(estimate) => (self.label(id), estimate),
                Err(e) => {
                    io.error(format!("{}: {e}", self.label(id)), Duration::from_secs(10));
                    return None;
                }
            },
        };
        self.removing = Some(asset_id.to_string());
        Some(asked)
    }

    /// The answer to `ask_remove`. A confirmed one removes on a thread.
    pub fn answer_remove(&mut self, confirmed: bool, delete_legacy: bool, io: &mut dyn SettingsIo) {
        let Some(asset_id) = self.removing.take() else {
            return;
        };
        if !confirmed {
            return;
        }
        let store = self.store.clone();
        let failed_id = match asset_id.as_str() {
            SPEAKER => SPEAKER_IDS[0].to_string(),
            id => id.to_string(),
        };
        io.tasks().spawn_thread_or(
            &format!("remove-{asset_id}"),
            move || {
                // The speaker pair is one thing to the user: one worker and one notice.
                let ids: Vec<&str> = match asset_id.as_str() {
                    SPEAKER => SPEAKER_IDS.to_vec(),
                    id => vec![id],
                };
                let mut freed = 0;
                for id in &ids {
                    match store.remove(id, delete_legacy) {
                        Ok(bytes) => freed += bytes,
                        Err(e) => return Task::Failed(id.to_string(), e.to_string()),
                    }
                }
                Task::Removed(ids[0].to_string(), freed)
            },
            move |message| Task::Failed(failed_id, message),
        );
    }

    // --- Audio probe ----------------------------------------------------

    pub fn probe_running(&self) -> bool {
        self.probe_stop.is_some()
    }

    /// The silence threshold the meters are drawn against.
    pub fn silence_threshold(&self) -> f64 {
        self.loaded.raw["transcription"]["silence_threshold"]
            .as_f64()
            .unwrap_or(0.01)
    }

    /// Listen to the microphone the form names and to system audio.
    pub fn start_probe(&mut self, form: &mut dyn FormValues, io: &mut dyn SettingsIo) {
        if self.probe_stop.is_some() {
            return;
        }
        let device = form.select("cfg-audio-device");
        let device = (device != DEFAULT_DEVICE).then_some(device);
        let (mic, system) = crate::app::container::build_audio_sources(device, true);
        let stop = Arc::new(AtomicBool::new(false));
        self.probe_stop = Some(stop.clone());
        let tasks = io.tasks();
        let sender = tasks.clone();
        tasks.spawn_thread("audio-probe", move || {
            let post = |event| sender.post(Task::Probe(event));
            probe::run_audio_probe(mic, system, &post, &stop, probe::probe_seconds());
        });
    }

    pub fn stop_probe(&mut self) {
        if let Some(stop) = &self.probe_stop {
            stop.store(true, Ordering::SeqCst);
        }
    }

    fn on_probe(&mut self, event: ProbeEvent, io: &mut dyn SettingsIo) {
        let meter = |source: Source| match source {
            Source::Mic => 0,
            Source::System => 1,
        };
        match event {
            ProbeEvent::Level(source, rms) => {
                let meter = &mut self.meters[meter(source)];
                meter.levels.pop_front();
                meter.levels.push_back(rms);
                meter.live = true;
            }
            ProbeEvent::Permission(source, state) => {
                let (text, color) = match state {
                    "allowed" => ("Allowed", Some("success")),
                    "waiting" => ("Waiting for permission", Some("warning")),
                    "no_signal" => ("No signal", Some("error")),
                    other => (other, None),
                };
                self.meters[meter(source)].permission = Some(Permission { text, color });
            }
            ProbeEvent::Error(e) => {
                io.error(format!("Audio test failed: {e}"), Duration::from_secs(10));
            }
            ProbeEvent::Done => self.probe_stop = None,
        }
    }

    // --- Tasks ----------------------------------------------------------

    /// A task result. Gives the payload back when it is not a Settings result.
    pub fn task_result(
        &mut self,
        payload: Payload,
        form: &mut dyn FormValues,
        io: &mut dyn SettingsIo,
    ) -> Result<Step, Payload> {
        let payload = match payload.downcast::<Edited>() {
            Ok(edited) => {
                return Ok(match self.after_edit(edited.0, io) {
                    true => Step::Reloaded,
                    false => Step::None,
                });
            }
            Err(payload) => payload,
        };
        let task = payload.downcast::<Task>()?;
        self.on_task(*task, form, io);
        Ok(Step::None)
    }

    fn on_task(&mut self, task: Task, form: &mut dyn FormValues, io: &mut dyn SettingsIo) {
        match task {
            Task::Loaded(Ok(states)) => {
                for state in states {
                    if self.downloads.contains_key(state.asset.id) {
                        continue; // a download started before the load finished; its row is newer
                    }
                    self.apply_state(state);
                }
                self.update_speaker_status();
            }
            Task::Loaded(Err(e)) => {
                error!(target: "ltn.config", "Cannot read the model cache: {e}");
                io.error(
                    format!("Cannot read the model cache: {e}"),
                    Duration::from_secs(10),
                );
            }
            Task::Progress(id, progress) => self.on_progress(id, progress),
            Task::Installed(id) => {
                self.downloads.remove(&id);
                self.refresh(&id);
                io.info(
                    format!("{} is ready.", self.label(&id)),
                    Duration::from_secs(4),
                );
            }
            Task::Cancelled(id) => {
                self.downloads.remove(&id);
                self.refresh(&id);
                if !SPEAKER_IDS.contains(&id.as_str())
                    && let Some(row) = self.rows.get_mut(id.as_str())
                {
                    // The `.part` stays, and Download continues from it.
                    row.paused = row.kept > 0;
                }
                io.info(
                    format!("Download cancelled: {}.", self.label(&id)),
                    Duration::from_secs(4),
                );
            }
            Task::Failed(id, e) => {
                self.downloads.remove(&id);
                self.refresh(&id);
                io.error(format!("{}: {e}", self.label(&id)), Duration::from_secs(10));
            }
            Task::Removed(id, freed) => {
                let ids: Vec<&str> = match SPEAKER_IDS.contains(&id.as_str()) {
                    true => SPEAKER_IDS.to_vec(),
                    false => vec![id.as_str()],
                };
                for asset_id in ids {
                    self.refresh(asset_id);
                }
                io.info(
                    format!("Freed {}.", megabytes(freed)),
                    Duration::from_secs(4),
                );
            }
            Task::OllamaText(text) => self.models_text = Some(text),
            Task::PullFailed(model, e) => {
                self.models_text = Some(format!("Pull failed: {e}"));
                io.error(
                    format!("Pull failed for {model}: {e}"),
                    Duration::from_secs(10),
                );
            }
            Task::PullDone(model) => {
                io.info(format!("{model} is ready."), Duration::from_secs(5));
                self.refresh_models(form, io);
            }
            Task::AiTested(check) => self.ai_tested(check, io),
            Task::Probe(event) => self.on_probe(event, io),
        }
    }

    fn on_progress(&mut self, id: String, progress: AssetProgress) {
        if SPEAKER_IDS.contains(&id.as_str()) {
            self.speaker_progress.insert(id, progress);
            self.update_speaker_status();
            return;
        }
        let size = self.asset_size(&id).max(1);
        if let Some(row) = self.rows.get_mut(id.as_str()) {
            let total = if progress.total > 0 {
                progress.total
            } else {
                size
            };
            row.progress = Some(AssetProgress { total, ..progress });
            row.kept = progress.downloaded;
        }
    }
}

/// One estimate for assets removed together. Their older copies are named by their common folder.
fn combined(estimates: &[RemovalEstimate]) -> RemovalEstimate {
    let legacy: Vec<&Path> = estimates
        .iter()
        .filter_map(|estimate| estimate.legacy_path.as_deref().map(Path::new))
        .collect();
    let common = legacy.split_first().map(|(first, rest)| {
        let mut common = first.to_path_buf();
        for path in rest {
            while !path.starts_with(&common) {
                if !common.pop() {
                    break;
                }
            }
        }
        common.to_string_lossy().to_string()
    });
    RemovalEstimate {
        frees: estimates.iter().map(|e| e.frees).sum(),
        frees_with_legacy: estimates.iter().map(|e| e.frees_with_legacy).sum(),
        legacy_path: common,
    }
}

/// Open the macOS privacy pane `pane` (`Privacy_Microphone`, `Privacy_ScreenCapture`).
pub fn open_privacy(pane: &str) {
    if !cfg!(target_os = "macos") {
        return;
    }
    let url = format!("x-apple.systempreferences:com.apple.preference.security?{pane}");
    if let Err(e) = std::process::Command::new("/usr/bin/open").arg(url).spawn() {
        error!(target: "ltn.config", "Cannot open the privacy settings: {e}");
    }
}
