//! Settings (Python's `ConfigApp`): a page list at the left, one page at a time.

mod data;
mod dialogs;
mod form;
mod models;
mod pages;
mod preview;
mod probe;

pub use data::{FORM_OWNED, collect, load};
pub use dialogs::DISCARD_BINDINGS;
pub use pages::THEME_ORDER;

use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use crossterm::event::KeyEvent;
use indexmap::IndexMap;
use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::Line;
use ratatui::widgets::{Block, BorderType, Borders, Paragraph};
use tracing::{error, info};
use yaml_serde::Value;

use self::data::{Loaded, model_error, numeric_errors};
use self::dialogs::{Answer, Dialog};
use self::form::{FormEvent, Kind, Page};
use self::models::{ModelChooser, SpeakerState};
use self::probe::{ProbeEvent, Source};
use crate::adapters::assets::download::DownloadError;
use crate::adapters::assets::store::{CachedModelStore, StoreError};
use crate::adapters::config_file::{config_file_path, write_config};
use crate::adapters::model_catalog::{NEMOTRON_ID, RUNTIME_ID};
use crate::adapters::paths::Paths;
use crate::app::download_progress::{describe, megabytes};
use crate::app::preflight::{Check, Status};
use crate::config::{InfraConfig, build_app_config, build_infra_config};
use crate::domain::model_asset::{AssetGroup, AssetProgress, AssetState, AssetStatus};
use crate::tui::host::{
    Action, Ctx, MouseInput, MouseKind, Msg, Payload, Screen, Severity, TaskSender,
};
use crate::tui::keys::{Binding, SCREEN_BASE, bind};
use crate::tui::screens::template_picker::resolve_editor;
use crate::tui::theme::Theme;
use crate::tui::widgets::list::ListCursor;
use crate::tui::widgets::status_bar::rms_to_char;

pub const SETTINGS_BINDINGS: [Binding; 6] = [
    SCREEN_BASE[0],
    SCREEN_BASE[1],
    SCREEN_BASE[2],
    SCREEN_BASE[3],
    bind("escape", "quit_app", "Quit").priority(),
    bind("ctrl+s", "save_config", "Save").priority(),
];

const NAV: [&str; 5] = ["Models", "Digest", "Audio", "Sessions", "Appearance"];
const NAV_WIDTH: u16 = 20;
const MEGABYTE: u64 = 1024 * 1024;
const SPEAKER_IDS: [&str; 2] = [RUNTIME_ID, NEMOTRON_ID];
/// The Appearance page is a plain column; the others scroll and take the focus.
const APPEARANCE: usize = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Focus {
    Nav,
    /// The page's scroll container.
    Container,
    Page,
}

/// Results from threads and tasks, back on the event loop.
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
    Edited(Result<(), String>),
}

pub struct SettingsScreen {
    paths: Paths,
    config_path: Option<PathBuf>,
    store: Arc<CachedModelStore>,
    loaded: Loaded,
    pages: [Page; 5],
    nav: ListCursor,
    focus: Focus,
    baseline: Option<Value>,
    saved: bool,
    clear_api_key: bool,
    states: HashMap<String, AssetState>,
    downloads: HashMap<String, Arc<AtomicBool>>,
    speaker_progress: IndexMap<String, AssetProgress>,
    probe_stop: Option<Arc<AtomicBool>>,
    levels: [VecDeque<f64>; 2],
    /// The asset a Remove dialog asks about; the speaker pair is `speaker`.
    removing: Option<String>,
    discarding: bool,
}

fn build_pages(loaded: &Loaded, store: &CachedModelStore, devices: &[String]) -> [Page; 5] {
    let source = pages::Source {
        raw: &loaded.raw,
        infra: &loaded.infra,
        catalog: store.catalog(),
        cache_root: store.cache_root(),
        input_devices: devices,
    };
    [
        pages::models_page(&source),
        pages::digest_page(&loaded.raw),
        pages::audio_page(&loaded.raw, source.input_devices),
        pages::sessions_page(&loaded.raw, &loaded.infra),
        pages::appearance_page(&loaded.infra),
    ]
}

impl SettingsScreen {
    pub fn new(paths: Paths, config_path: Option<PathBuf>) -> Self {
        let store = Arc::new(CachedModelStore::new(&paths));
        let devices = crate::adapters::audio::microphone::list_input_devices();
        Self::with(paths, config_path, store, &devices)
    }

    /// Build with a given store and device list, for tests.
    fn with(
        paths: Paths,
        config_path: Option<PathBuf>,
        store: Arc<CachedModelStore>,
        devices: &[String],
    ) -> Self {
        let loaded = load(&paths, config_path.as_deref());
        let pages = build_pages(&loaded, &store, devices);
        let mut nav = ListCursor::default();
        nav.reset(NAV.len());
        nav.select(0);
        Self {
            paths,
            config_path,
            store,
            loaded,
            pages,
            nav,
            focus: Focus::Nav,
            baseline: None,
            saved: false,
            clear_api_key: false,
            states: HashMap::new(),
            downloads: HashMap::new(),
            speaker_progress: IndexMap::new(),
            probe_stop: None,
            levels: [VecDeque::from(vec![0.0; 12]), VecDeque::from(vec![0.0; 12])],
            removing: None,
            discarding: false,
        }
    }

    fn page_index(&self) -> usize {
        self.nav.index.unwrap_or(0)
    }

    fn page(&mut self) -> &mut Page {
        let index = self.page_index();
        &mut self.pages[index]
    }

    fn config_file(&self) -> PathBuf {
        self.config_path
            .clone()
            .unwrap_or_else(|| config_file_path(&self.paths))
    }

    fn collect(&mut self) -> Value {
        collect(&mut self.pages, &self.loaded.raw, self.clear_api_key)
    }

    fn has_unsaved_changes(&mut self) -> bool {
        let Some(baseline) = self.baseline.clone() else {
            return false;
        };
        self.collect() != baseline
    }

    // --- Save -----------------------------------------------------------

    fn save(&mut self, ctx: &mut Ctx) {
        let errors = numeric_errors(&self.pages);
        if !errors.is_empty() {
            ctx.error(errors.join("\n"), Duration::from_secs(10));
            return;
        }
        let data = self.collect();
        let model = data["transcription"]["model"]
            .as_str()
            .unwrap_or("")
            .to_string();
        if let Some(message) = model_error(&model) {
            ctx.error(message, Duration::from_secs(8));
            return;
        }
        let diarization = data["diarization"]["enabled"].as_bool().unwrap_or(false);
        let save_audio = data["output"]["save_audio"].as_bool().unwrap_or(true);
        if diarization && !save_audio {
            ctx.error(
                "Speaker labels need Save audio turned on.",
                Duration::from_secs(8),
            );
            return;
        }
        if let Err(e) = build_app_config(&data).and_then(|_| build_infra_config(&data).map(|_| ()))
        {
            ctx.error(format!("Validation error: {e}"), Duration::from_secs(8));
            return;
        }
        let path = match write_config(&self.paths, &data, self.config_path.as_deref(), &FORM_OWNED)
        {
            Ok(path) => path,
            Err(e) => {
                error!(target: "ltn.config", "Cannot save config: {e}");
                ctx.error(format!("Cannot save config: {e}"), Duration::from_secs(10));
                return;
            }
        };
        self.saved = true;
        self.clear_api_key = false;
        let has_key = data["openai"]["api_key"]
            .as_str()
            .is_some_and(|key| !key.is_empty());
        self.pages[0].show("clear-key-row", has_key);
        self.baseline = Some(self.collect());
        ctx.notify(format!("Saved to {}", path.display()));
    }

    // --- Leave ----------------------------------------------------------

    fn quit(&mut self) -> Action {
        let index = self.page_index();
        if self.pages[index].expanded() {
            // Esc closes an open dropdown, as it does everywhere else.
            self.pages[index].close_selects();
            return Action::None;
        }
        if self.focus != Focus::Nav {
            // Inside a page, Esc goes back to the page list; from the list, it leaves.
            self.pages[index].set_focus(None);
            self.focus = Focus::Nav;
            return Action::None;
        }
        if self.has_unsaved_changes() {
            self.discarding = true;
            return Action::Push(Box::new(Dialog::discard()));
        }
        self.leave()
    }

    fn leave(&mut self) -> Action {
        // A download thread would post into a screen that is gone. Stop it; the .part stays.
        for cancel in self.downloads.values() {
            cancel.store(true, Ordering::SeqCst);
        }
        if let Some(stop) = &self.probe_stop {
            stop.store(true, Ordering::SeqCst);
        }
        Action::Pop(Some(Box::new(SettingsClosed { saved: self.saved })))
    }

    // --- Focus ----------------------------------------------------------

    fn focus_next(&mut self, forward: bool) {
        let index = self.page_index();
        let has_container = index != APPEARANCE;
        match (self.focus, forward) {
            (Focus::Nav, true) if has_container => self.focus = Focus::Container,
            (Focus::Nav, true) | (Focus::Container, true) => self.enter_page(true),
            (Focus::Nav, false) => self.enter_page(false),
            (Focus::Container, false) => self.focus = Focus::Nav,
            (Focus::Page, _) => {
                if self.pages[index].step_focus(forward).is_none() {
                    self.focus = match (forward, has_container) {
                        (false, true) => Focus::Container,
                        _ => Focus::Nav,
                    };
                }
            }
        }
    }

    fn enter_page(&mut self, first: bool) {
        let page = self.page();
        let stop = if first {
            page.first_stop()
        } else {
            page.last_stop()
        };
        page.set_focus(stop);
        self.focus = match stop {
            Some(_) => Focus::Page,
            None => Focus::Nav,
        };
    }

    // --- Page events ----------------------------------------------------

    fn on_event(&mut self, event: FormEvent, ctx: &mut Ctx) -> Action {
        match event {
            FormEvent::Changed("cfg-theme", value) => ctx.set_theme(&value),
            FormEvent::Changed(..) => {}
            FormEvent::Custom("model-chosen", _) => {
                // A custom path wins over the chosen row, so choosing a row clears it.
                self.pages[0].set_input("cfg-trans-custom-path", "");
            }
            FormEvent::Custom(..) => {}
            FormEvent::Pressed(id) => return self.pressed(&id, ctx),
        }
        Action::None
    }

    fn pressed(&mut self, id: &str, ctx: &mut Ctx) -> Action {
        match id {
            "cfg-openai-clear-key" => {
                self.pages[0].set_input("cfg-openai-api-key", "");
                self.clear_api_key = true;
                ctx.notify_with(
                    "The stored key is removed on save.",
                    Severity::Information,
                    Duration::from_secs(4),
                    "",
                );
            }
            "cfg-ai-test" => self.test_ai(ctx),
            "cfg-ollama-refresh" => self.refresh_ollama(ctx),
            "cfg-ollama-pull-digest" => self.pull_ollama("cfg-digest-model", ctx),
            "cfg-ollama-pull-interactive" => self.pull_ollama("cfg-interactive-model", ctx),
            "models-download-all" => self.download_all_missing(ctx),
            "models-open-cache" => {
                if let Err(e) = crate::tui::system::open_folder(self.store.cache_root()) {
                    error!(target: "ltn.config", "Cannot open the cache folder: {e}");
                }
            }
            "speaker-install" => {
                for asset_id in SPEAKER_IDS {
                    self.start_download(asset_id, ctx);
                }
            }
            "speaker-remove" => return self.confirm_remove("speaker", ctx),
            "speaker-cancel" => {
                for asset_id in SPEAKER_IDS {
                    self.cancel_download(asset_id);
                }
            }
            "audio-start" => self.start_probe(ctx),
            "audio-stop" => {
                if let Some(stop) = &self.probe_stop {
                    stop.store(true, Ordering::SeqCst);
                }
            }
            "audio-grant-system" => open_privacy("Privacy_ScreenCapture"),
            "audio-grant-mic" => open_privacy("Privacy_Microphone"),
            other => {
                if let Some(asset_id) = other.strip_prefix("download-") {
                    self.start_download(asset_id, ctx);
                } else if let Some(asset_id) = other.strip_prefix("remove-") {
                    return self.confirm_remove(asset_id, ctx);
                } else if let Some(asset_id) = other.strip_prefix("cancel-") {
                    self.cancel_download(asset_id);
                }
            }
        }
        Action::None
    }

    // --- AI -------------------------------------------------------------

    fn form_infra(&mut self, ctx: &mut Ctx) -> Option<(InfraConfig, Value)> {
        let data = self.collect();
        match build_infra_config(&data) {
            Ok(infra) => Some((infra, data)),
            Err(e) => {
                ctx.error(
                    format!("Invalid provider config: {e}"),
                    crate::tui::host::NOTIFICATION_TIMEOUT,
                );
                None
            }
        }
    }

    fn test_ai(&mut self, ctx: &mut Ctx) {
        let Some((infra, data)) = self.form_infra(ctx) else {
            return;
        };
        let client = match crate::app::container::build_llm_client(&infra) {
            Ok(client) => client,
            Err(e) => {
                ctx.error(e, Duration::from_secs(8));
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
        ctx.notify_with(
            "Testing the AI: connection, models, one short reply...",
            Severity::Information,
            Duration::from_secs(4),
            "",
        );
        let provider = infra.llm_provider.clone();
        ctx.spawn_or(
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

    fn ai_tested(&mut self, check: Check, ctx: &mut Ctx) {
        if check.status == Status::Pass {
            ctx.notify_with(
                format!("AI works: {}", check.detail),
                Severity::Information,
                Duration::from_secs(6),
                "",
            );
            return;
        }
        let text = match check.fix.is_empty() {
            true => check.detail.clone(),
            false => format!("{}. {}", check.detail.trim_end_matches('.'), check.fix),
        };
        ctx.error(text, Duration::from_secs(12));
    }

    fn set_models_text(&mut self, text: &str) {
        self.pages[0].set_text("cfg-ollama-models", text);
    }

    fn refresh_ollama(&mut self, ctx: &mut Ctx) {
        let Some((infra, _)) = self.form_infra(ctx) else {
            return;
        };
        self.set_models_text("Loading...");
        ctx.spawn_or(
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

    fn pull_ollama(&mut self, input_id: &str, ctx: &mut Ctx) {
        let model = self.pages[0].input(input_id).trim().to_string();
        if model.is_empty() {
            ctx.warn("Enter a model name first.", Duration::from_secs(4));
            return;
        }
        let Some((infra, _)) = self.form_infra(ctx) else {
            return;
        };
        self.set_models_text(&format!("Pulling {model}..."));
        let sender = ctx.task_sender();
        let failed_model = model.clone();
        ctx.spawn_or(
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

    fn open_raw_editor(&mut self, ctx: &mut Ctx) {
        let Some(editor) = resolve_editor() else {
            ctx.error(
                "No editor found ($VISUAL / $EDITOR)",
                crate::tui::host::NOTIFICATION_TIMEOUT,
            );
            return;
        };
        let path = self.config_file();
        if !path.exists() {
            let written = path
                .parent()
                .map_or(Ok(()), std::fs::create_dir_all)
                .and_then(|()| {
                    std::fs::write(&path, crate::adapters::yaml_io::to_yaml(&self.loaded.raw))
                });
            if let Err(e) = written {
                error!(target: "ltn.config", "Cannot create {}: {e}", path.display());
                ctx.error(
                    format!("Cannot create {}: {e}", path.display()),
                    crate::tui::host::NOTIFICATION_TIMEOUT,
                );
                return;
            }
        }
        ctx.suspend(move || {
            let status = std::process::Command::new(&editor[0])
                .args(&editor[1..])
                .arg(&path)
                .status();
            Task::Edited(status.map(|_| ()).map_err(|e| {
                // $EDITOR can name a program that is not installed.
                error!(target: "ltn.config", "Cannot open editor {editor:?}: {e}");
                e.to_string()
            }))
        });
    }

    fn after_edit(&mut self, result: Result<(), String>, ctx: &mut Ctx) {
        if let Err(e) = result {
            ctx.error(
                format!("Cannot open editor: {e}"),
                crate::tui::host::NOTIFICATION_TIMEOUT,
            );
            return;
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
            ctx.error(
                format!("Cannot reload {name}: {}", loaded.error),
                Duration::from_secs(12),
            );
            return;
        }
        self.loaded = loaded;
        self.repopulate(ctx);
        self.baseline = Some(self.collect());
    }

    /// Push the reloaded values back into the fields, as `_repopulate_fields`.
    /// The custom model path, the buttons and the status lines stay as they are.
    fn repopulate(&mut self, ctx: &mut Ctx) {
        let devices = crate::adapters::audio::microphone::list_input_devices();
        let fresh = build_pages(&self.loaded, &self.store, &devices);
        for (page, new) in self.pages.iter_mut().zip(fresh) {
            for item in new.items {
                let replace = match &item.kind {
                    Kind::Field { .. }
                    | Kind::Switch { .. }
                    | Kind::Select { .. }
                    | Kind::TextArea(_) => true,
                    Kind::Custom(_) => item.id == "models-language",
                    _ => false,
                };
                if !replace || item.id.is_empty() {
                    continue;
                }
                if let Some(old) = page.item_mut(item.id) {
                    old.kind = item.kind;
                }
            }
        }
        let model = pages::python_str(
            self.loaded
                .raw
                .get("transcription")
                .and_then(|t| t.get("model")),
        );
        if let Some(chooser) = self.chooser() {
            chooser.select_by_uri(&model);
        }
        // Setting the theme select applies the theme, as its Changed event does in Python.
        let theme = self.loaded.infra.theme.clone();
        if ctx.theme.name != theme {
            ctx.set_theme(&theme);
        }
    }

    // --- Models ---------------------------------------------------------

    fn label(&self, asset_id: &str) -> String {
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

    fn chooser(&mut self) -> Option<&mut ModelChooser> {
        self.pages[0].custom::<ModelChooser>("models-chooser")
    }

    fn apply_state(&mut self, state: AssetState) {
        if state.status != AssetStatus::Downloading {
            self.speaker_progress.shift_remove(state.asset.id);
        }
        if state.asset.group == AssetGroup::Transcription
            && let Some(chooser) = self.chooser()
        {
            chooser.set_state(&state);
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
        let mut speaker = SpeakerState::default();
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
        if let Some(row) = self.pages[0].custom::<SpeakerState>("speaker-state") {
            *row = speaker;
        }
    }

    fn disk_size(&self, asset_id: &str) -> u64 {
        // The catalog size is the download; a runtime archive takes more space once extracted.
        match self.states.get(asset_id) {
            Some(state) if state.installed_bytes > 0 => state.installed_bytes,
            _ => self.asset_size(asset_id),
        }
    }

    fn download_all_missing(&mut self, ctx: &mut Ctx) {
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
            self.start_download(asset_id, ctx);
        }
    }

    fn start_download(&mut self, asset_id: &str, ctx: &mut Ctx) {
        if self.downloads.contains_key(asset_id) {
            return;
        }
        // Registered on the event loop, so a second press never starts a second download.
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
        } else if let Some(chooser) = self.chooser() {
            chooser.mark_downloading(asset_id);
        }
        let store = self.store.clone();
        let sender: TaskSender = ctx.task_sender();
        let id = asset_id.to_string();
        let failed_id = id.clone();
        ctx.spawn_thread_or(
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

    fn cancel_download(&mut self, asset_id: &str) {
        if let Some(cancel) = self.downloads.get(asset_id) {
            cancel.store(true, Ordering::SeqCst);
        }
    }

    fn confirm_remove(&mut self, asset_id: &str, ctx: &mut Ctx) -> Action {
        let (label, estimate) = match asset_id {
            "speaker" => {
                let mut estimates = Vec::new();
                for id in SPEAKER_IDS {
                    match self.store.removal(id) {
                        Ok(estimate) => estimates.push(estimate),
                        Err(e) => {
                            ctx.error(format!("{}: {e}", self.label(id)), Duration::from_secs(10));
                            return Action::None;
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
                    ctx.error(format!("{}: {e}", self.label(id)), Duration::from_secs(10));
                    return Action::None;
                }
            },
        };
        self.removing = Some(asset_id.to_string());
        Action::Push(Box::new(Dialog::remove(&label, &estimate)))
    }

    fn remove(&mut self, asset_id: String, delete_legacy: bool, ctx: &mut Ctx) {
        let store = self.store.clone();
        let failed_id = match asset_id.as_str() {
            "speaker" => SPEAKER_IDS[0].to_string(),
            id => id.to_string(),
        };
        ctx.spawn_thread_or(
            &format!("remove-{asset_id}"),
            move || {
                // The speaker pair is one thing to the user: one worker and one toast.
                let ids: Vec<&str> = match asset_id.as_str() {
                    "speaker" => SPEAKER_IDS.to_vec(),
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

    fn set_probe_running(&mut self, running: bool) {
        let page = &mut self.pages[2];
        let focused_button = page
            .focus
            .and_then(|(index, _)| (page.items[index].id == "probe-buttons").then_some(()));
        if let Some(button) = page.button_mut("audio-start") {
            button.disabled = running;
        }
        if let Some(button) = page.button_mut("audio-stop") {
            button.disabled = !running;
        }
        // A disabled button drops focus. The other one takes it, so Enter works again.
        if focused_button.is_some() {
            let index = page
                .items
                .iter()
                .position(|item| item.id == "probe-buttons");
            page.focus = index.map(|index| (index, 0));
        }
    }

    fn start_probe(&mut self, ctx: &mut Ctx) {
        if self.probe_stop.is_some() {
            return;
        }
        let device = self.pages[2].select("cfg-audio-device");
        let device = (device != pages::DEFAULT_DEVICE).then_some(device);
        let (mic, system) = crate::app::container::build_audio_sources(device, true);
        let stop = Arc::new(AtomicBool::new(false));
        self.probe_stop = Some(stop.clone());
        self.set_probe_running(true);
        let sender = ctx.task_sender();
        ctx.spawn_thread("audio-probe", move || {
            let post = |event| sender.post(Task::Probe(event));
            probe::run_audio_probe(mic, system, &post, &stop, probe::probe_seconds());
        });
    }

    fn on_probe(&mut self, event: ProbeEvent, ctx: &mut Ctx) {
        let silence = self.loaded.raw["transcription"]["silence_threshold"]
            .as_f64()
            .unwrap_or(0.01);
        match event {
            ProbeEvent::Level(source, rms) => {
                let (levels, id) = match source {
                    Source::Mic => (&mut self.levels[0], "audio-meter-mic"),
                    Source::System => (&mut self.levels[1], "audio-meter-system"),
                };
                levels.pop_front();
                levels.push_back(rms);
                let text: String = levels
                    .iter()
                    .map(|level| rms_to_char(*level, silence))
                    .collect();
                self.pages[2].set_text(id, &text);
            }
            ProbeEvent::Permission(source, state) => {
                let id = match source {
                    Source::Mic => "audio-perm-mic",
                    Source::System => "audio-perm-system",
                };
                let (text, color) = match state {
                    "allowed" => ("Allowed", Some("success")),
                    "waiting" => ("Waiting for permission", Some("warning")),
                    "no_signal" => ("No signal", Some("error")),
                    other => (other, None),
                };
                if let Some(item) = self.pages[2].item_mut(id) {
                    item.kind = Kind::Text(text.into(), color);
                }
            }
            ProbeEvent::Error(e) => {
                ctx.error(format!("Audio test failed: {e}"), Duration::from_secs(10));
            }
            ProbeEvent::Done => {
                self.probe_stop = None;
                self.set_probe_running(false);
            }
        }
    }

    // --- Tasks ----------------------------------------------------------

    fn on_task(&mut self, task: Task, ctx: &mut Ctx) {
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
                ctx.error(
                    format!("Cannot read the model cache: {e}"),
                    Duration::from_secs(10),
                );
            }
            Task::Progress(id, progress) => {
                if SPEAKER_IDS.contains(&id.as_str()) {
                    self.speaker_progress.insert(id, progress);
                    self.update_speaker_status();
                } else if let Some(chooser) = self.chooser() {
                    chooser.set_progress(&id, progress);
                }
            }
            Task::Installed(id) => {
                self.downloads.remove(&id);
                self.refresh(&id);
                ctx.notify_with(
                    format!("{} is ready.", self.label(&id)),
                    Severity::Information,
                    Duration::from_secs(4),
                    "",
                );
            }
            Task::Cancelled(id) => {
                self.downloads.remove(&id);
                self.refresh(&id);
                if !SPEAKER_IDS.contains(&id.as_str())
                    && let Some(chooser) = self.chooser()
                {
                    chooser.mark_paused(&id);
                }
                ctx.notify_with(
                    format!("Download cancelled: {}.", self.label(&id)),
                    Severity::Information,
                    Duration::from_secs(4),
                    "",
                );
            }
            Task::Failed(id, e) => {
                self.downloads.remove(&id);
                self.refresh(&id);
                ctx.error(format!("{}: {e}", self.label(&id)), Duration::from_secs(10));
            }
            Task::Removed(id, freed) => {
                let ids: Vec<&str> = match SPEAKER_IDS.contains(&id.as_str()) {
                    true => SPEAKER_IDS.to_vec(),
                    false => vec![id.as_str()],
                };
                for asset_id in ids {
                    self.refresh(asset_id);
                }
                ctx.notify_with(
                    format!("Freed {}.", megabytes(freed)),
                    Severity::Information,
                    Duration::from_secs(4),
                    "",
                );
            }
            Task::OllamaText(text) => self.set_models_text(&text),
            Task::PullFailed(model, e) => {
                self.set_models_text(&format!("Pull failed: {e}"));
                ctx.error(
                    format!("Pull failed for {model}: {e}"),
                    Duration::from_secs(10),
                );
            }
            Task::PullDone(model) => {
                ctx.notify_with(
                    format!("{model} is ready."),
                    Severity::Information,
                    Duration::from_secs(5),
                    "",
                );
                self.refresh_ollama(ctx);
            }
            Task::AiTested(check) => self.ai_tested(check, ctx),
            Task::Probe(event) => self.on_probe(event, ctx),
            Task::Edited(result) => self.after_edit(result, ctx),
        }
    }

    // --- Drawing --------------------------------------------------------

    /// A click inside the page box.
    fn click_page(&mut self, input: &MouseInput, ctx: &mut Ctx) -> Action {
        let index = self.page_index();
        let page = &mut self.pages[index];
        if !page.takes_click(input.column, input.row) {
            // Empty page space: the scroll container takes the focus.
            if index != APPEARANCE {
                page.set_focus(None);
                self.focus = Focus::Container;
            }
            return Action::None;
        }
        let event = page.click(input.column, input.row);
        if page.focus.is_some() {
            self.focus = Focus::Page;
        }
        match event {
            Some(event) => self.on_event(event, ctx),
            None => Action::None,
        }
    }

    /// The page list in a rounded box titled "Settings".
    fn draw_nav(&mut self, frame: &mut Frame, area: Rect, theme: &Theme, ctx: &Ctx) {
        let focused = self.focus == Focus::Nav;
        let background = theme.color("background");
        let inner = crate::tui::widgets::rounded(
            frame,
            area,
            "Settings",
            crate::tui::widgets::border_color(focused, theme),
            background,
            theme,
        );
        let rows: Vec<Line> = NAV.iter().map(|label| Line::from(*label)).collect();
        self.nav
            .render(frame, inner, &rows, focused, background, theme);
        ctx.hit(inner, "nav", 0);
    }

    /// The page in a rounded box: its name at the left of the top border, the
    /// config file at the right. Returns the inside.
    fn draw_page_box(&self, frame: &mut Frame, area: Rect, theme: &Theme) -> Rect {
        let focused = matches!(self.focus, Focus::Container | Focus::Page);
        let border = crate::tui::widgets::border_color(focused, theme);
        let background = theme.color("background");
        let name = NAV[self.page_index()];
        // The path gives way to the name: it loses its start first.
        let room = usize::from(area.width).saturating_sub(name.len() + 10);
        let path = tail(&self.config_file().display().to_string(), room);
        let block = Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .style(Style::default().bg(background))
            .border_style(Style::default().fg(border).bg(background))
            .title(Line::from(vec![
                ratatui::text::Span::styled("─", Style::default().fg(border)),
                ratatui::text::Span::styled(
                    format!(" {name} "),
                    Style::default()
                        .fg(theme.color("foreground"))
                        .add_modifier(Modifier::BOLD),
                ),
            ]))
            .title(
                Line::styled(
                    format!(" {path} "),
                    Style::default().fg(theme.on("text-muted", "background")),
                )
                .right_aligned(),
            );
        let inner = block.inner(area);
        frame.render_widget(block, area);
        inner
    }
}

/// The end of `text` in `width` columns, with `…` in front when it is cut.
fn tail(text: &str, width: usize) -> String {
    if crate::tui::text::str_width(text) <= width {
        return text.to_string();
    }
    let mut kept: Vec<char> = Vec::new();
    let mut used = 1;
    for character in text.chars().rev() {
        let cell = crate::tui::text::str_width(&character.to_string());
        if used + cell > width {
            break;
        }
        used += cell;
        kept.push(character);
    }
    let end: String = kept.into_iter().rev().collect();
    match width {
        0 => String::new(),
        _ => format!("…{end}"),
    }
}

/// One estimate for assets removed together. Their older copies are named by their common folder.
fn combined(
    estimates: &[crate::domain::model_asset::RemovalEstimate],
) -> crate::domain::model_asset::RemovalEstimate {
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
    crate::domain::model_asset::RemovalEstimate {
        frees: estimates.iter().map(|e| e.frees).sum(),
        frees_with_legacy: estimates.iter().map(|e| e.frees_with_legacy).sum(),
        legacy_path: common,
    }
}

fn open_privacy(pane: &str) {
    if !cfg!(target_os = "macos") {
        return;
    }
    let url = format!("x-apple.systempreferences:com.apple.preference.security?{pane}");
    if let Err(e) = std::process::Command::new("/usr/bin/open").arg(url).spawn() {
        error!(target: "ltn.config", "Cannot open the privacy settings: {e}");
    }
}

/// A one-row bar with `padding: 0 1`, the text at the left.
fn left_bar(
    frame: &mut Frame,
    area: Rect,
    text: &str,
    background: &str,
    foreground: &str,
    bold: bool,
    theme: &Theme,
) {
    let mut style = Style::default()
        .bg(theme.color(background))
        .fg(theme.on(foreground, background));
    if bold {
        style = style.add_modifier(Modifier::BOLD);
    }
    frame.render_widget(Paragraph::new(format!(" {text}")).style(style), area);
}

/// What Settings popped with: true when it saved at least once.
pub struct SettingsClosed {
    pub saved: bool,
}

impl Screen for SettingsScreen {
    fn mounted(&mut self, ctx: &mut Ctx) -> Action {
        ctx.set_theme(&self.loaded.infra.theme);
        self.baseline = Some(self.collect());
        if !self.loaded.error.is_empty() {
            ctx.error(
                format!(
                    "config.yaml cannot be read ({}). Showing defaults. Press [e] to fix the file.",
                    self.loaded.error
                ),
                Duration::from_secs(20),
            );
        }
        let store = self.store.clone();
        ctx.spawn_thread_or(
            "models-load",
            move || Task::Loaded(store.list_assets().map_err(|e| e.to_string())),
            |message| Task::Loaded(Err(message)),
        );
        Action::None
    }

    fn render(&mut self, frame: &mut Frame, area: Rect, ctx: &Ctx) {
        let theme = &ctx.theme;
        if area.height < 4 {
            return;
        }
        left_bar(
            frame,
            Rect::new(area.x, area.bottom() - 1, area.width, 1),
            "[Ctrl+S] Save  [t] Test AI  [e] Edit raw file  [Esc] Back",
            "surface",
            "text-muted",
            false,
            theme,
        );
        let shell = Rect::new(area.x, area.y, area.width, area.height - 1);
        let nav = Rect::new(shell.x, shell.y, NAV_WIDTH.min(shell.width), shell.height);
        self.draw_nav(frame, nav, theme, ctx);
        let page_box = Rect::new(
            shell.x + nav.width,
            shell.y,
            shell.width - nav.width,
            shell.height,
        );
        ctx.hit(page_box, "page", 0);
        let page_area = self.draw_page_box(frame, page_box, theme);
        let index = self.page_index();
        self.pages[index].render(frame, page_area, theme);
    }

    /// A click on a page row opens it; on the page it focuses and presses
    /// what is under the pointer. The wheel scrolls the page.
    fn mouse(&mut self, input: &MouseInput, ctx: &mut Ctx) -> Action {
        let index = self.page_index();
        let open = self.pages[index].expanded();
        match input.kind {
            MouseKind::WheelUp | MouseKind::WheelDown => {
                if open || input.on("page").is_some() {
                    self.pages[index].wheel(input.kind == MouseKind::WheelDown);
                }
                Action::None
            }
            MouseKind::Click if open => self.click_page(input, ctx),
            MouseKind::Click if input.on("nav").is_some() => {
                if let Some(row) = self.nav.item_at(input.row) {
                    self.pages[index].close_selects();
                    self.nav.select(row);
                    self.focus = Focus::Nav;
                }
                Action::None
            }
            MouseKind::Click if input.on("page").is_some() => self.click_page(input, ctx),
            MouseKind::Click => Action::None,
        }
    }

    fn key(&mut self, key: &KeyEvent, name: &str, ctx: &mut Ctx) -> Action {
        match name {
            "escape" => return self.quit(),
            "ctrl+s" => {
                self.save(ctx);
                return Action::None;
            }
            _ => {}
        }
        let index = self.page_index();
        match self.focus {
            Focus::Page => {
                if let Some(event) = self.pages[index].key(key, name) {
                    return match event {
                        Some(event) => self.on_event(event, ctx),
                        None => Action::None,
                    };
                }
            }
            Focus::Nav if matches!(name, "up" | "down" | "home" | "end") => {
                self.nav.key(name);
                return Action::None;
            }
            Focus::Container
                if matches!(name, "up" | "down" | "pageup" | "pagedown" | "home" | "end") =>
            {
                self.pages[index].scroll_key(name);
                return Action::None;
            }
            _ => {}
        }
        match name {
            "tab" => self.focus_next(true),
            "shift+tab" => self.focus_next(false),
            "t" => self.test_ai(ctx),
            "e" => self.open_raw_editor(ctx),
            _ => {}
        }
        Action::None
    }

    fn paste(&mut self, text: &str, _ctx: &mut Ctx) -> Action {
        if self.focus == Focus::Page {
            self.page().paste(text);
        }
        Action::None
    }

    fn message(&mut self, msg: Msg, ctx: &mut Ctx) -> (Option<Msg>, Action) {
        match msg {
            Msg::Task(payload) => match payload.downcast::<Task>() {
                Ok(task) => {
                    self.on_task(*task, ctx);
                    (None, Action::None)
                }
                Err(payload) => (Some(Msg::Task(payload)), Action::None),
            },
            other => (Some(other), Action::None),
        }
    }

    fn resumed(&mut self, result: Option<Payload>, ctx: &mut Ctx) -> Action {
        let answer = result
            .and_then(|payload| payload.downcast::<Answer>().ok())
            .map(|answer| *answer);
        let confirmed = answer.as_ref().is_some_and(|answer| answer.confirmed);
        if std::mem::take(&mut self.discarding) {
            return match confirmed {
                true => self.leave(),
                false => Action::None,
            };
        }
        if let (Some(asset_id), Some(answer)) = (self.removing.take(), answer)
            && answer.confirmed
        {
            self.remove(asset_id, answer.delete_legacy, ctx);
        }
        Action::None
    }
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
