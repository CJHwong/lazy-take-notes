//! Settings: Revert and Save in the title bar, a page list at the left, and
//! one page of cards at the right. The flow lives in `SettingsRuntime`; this
//! view owns the fields, lends a copy of them to each runtime call, and shows
//! the state the runtime keeps.

mod pages;
mod questions;
#[cfg(all(test, feature = "gui-test"))]
mod tests;

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use gpui_kit::component::IndexPath;
use gpui_kit::component::input::{InputState, TextareaState};
use gpui_kit::component::select::{SelectItem, SelectState};
use gpui_kit::*;
use indexmap::IndexMap;
use tracing::error;

use crate::adapters::assets::store::CachedModelStore;
use crate::adapters::model_catalog::asset_for_model;
use crate::adapters::paths::Paths;
use crate::app::session::Severity;
use crate::app::settings::form::{
    DIARIZATION_DEVICES, FormValues, LLM_PROVIDERS, OpeningValues, THEME_ORDER,
    TRANSCRIPTION_PROVIDERS, load, model_uri, opening_values,
};
use crate::app::settings::rows::{language_choices, select_options};
use crate::app::settings::{Edited, SettingsIo, SettingsRuntime, Step};
use crate::app::tasks::Tasks;
use crate::domain::model_asset::{AssetGroup, ModelAsset};
use crate::gui::bridge::{Bridge, Incoming, notify};
use crate::gui::widgets::DragArm;

pub use questions::ask_discard;

/// How a broken config.yaml notice tells the user to fix it here.
const FIX_HINT: &str = "Use Open Config File to fix it.";

pub enum SettingsEvent {
    Closed,
    /// config.yaml was written: the output folder may have moved.
    Saved,
}

/// The pages, in the list's order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Page {
    Models,
    Digest,
    Audio,
    Sessions,
}

pub const PAGES: [(Page, &str); 4] = [
    (Page::Models, "Models"),
    (Page::Digest, "Notes"),
    (Page::Audio, "Audio"),
    (Page::Sessions, "Sessions"),
];

/// The text fields: id, placeholder, masked.
const INPUTS: [(&str, &str, bool); 16] = [
    ("cfg-ollama-host", "http://localhost:11434", false),
    ("cfg-openai-base-url", "https://api.openai.com/v1", false),
    ("cfg-openai-api-key", "No key stored", true),
    ("cfg-digest-model", "", false),
    ("cfg-interactive-model", "", false),
    (
        "cfg-trans-custom-path",
        "Absolute path to a custom GGML file",
        false,
    ),
    ("cfg-digest-min-lines", "15", false),
    ("cfg-digest-min-interval", "60", false),
    ("cfg-digest-max-lines", "30", false),
    ("cfg-digest-compact", "100000", false),
    ("cfg-trans-chunk", "25.0", false),
    ("cfg-trans-overlap", "1.0", false),
    ("cfg-trans-silence", "0.01", false),
    ("cfg-trans-pause", "1.5", false),
    ("cfg-diarization-interval", "", false),
    ("cfg-output-dir", "~/Documents/LazyTakeNotes", false),
];

/// The dropdowns with a fixed option list. The microphone list is separate.
const SELECTS: [(&str, &[&str]); 4] = [
    ("cfg-llm-provider", &LLM_PROVIDERS),
    ("cfg-trans-provider", &TRANSCRIPTION_PROVIDERS),
    ("cfg-diarization-device", &DIARIZATION_DEVICES),
    // The terminal theme has no page here: the window follows the macOS
    // appearance. Its select still holds the value, so Save keeps it.
    ("cfg-theme", &THEME_ORDER),
];
const DEVICE: &str = "cfg-audio-device";
const HINTS: &str = "cfg-recognition-hints";
const CUSTOM_PATH: &str = "cfg-trans-custom-path";
const API_KEY: &str = "cfg-openai-api-key";

/// One option of a dropdown: what it shows, and the value Save writes.
#[derive(Debug, Clone)]
pub struct Choice {
    label: SharedString,
    value: String,
}

impl SelectItem for Choice {
    type Value = String;

    fn title(&self) -> SharedString {
        self.label.clone()
    }

    fn value(&self) -> &String {
        &self.value
    }
}

type Picker = Entity<SelectState<Vec<Choice>>>;

/// A dropdown over `pairs` (label, value), at `value`.
fn picker(pairs: Vec<(String, String)>, value: &str, window: &mut Window, cx: &mut App) -> Picker {
    let index = pairs.iter().position(|(_, v)| v == value);
    let choices = pairs
        .into_iter()
        .map(|(label, value)| Choice {
            label: label.into(),
            value,
        })
        .collect();
    cx.new(|cx| SelectState::new(choices, index.map(IndexPath::new), window, cx))
}

fn selected(picker: &Picker, cx: &App) -> String {
    picker
        .read(cx)
        .selected_value()
        .cloned()
        .unwrap_or_default()
}

/// One language override: a locale, then its model.
struct LanguageRow {
    locale: Entity<InputState>,
    model: Picker,
}

/// The form as it is now, lent to one runtime call. `FormValues` only reads,
/// so a copy behaves as the fields themselves.
#[derive(Default)]
struct Snapshot {
    text: HashMap<&'static str, String>,
    switches: HashMap<&'static str, bool>,
    hints: String,
    languages: IndexMap<String, String>,
    chosen: String,
}

impl FormValues for Snapshot {
    fn input(&mut self, id: &str) -> String {
        self.text.get(id).cloned().unwrap_or_default()
    }

    fn switch(&mut self, id: &str) -> bool {
        self.switches.get(id).copied().unwrap_or(false)
    }

    fn select(&mut self, id: &str) -> String {
        self.input(id)
    }

    fn text_area(&mut self, _id: &str) -> String {
        self.hints.clone()
    }

    fn language_models(&mut self) -> IndexMap<String, String> {
        self.languages.clone()
    }

    fn chosen_model_uri(&mut self) -> String {
        self.chosen.clone()
    }
}

/// The runtime's notices wait here until the call returns; they then become toasts.
struct Io {
    tasks: Tasks,
    notices: Vec<(String, Severity, Duration)>,
}

impl SettingsIo for Io {
    fn notify(&mut self, text: String, severity: Severity, timeout: Duration) {
        self.notices.push((text, severity, timeout));
    }

    fn tasks(&self) -> Tasks {
        self.tasks.clone()
    }
}

pub struct SettingsView {
    runtime: SettingsRuntime,
    paths: Paths,
    config_path: Option<PathBuf>,
    bridge: Bridge,
    page: Page,
    /// The speech model assets, in catalog order.
    assets: Vec<ModelAsset>,
    /// The microphones the device list offers.
    devices: Vec<String>,
    /// List the microphones again on each reload. Tests keep their own list.
    live_devices: bool,
    inputs: HashMap<&'static str, Entity<InputState>>,
    selects: HashMap<&'static str, Picker>,
    switches: HashMap<&'static str, bool>,
    hints: Entity<TextareaState>,
    languages: Vec<LanguageRow>,
    /// The marked speech model row, by asset id.
    chosen: Option<&'static str>,
    /// The row the config's own model value names: a short name, an id or
    /// an `hf://` URI. While it stays chosen, Save keeps that value as written.
    loaded_choice: Option<&'static str>,
    /// A key is stored, so the clear-key button shows.
    key_stored: bool,
    /// A session runs, so no model or speaker asset may be removed.
    removal_locked: bool,
    focus: FocusHandle,
    drag: DragArm,
    _pump: Task<()>,
}

impl EventEmitter<SettingsEvent> for SettingsView {}

impl SettingsView {
    pub fn new(
        paths: Paths,
        config_path: Option<PathBuf>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Self {
        let store = Arc::new(CachedModelStore::new(&paths));
        let devices = crate::adapters::audio::microphone::list_input_devices();
        let mut view = Self::with(paths, config_path, store, devices, window, cx);
        view.live_devices = true;
        view
    }

    /// Build with a given store and device list, so tests open no audio device.
    fn with(
        paths: Paths,
        config_path: Option<PathBuf>,
        store: Arc<CachedModelStore>,
        devices: Vec<String>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Self {
        let runtime = SettingsRuntime::new(paths.clone(), config_path.clone(), store);
        let assets: Vec<ModelAsset> = runtime
            .store()
            .catalog()
            .iter()
            .filter(|asset| asset.group == AssetGroup::Transcription)
            .cloned()
            .collect();
        let inputs = INPUTS
            .iter()
            .map(|(id, placeholder, masked)| {
                let input = cx.new(|cx| {
                    InputState::new(window, cx)
                        .placeholder(*placeholder)
                        .masked(*masked)
                });
                (*id, input)
            })
            .collect();
        let hints = cx.new(|cx| TextareaState::new(window, cx).auto_grow(4, 12));
        let mut bridge = Bridge::new();
        let pump = bridge.pump(window, cx, Self::incoming);
        let mut view = Self {
            runtime,
            paths,
            config_path,
            bridge,
            page: Page::Models,
            assets,
            devices,
            live_devices: false,
            inputs,
            selects: HashMap::new(),
            switches: HashMap::new(),
            hints,
            languages: Vec::new(),
            chosen: None,
            loaded_choice: None,
            key_stored: false,
            removal_locked: false,
            focus: cx.focus_handle(),
            drag: DragArm::default(),
            _pump: pump,
        };
        let values = opening_values(&view.runtime.loaded, &view.devices);
        view.fill(&values, true, window, cx);
        view.focus.focus(window, cx);
        view.call(window, cx, |runtime, form, io| {
            runtime.mounted(form, io, FIX_HINT)
        });
        view
    }

    // --- The fields -----------------------------------------------------

    /// Put `values` into every field. `custom_path` false keeps the custom
    /// model path as it is, as the TUI does after a hand edit.
    fn fill(
        &mut self,
        values: &OpeningValues,
        custom_path: bool,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        for (id, input) in &self.inputs {
            if *id == CUSTOM_PATH && !custom_path {
                continue;
            }
            let text = values.text(id).to_string();
            input.update(cx, |input, cx| input.set_value(text, window, cx));
        }
        for (id, options) in SELECTS {
            let value = values.text(id);
            let pairs = select_options(options, value)
                .into_iter()
                .map(|name| (name.clone(), name))
                .collect();
            self.selects.insert(id, picker(pairs, value, window, cx));
        }
        let device = picker(
            values.device_options.clone(),
            values.text(DEVICE),
            window,
            cx,
        );
        self.selects.insert(DEVICE, device);
        self.switches = values.switches.iter().map(|(id, on)| (*id, *on)).collect();
        let hints = values.text(HINTS).to_string();
        self.hints
            .update(cx, |area, cx| area.set_value(hints, window, cx));
        self.languages = values
            .language_rows
            .iter()
            .map(|(locale, uri)| self.language_row(locale, uri, window, cx))
            .collect();
        self.chosen = asset_for_model(&values.model, &self.assets).map(|asset| asset.id);
        self.loaded_choice = self.chosen;
        self.key_stored = values.api_key_stored;
        cx.notify();
    }

    fn language_row(
        &self,
        locale: &str,
        uri: &str,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> LanguageRow {
        let (choices, value) = language_choices(&self.assets, uri);
        let locale = locale.to_string();
        LanguageRow {
            locale: cx.new(|cx| {
                InputState::new(window, cx)
                    .placeholder("zh")
                    .default_value(locale)
            }),
            model: picker(choices, &value, window, cx),
        }
    }

    fn snapshot(&self, cx: &App) -> Snapshot {
        let mut text: HashMap<&'static str, String> = self
            .inputs
            .iter()
            .map(|(id, input)| (*id, input.read(cx).value().to_string()))
            .collect();
        for (id, picker) in &self.selects {
            text.insert(id, selected(picker, cx));
        }
        let mut languages = IndexMap::new();
        for row in &self.languages {
            let locale = row.locale.read(cx).value().trim().to_string();
            let model = selected(&row.model, cx);
            if !locale.is_empty() && !model.is_empty() {
                languages.insert(locale, model);
            }
        }
        // Empty keeps the saved value, so an unchanged short name stays short.
        let chosen = match self.chosen == self.loaded_choice {
            true => String::new(),
            false => self
                .assets
                .iter()
                .find(|asset| Some(asset.id) == self.chosen)
                .map(model_uri)
                .unwrap_or_default(),
        };
        Snapshot {
            text,
            switches: self.switches.clone(),
            hints: self.hints.read(cx).value().to_string(),
            languages,
            chosen,
        }
    }

    /// Run one runtime call over a copy of the form, then show its notices.
    fn call<R>(
        &mut self,
        window: &mut Window,
        cx: &mut Context<Self>,
        run: impl FnOnce(&mut SettingsRuntime, &mut Snapshot, &mut Io) -> R,
    ) -> R {
        let mut form = self.snapshot(cx);
        let mut io = Io {
            tasks: self.bridge.tasks.clone(),
            notices: Vec::new(),
        };
        let result = run(&mut self.runtime, &mut form, &mut io);
        for (text, severity, timeout) in io.notices {
            notify(window, cx, text, severity, timeout);
        }
        cx.notify();
        result
    }

    pub fn has_unsaved_changes(&self, cx: &App) -> bool {
        self.runtime.has_unsaved_changes(&mut self.snapshot(cx))
    }

    // --- Commands -------------------------------------------------------

    fn save(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let saved = self.call(window, cx, |runtime, form, io| runtime.save(form, io));
        if let Some(has_key) = saved {
            self.key_stored = has_key;
            cx.emit(SettingsEvent::Saved);
        }
    }

    /// Read config.yaml again and show it. A file that does not parse keeps
    /// the form as it is: the defaults in its place would overwrite the file on Save.
    fn revert(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let loaded = load(&self.paths, self.config_path.as_deref());
        if !loaded.error.is_empty() {
            let path = self.runtime.config_file();
            error!(target: "ltn.config", "Cannot reload {}: {}", path.display(), loaded.error);
            let text = format!("Cannot reload {}: {}", path.display(), loaded.error);
            return notify(window, cx, text, Severity::Error, Duration::from_secs(12));
        }
        self.runtime.loaded = loaded;
        self.runtime.clear_api_key = false;
        self.show_loaded(true, window, cx);
    }

    /// Put the runtime's loaded config into the fields and take it as saved.
    fn show_loaded(&mut self, custom_path: bool, window: &mut Window, cx: &mut Context<Self>) {
        if self.live_devices {
            // A headset plugged in after the pane opened joins the list.
            self.devices = crate::adapters::audio::microphone::list_input_devices();
        }
        let values = opening_values(&self.runtime.loaded, &self.devices);
        self.fill(&values, custom_path, window, cx);
        let mut form = self.snapshot(cx);
        self.runtime.mark_saved_state(&mut form);
    }

    /// Esc: from a field it goes back to the pane, as the TUI goes back to
    /// the page list. From the pane it leaves, after the question when
    /// something is not saved.
    fn back(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if !self.focus.is_focused(window) {
            return self.focus.focus(window, cx);
        }
        if !self.has_unsaved_changes(cx) {
            return cx.emit(SettingsEvent::Closed);
        }
        let view = cx.entity().downgrade();
        ask_discard(window, cx, move |_, cx| {
            let _ = view.update(cx, |_, cx| cx.emit(SettingsEvent::Closed));
        });
    }

    fn clear_key(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if let Some(input) = self.inputs.get(API_KEY) {
            input.update(cx, |input, cx| input.set_value("", window, cx));
        }
        self.call(window, cx, |runtime, _, io| runtime.clear_key(io));
    }

    /// A click on a speech model row marks it. A custom path wins over the
    /// marked row, so marking one clears it.
    fn choose_model(
        &mut self,
        asset_id: &'static str,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.chosen = Some(asset_id);
        if let Some(input) = self.inputs.get(CUSTOM_PATH) {
            input.update(cx, |input, cx| input.set_value("", window, cx));
        }
        cx.notify();
    }

    fn add_language(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let row = self.language_row("", "", window, cx);
        self.languages.push(row);
        cx.notify();
    }

    fn remove_language(&mut self, index: usize, cx: &mut Context<Self>) {
        if index < self.languages.len() {
            self.languages.remove(index);
        }
        cx.notify();
    }

    fn open_cache(&mut self, cx: &mut Context<Self>) {
        cx.open_with_system(self.runtime.store().cache_root());
    }

    /// config.yaml in the system's text editor. The form reads it again when
    /// the editor closes it.
    fn open_config(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let Some(path) = self.call(window, cx, |runtime, _, io| runtime.prepare_raw_edit(io))
        else {
            return;
        };
        self.bridge.tasks.spawn_thread("config-editor", move || {
            // -W waits until the editor closes the file; -t opens the text editor.
            let status = std::process::Command::new("open")
                .args(["-W", "-t"])
                .arg(&path)
                .status();
            Edited(status.map(|_| ()).map_err(|e| {
                error!(target: "ltn.config", "Cannot open the text editor: {e}");
                e.to_string()
            }))
        });
    }

    /// Lock or unlock the Remove buttons while a session runs.
    pub fn lock_removal(&mut self, locked: bool, cx: &mut Context<Self>) {
        self.removal_locked = locked;
        cx.notify();
    }

    fn ask_remove(&mut self, asset_id: &str, window: &mut Window, cx: &mut Context<Self>) {
        if self.removal_locked {
            return;
        }
        let asked = self.call(window, cx, |runtime, _, io| {
            runtime.ask_remove(asset_id, io)
        });
        if let Some((label, estimate)) = asked {
            questions::ask_remove(&label, &estimate, cx.entity().downgrade(), window, cx);
        }
    }

    fn answer_remove(
        &mut self,
        confirmed: bool,
        delete_legacy: bool,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.call(window, cx, |runtime, _, io| {
            runtime.answer_remove(confirmed, delete_legacy, io)
        });
    }

    fn incoming(&mut self, incoming: Incoming, window: &mut Window, cx: &mut Context<Self>) {
        let Incoming::Task(payload) = incoming else {
            return;
        };
        let step = self.call(window, cx, |runtime, form, io| {
            runtime.task_result(payload, form, io)
        });
        if let Ok(Step::Reloaded) = step {
            self.show_loaded(false, window, cx);
        }
    }
}

impl Drop for SettingsView {
    /// The pane went away: stop the downloads and the probe. The `.part` files stay.
    fn drop(&mut self) {
        self.runtime.leave();
    }
}

impl Focusable for SettingsView {
    fn focus_handle(&self, _: &App) -> FocusHandle {
        self.focus.clone()
    }
}
