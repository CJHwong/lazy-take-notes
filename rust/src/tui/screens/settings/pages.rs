//! The five settings pages as item tables, in Python's compose order.

use std::path::Path;

use super::form::{Button, Item, Kind, Page};
use super::models::{LanguageEditor, ModelChooser, SpeakerState};
use super::preview::Preview;
use crate::app::settings::form::{
    DIARIZATION_DEVICES, LLM_PROVIDERS, OpeningValues, THEME_ORDER, TRANSCRIPTION_PROVIDERS,
};
use crate::domain::model_asset::ModelAsset;
use crate::tui::widgets::input::Input;
use crate::tui::widgets::select::Select;
use crate::tui::widgets::text_area::TextArea;

/// What the pages are built from.
pub struct Source<'a> {
    /// What each field shows.
    pub values: &'a OpeningValues,
    pub catalog: &'a [ModelAsset],
    pub cache_root: &'a Path,
}

/// A section heading. It draws its own rule.
fn title(id: &'static str, text: &str) -> [Item; 1] {
    [Item::new(id, Kind::Title(text.into()))]
}

fn desc(id: &'static str, text: &str) -> Item {
    Item::new(id, Kind::Desc(text.into()))
}

fn field(id: &'static str, label: &str, value: &str, placeholder: &str) -> Item {
    let mut input = Input::new(placeholder);
    input.compact = Some(("surface-lighten-1", "surface-lighten-2"));
    input.set_value(value);
    Item::new(
        id,
        Kind::Field {
            label: label.into(),
            input,
            help: String::new(),
        },
    )
}

fn switch(id: &'static str, label: &str, on: bool) -> Item {
    Item::new(
        id,
        Kind::Switch {
            label: label.into(),
            on,
            help: String::new(),
        },
    )
}

/// `SelectRow`: the value joins the options when it is not one of them.
fn select_row(id: &'static str, label: &str, options: &[&str], value: &str, help: &str) -> Item {
    let mut names: Vec<String> = options.iter().map(|name| name.to_string()).collect();
    if !names.iter().any(|name| name == value) {
        names.push(value.into());
    }
    let pairs = names.into_iter().map(|name| (name.clone(), name)).collect();
    Item::new(
        id,
        Kind::Select {
            label: Some(label.into()),
            select: Select::new(pairs, value),
            help: help.into(),
        },
    )
}

fn buttons(id: &'static str, gap: u16, list: &[(&str, &str)]) -> Item {
    Item::new(
        id,
        Kind::Buttons {
            gap,
            buttons: list
                .iter()
                .map(|(id, label)| Button::new(id, label))
                .collect(),
        },
    )
}

pub fn models_page(source: &Source) -> Page {
    let values = source.values;
    let transcription_assets: Vec<ModelAsset> = source
        .catalog
        .iter()
        .filter(|asset| asset.group == crate::domain::model_asset::AssetGroup::Transcription)
        .cloned()
        .collect();
    let mut api_key = field(
        "cfg-openai-api-key",
        "API key",
        values.text("cfg-openai-api-key"),
        "No key stored",
    );
    if let Kind::Field { input, .. } = &mut api_key.kind {
        input.password = true;
    }
    let mut custom = Input::new("Absolute path to a custom GGML file");
    custom.compact = Some(("surface-lighten-1", "surface-lighten-2"));
    custom.set_value(values.text("cfg-trans-custom-path"));
    let mut items = Vec::new();
    items.extend(title("", "Text generation"));
    items.push(select_row(
        "cfg-llm-provider",
        "Provider",
        &LLM_PROVIDERS,
        values.text("cfg-llm-provider"),
        "Ollama runs a model here, or a cloud model after ollama signin. OpenAI needs an API key.",
    ));
    items.push(field(
        "cfg-ollama-host",
        "Server",
        values.text("cfg-ollama-host"),
        "http://localhost:11434",
    ));
    items.push(field(
        "cfg-openai-base-url",
        "API base",
        values.text("cfg-openai-base-url"),
        "https://api.openai.com/v1",
    ));
    items.push(api_key);
    items.push(
        buttons(
            "clear-key-row",
            0,
            &[("cfg-openai-clear-key", "Clear stored key")],
        )
        .margins(0, 0)
        .hidden(!values.api_key_stored),
    );
    items.push(field(
        "cfg-digest-model",
        "Summary model",
        values.text("cfg-digest-model"),
        "",
    ));
    items.push(field(
        "cfg-interactive-model",
        "Quick action model",
        values.text("cfg-interactive-model"),
        "",
    ));
    items.push(desc(
        "cfg-ollama-models",
        "Press Refresh to list the models on the server.",
    ));
    items.push(buttons(
        "ollama-buttons",
        1,
        &[
            ("cfg-ai-test", "Test"),
            ("cfg-ollama-refresh", "Refresh"),
            ("cfg-ollama-pull-digest", "Pull summary"),
            ("cfg-ollama-pull-interactive", "Pull quick action"),
        ],
    ));
    items.extend(title("", "Speech-to-Text"));
    items.push(select_row(
        "cfg-trans-provider",
        "Engine",
        &TRANSCRIPTION_PROVIDERS,
        values.text("cfg-trans-provider"),
        "whisper-cpp runs locally. Plugin engines appear when installed.",
    ));
    items.push(Item::new("", Kind::Label("Default model".into())).margins(0, 0));
    items.push(Item::new(
        "models-chooser",
        Kind::Custom(Box::new(ModelChooser::new(
            transcription_assets.clone(),
            &values.model,
        ))),
    ));
    items.push(buttons(
        "download-all-row",
        1,
        &[("models-download-all", "Download all missing")],
    ));
    items.extend(title("", "Language models"));
    items.push(desc(
        "",
        "Use a different model for a locale. The default applies otherwise.",
    ));
    items.push(Item::new(
        "models-language",
        Kind::Custom(Box::new(LanguageEditor::new(
            &values.language_rows,
            transcription_assets,
        ))),
    ));
    items.extend(title("", "Speaker assets"));
    items.push(
        Item::new(
            "speaker-state",
            Kind::Custom(Box::new(SpeakerState::default())),
        )
        .margins(0, 1),
    );
    items.extend(title("", "Advanced"));
    items.push(Item::new("cfg-trans-custom-path", Kind::Input(custom)));
    items.push(desc(
        "cache-path",
        &format!("Cache: {}", source.cache_root.display()),
    ));
    items.push(
        buttons(
            "open-cache-row",
            0,
            &[("models-open-cache", "Open cache folder")],
        )
        .margins(0, 0),
    );
    Page::new(items)
}

pub fn digest_page(values: &OpeningValues) -> Page {
    let mut items = Vec::new();
    items.extend(title("", "When to summarize"));
    items.push(field(
        "cfg-digest-min-lines",
        "After lines",
        values.text("cfg-digest-min-lines"),
        "15",
    ));
    items.push(field(
        "cfg-digest-min-interval",
        "Wait seconds",
        values.text("cfg-digest-min-interval"),
        "60",
    ));
    items.push(field(
        "cfg-digest-max-lines",
        "Force after",
        values.text("cfg-digest-max-lines"),
        "30",
    ));
    items.push(desc(
        "",
        "Force after is optional. Blank means twice After lines.",
    ));
    items.extend(title("", "Advanced"));
    items.push(field(
        "cfg-digest-compact",
        "Compaction threshold",
        values.text("cfg-digest-compact"),
        "100000",
    ));
    items.push(desc(
        "",
        "Compress history when it exceeds this token count. Change it only if you hit context-length errors.",
    ));
    Page::new(items)
}

pub fn audio_page(values: &OpeningValues) -> Page {
    let device = Select::new(
        values.device_options.clone(),
        values.text("cfg-audio-device"),
    );
    let meter = || "▁".repeat(12);
    let mut items = Vec::new();
    items.extend(title("", "Input check"));
    items.push(desc(
        "",
        "Start a 15 second test and make a sound. Nothing is saved.",
    ));
    items.push(buttons(
        "probe-buttons",
        2,
        &[("audio-start", "Start test"), ("audio-stop", "Stop")],
    ));
    items.push(Item::new("", Kind::Text("Microphone".into(), None)));
    items.push(Item::new("audio-meter-mic", Kind::Text(meter(), Some("accent"))).margins(0, 1));
    items.push(Item::new(
        "audio-perm-mic",
        Kind::Text("Not tested".into(), None),
    ));
    items.push(Item::new("", Kind::Text("System audio".into(), None)));
    items.push(Item::new("audio-meter-system", Kind::Text(meter(), Some("accent"))).margins(0, 1));
    items.push(Item::new(
        "audio-perm-system",
        Kind::Text("Not tested".into(), None),
    ));
    items.push(buttons(
        "grant-buttons",
        2,
        &[
            ("audio-grant-system", "Grant system audio"),
            ("audio-grant-mic", "Open microphone settings"),
        ],
    ));
    items.extend(title("", "Microphone"));
    items.push(Item::new(
        "cfg-audio-device",
        Kind::Select {
            label: None,
            select: device,
            help: String::new(),
        },
    ));
    items.push(desc(
        "",
        "A removed device falls back to the system default with a warning.",
    ));
    items.extend(title("", "Audio processing"));
    items.push(field(
        "cfg-trans-chunk",
        "Chunk duration",
        values.text("cfg-trans-chunk"),
        "25.0",
    ));
    items.push(field(
        "cfg-trans-overlap",
        "Overlap",
        values.text("cfg-trans-overlap"),
        "1.0",
    ));
    items.push(field(
        "cfg-trans-silence",
        "Silence threshold",
        values.text("cfg-trans-silence"),
        "0.01",
    ));
    items.push(field(
        "cfg-trans-pause",
        "Pause duration",
        values.text("cfg-trans-pause"),
        "1.5",
    ));
    items.extend(title("", "Speaker labels"));
    items.push(switch(
        "cfg-diarization-enabled",
        "Label speakers",
        values.switch("cfg-diarization-enabled"),
    ));
    items.push(select_row(
        "cfg-diarization-device",
        "Device",
        &DIARIZATION_DEVICES,
        values.text("cfg-diarization-device"),
        "",
    ));
    items.push(field(
        "cfg-diarization-interval",
        "Update every",
        values.text("cfg-diarization-interval"),
        "",
    ));
    let mut page = Page::new(items);
    if let Some(button) = page.button_mut("audio-stop") {
        button.disabled = true;
    }
    page
}

pub fn sessions_page(values: &OpeningValues) -> Page {
    let mut area = TextArea::default();
    area.set_text(values.text("cfg-recognition-hints"));
    let mut items = Vec::new();
    items.extend(title("", "Session storage"));
    items.push(field(
        "cfg-output-dir",
        "Output folder",
        values.text("cfg-output-dir"),
        "~/Documents/LazyTakeNotes",
    ));
    items.push(switch(
        "cfg-output-save-audio",
        "Save audio",
        values.switch("cfg-output-save-audio"),
    ));
    items.push(switch(
        "cfg-output-save-notes-history",
        "Keep notes history",
        values.switch("cfg-output-save-notes-history"),
    ));
    items.push(switch(
        "cfg-output-save-context",
        "Save session context",
        values.switch("cfg-output-save-context"),
    ));
    items.push(switch(
        "cfg-output-save-debug-log",
        "Save debug log",
        values.switch("cfg-output-save-debug-log"),
    ));
    items.push(switch(
        "cfg-output-auto-label",
        "Auto-label sessions",
        values.switch("cfg-output-auto-label"),
    ));
    items.extend(title("", "Custom vocabulary"));
    items.push(desc(
        "",
        "Names, acronyms, or jargon the recognizer should watch for. One per line.",
    ));
    items.push(Item::new("cfg-recognition-hints", Kind::TextArea(area)));
    items.extend(title("", "Templates"));
    items.push(switch(
        "cfg-show-builtin-templates",
        "Show built-in templates",
        values.switch("cfg-show-builtin-templates"),
    ));
    items.push(desc(
        "",
        "Bundled examples appear in the picker. Turn it off to see only your own.",
    ));
    Page::new(items)
}

pub fn appearance_page(values: &OpeningValues) -> Page {
    let mut theme = select_row(
        "cfg-theme",
        "Theme",
        &THEME_ORDER,
        values.text("cfg-theme"),
        "",
    );
    theme.max_width = Some(70);
    let mut items = Vec::new();
    items.extend(title("", "Theme"));
    items.push(theme);
    items.push(Item::new("preview", Kind::Custom(Box::new(Preview::new()))));
    let mut page = Page::new(items);
    // `AppearanceTab` is a Vertical: it does not scroll, and the preview fills the rest.
    page.fill_last = true;
    page.max_width = u16::MAX;
    page
}
