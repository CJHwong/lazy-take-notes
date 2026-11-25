//! The settings runtime driven without a screen: a map of field values, and a
//! front end that keeps the notices.

use std::sync::Mutex;

use super::*;

/// The form as a map of field values; a field not in the map is empty.
#[derive(Default)]
struct Form {
    fields: HashMap<&'static str, String>,
    switches: HashMap<&'static str, bool>,
    languages: IndexMap<String, String>,
}

impl FormValues for Form {
    fn input(&mut self, id: &str) -> String {
        self.fields.get(id).cloned().unwrap_or_default()
    }

    fn switch(&mut self, id: &str) -> bool {
        self.switches.get(id).copied().unwrap_or(false)
    }

    fn select(&mut self, id: &str) -> String {
        self.input(id)
    }

    fn text_area(&mut self, id: &str) -> String {
        self.input(id)
    }

    fn language_models(&mut self) -> IndexMap<String, String> {
        self.languages.clone()
    }

    fn chosen_model_uri(&mut self) -> String {
        self.input("chosen-model")
    }
}

/// A form that passes every check on Save.
fn valid_form(root: &Path) -> Form {
    let model = root.join("model.bin");
    std::fs::write(&model, b"").unwrap();
    let mut form = Form::default();
    for (id, value) in [
        ("cfg-llm-provider", "ollama"),
        ("cfg-trans-provider", "whisper-cpp"),
        ("cfg-theme", "textual-dark"),
        ("cfg-audio-device", DEFAULT_DEVICE),
        ("cfg-diarization-device", "auto"),
        ("cfg-trans-custom-path", model.to_str().unwrap()),
    ] {
        form.fields.insert(id, value.into());
    }
    form.switches.insert("cfg-output-save-audio", true);
    form
}

/// The front end: notices kept as text, task results kept for the test to hand back.
struct Front {
    notices: Vec<(Severity, String)>,
    results: Arc<Mutex<Vec<Payload>>>,
    runtime: tokio::runtime::Runtime,
}

impl Front {
    fn new() -> Self {
        Self {
            notices: Vec::new(),
            results: Arc::new(Mutex::new(Vec::new())),
            runtime: tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap(),
        }
    }

    fn texts(&self) -> Vec<&str> {
        self.notices.iter().map(|(_, text)| text.as_str()).collect()
    }
}

impl SettingsIo for Front {
    fn notify(&mut self, text: String, severity: Severity, _timeout: Duration) {
        self.notices.push((severity, text));
    }

    fn tasks(&self) -> Tasks {
        let results = self.results.clone();
        Tasks::new(
            self.runtime.handle().clone(),
            Arc::new(move |payload| results.lock().unwrap().push(payload)),
            Arc::new(|_| false),
        )
    }
}

fn settings(root: &Path) -> SettingsRuntime {
    let paths = Paths::under(root);
    let store = Arc::new(CachedModelStore::new(&paths));
    SettingsRuntime::new(paths, None, store)
}

fn first_speech_model(runtime: &SettingsRuntime) -> &'static str {
    runtime
        .store()
        .catalog()
        .iter()
        .find(|asset| asset.group == AssetGroup::Transcription)
        .unwrap()
        .id
}

#[test]
fn a_payload_from_elsewhere_is_given_back() {
    let root = tempfile::tempdir().unwrap();
    let mut runtime = settings(root.path());
    let (mut form, mut front) = (Form::default(), Front::new());
    let result = runtime.task_result(Box::new(7_u8), &mut form, &mut front);
    assert!(result.is_err_and(|payload| payload.downcast::<u8>().is_ok()));
}

#[test]
fn save_refuses_a_field_that_is_not_a_number() {
    let root = tempfile::tempdir().unwrap();
    let mut runtime = settings(root.path());
    let mut form = valid_form(root.path());
    form.fields.insert("cfg-trans-chunk", "soon".into());
    let mut front = Front::new();
    assert_eq!(runtime.save(&mut form, &mut front), None);
    assert_eq!(front.texts(), ["Chunk duration must be a number."]);
    assert!(!runtime.config_file().exists());
    assert!(!runtime.saved);
}

#[test]
fn save_refuses_speaker_labels_without_audio() {
    let root = tempfile::tempdir().unwrap();
    let mut runtime = settings(root.path());
    let mut form = valid_form(root.path());
    form.switches.insert("cfg-diarization-enabled", true);
    form.switches.insert("cfg-output-save-audio", false);
    let mut front = Front::new();
    assert_eq!(runtime.save(&mut form, &mut front), None);
    assert_eq!(front.texts(), ["Speaker labels need Save audio turned on."]);
}

#[test]
fn save_writes_the_file_and_takes_the_form_as_saved() {
    let root = tempfile::tempdir().unwrap();
    let mut runtime = settings(root.path());
    let mut form = valid_form(root.path());
    let mut front = Front::new();
    runtime.mounted(&mut form, &mut front);
    form.fields.insert("cfg-digest-model", "llama3".into());
    assert!(runtime.has_unsaved_changes(&mut form));
    assert_eq!(runtime.save(&mut form, &mut front), Some(false));
    assert!(runtime.saved);
    assert!(!runtime.has_unsaved_changes(&mut form));
    let written = std::fs::read_to_string(runtime.config_file()).unwrap();
    assert!(written.contains("llama3"), "{written}");
    assert!(front.texts()[0].starts_with("Saved to "));
}

#[test]
fn clearing_the_key_writes_it_as_removed() {
    let root = tempfile::tempdir().unwrap();
    let mut runtime = settings(root.path());
    let mut form = valid_form(root.path());
    let mut front = Front::new();
    runtime.clear_key(&mut front);
    assert_eq!(
        runtime.collect(&mut form)["openai"]["api_key"],
        Value::Null,
        "the key goes as null"
    );
    form.fields
        .insert("cfg-openai-api-key", "typed-after-clear".into());
    assert_eq!(
        runtime.save(&mut form, &mut front),
        Some(false),
        "a cleared key stays cleared, so its row hides"
    );
    assert!(!runtime.clear_api_key);
}

#[test]
fn a_cancelled_download_keeps_its_bytes_and_reads_paused() {
    let root = tempfile::tempdir().unwrap();
    let mut runtime = settings(root.path());
    let id = first_speech_model(&runtime);
    let (mut form, mut front) = (Form::default(), Front::new());
    let cancel = Arc::new(AtomicBool::new(false));
    runtime.downloads.insert(id.to_string(), cancel);
    let progress = AssetProgress {
        downloaded: 3 * MEGABYTE,
        total: 0,
        bytes_per_second: 0.0,
    };
    let posted: Payload = Box::new(Task::Progress(id.to_string(), progress));
    assert!(runtime.task_result(posted, &mut form, &mut front).is_ok());
    let row = &runtime.rows[id];
    assert_eq!(row.kept, 3 * MEGABYTE);
    assert!(
        row.progress.is_some_and(|p| p.total > 0),
        "the catalog size stands in"
    );
    let posted: Payload = Box::new(Task::Cancelled(id.to_string()));
    assert!(runtime.task_result(posted, &mut form, &mut front).is_ok());
    assert!(!runtime.downloading(id));
    assert!(runtime.rows[id].paused);
    assert_eq!(
        front.texts(),
        [format!("Download cancelled: {}.", runtime.label(id))]
    );
}

#[test]
fn speaker_progress_adds_both_assets() {
    let root = tempfile::tempdir().unwrap();
    let mut runtime = settings(root.path());
    let (mut form, mut front) = (Form::default(), Front::new());
    for id in SPEAKER_IDS {
        let progress = AssetProgress {
            downloaded: MEGABYTE,
            total: 2 * MEGABYTE,
            bytes_per_second: 0.0,
        };
        let posted: Payload = Box::new(Task::Progress(id.to_string(), progress));
        assert!(runtime.task_result(posted, &mut form, &mut front).is_ok());
    }
    assert_eq!(
        runtime.speaker.status,
        format!(
            "Downloading {} / {}",
            megabytes(2 * MEGABYTE),
            megabytes(4 * MEGABYTE)
        )
    );
    assert!(runtime.speaker.cancel);
    assert!(!runtime.speaker.install);
}

#[test]
fn leave_stops_the_downloads_and_the_probe() {
    let root = tempfile::tempdir().unwrap();
    let mut runtime = settings(root.path());
    let download = Arc::new(AtomicBool::new(false));
    let probe = Arc::new(AtomicBool::new(false));
    runtime.downloads.insert("any".into(), download.clone());
    runtime.probe_stop = Some(probe.clone());
    assert!(!runtime.leave());
    assert!(download.load(Ordering::SeqCst));
    assert!(probe.load(Ordering::SeqCst));
}

#[test]
fn probe_events_fill_the_meters() {
    let root = tempfile::tempdir().unwrap();
    let mut runtime = settings(root.path());
    let (mut form, mut front) = (Form::default(), Front::new());
    runtime.probe_stop = Some(Arc::new(AtomicBool::new(false)));
    for event in [
        ProbeEvent::Level(Source::Mic, 0.5),
        ProbeEvent::Permission(Source::System, "waiting"),
        ProbeEvent::Done,
    ] {
        let posted: Payload = Box::new(Task::Probe(event));
        assert!(runtime.task_result(posted, &mut form, &mut front).is_ok());
    }
    assert!(runtime.meters[0].live);
    assert_eq!(runtime.meters[0].levels.back(), Some(&0.5));
    assert!(!runtime.meters[1].live);
    assert_eq!(
        runtime.meters[1].permission.map(|p| p.text),
        Some("Waiting for permission")
    );
    assert!(!runtime.probe_running());
}

#[test]
fn after_the_editor_the_file_is_read_again() {
    let root = tempfile::tempdir().unwrap();
    let mut runtime = settings(root.path());
    let (mut form, mut front) = (Form::default(), Front::new());
    let failed: Payload = Box::new(Edited(Err("no such program".into())));
    let step = runtime.task_result(failed, &mut form, &mut front).ok();
    assert_eq!(step, Some(Step::None));
    assert_eq!(front.texts(), ["Cannot open editor: no such program"]);
    let path = runtime
        .prepare_raw_edit(&mut front)
        .expect("the file is created");
    std::fs::write(&path, "theme: nord\n").unwrap();
    let edited: Payload = Box::new(Edited(Ok(())));
    let step = runtime.task_result(edited, &mut form, &mut front).ok();
    assert_eq!(step, Some(Step::Reloaded));
    assert_eq!(runtime.theme(), "nord");
}

/// A form filled with what each field shows at open.
fn opened_form(values: &form::OpeningValues) -> Form {
    Form {
        fields: values
            .text
            .iter()
            .map(|(id, value)| (*id, value.clone()))
            .chain([("chosen-model", values.model.clone())])
            .collect(),
        switches: values.switches.iter().map(|(id, on)| (*id, *on)).collect(),
        languages: values
            .language_rows
            .iter()
            .filter(|(locale, model)| !locale.trim().is_empty() && !model.is_empty())
            .map(|(locale, model)| (locale.trim().to_string(), model.clone()))
            .collect(),
    }
}

#[test]
fn the_opening_values_collect_back_to_the_config_python_opens() {
    const FIXTURES: &str = include_str!("../../../tests/fixtures/settings.json");
    let cases: Vec<serde_json::Value> = serde_json::from_str(FIXTURES).unwrap();
    for case in cases {
        let name = case["name"].as_str().unwrap();
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::under(root.path());
        if let Some(text) = case["config"].as_str() {
            std::fs::create_dir_all(&paths.config_dir).unwrap();
            std::fs::write(paths.config_dir.join("config.yaml"), text).unwrap();
        }
        let loaded = load(&paths, None);
        let values = form::opening_values(&loaded, &[]);
        let collected = collect(&mut opened_form(&values), &loaded.raw, false);
        let collected = serde_json::to_value(&collected).unwrap();
        assert_eq!(collected, case["opened"], "{name}: the config at open");
    }
}

#[test]
fn opening_values_show_a_custom_model_in_its_own_field() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::under(root.path());
    std::fs::create_dir_all(&paths.config_dir).unwrap();
    let text = "transcription:\n  model: /models/mine.bin\n  input_device: Gone mic\n";
    std::fs::write(paths.config_dir.join("config.yaml"), text).unwrap();
    let values = form::opening_values(&load(&paths, None), &["Desk".into()]);
    assert_eq!(values.text("cfg-trans-custom-path"), "/models/mine.bin");
    assert_eq!(values.text("cfg-diarization-interval"), "60");
    assert_eq!(values.text("cfg-audio-device"), "Gone mic");
    let labels: Vec<&str> = values
        .device_options
        .iter()
        .map(|(label, _)| label.as_str())
        .collect();
    assert_eq!(labels, ["System default", "Desk", "Gone mic"]);
}
