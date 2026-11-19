//! `record --headless` and `transcribe --headless`: build the session, run it, report.
//!
//! Each function returns the exit code. `Err` is the text the CLI prints after
//! "Error: " with exit code 1, as click prints a `ClickException`.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use serde_json::Value as JsonValue;

use crate::adapters::assets::resolver::CachedModelResolver;
use crate::adapters::assets::store::CachedModelStore;
use crate::adapters::file_persistence::FilePersistence;
use crate::adapters::pyrepr::py_repr_json;
use crate::adapters::session_controller::SessionController;
use crate::adapters::session_status::STATUS_FILE;
use crate::adapters::templates::YamlTemplates;
use crate::adapters::whisper::transcriber::SubprocessWhisperTranscriber;
use crate::app::container::{
    Loaded, build_llm_client, build_mixed_source, check_transcription_provider, load_config,
    make_session_dir, resolve_base_dir,
};
use crate::app::events::{Event, Poster, WorkerState};
use crate::app::headless::{
    HeadlessSession, Launcher, StatusWriter, banner_lines, holds_only_status, summary_lines,
};
use crate::app::keep_awake::keep_awake;
use crate::app::logging::{close_file_logging, setup_file_logging};
use crate::app::preflight::{PreflightPlan, format_report_line, run_preflight};
use crate::app::workers::audio::{AudioWorkerConfig, LoopTiming, run_audio_worker};
use crate::app::workers::file_transcription::{FileJob, download_percent, run_file_transcription};
use crate::domain::config::AppConfig;
use crate::domain::session_status as st;
use crate::domain::template::SessionTemplate;
use crate::ports::model_resolver::ModelResolver;
use crate::ports::transcriber::Transcriber;
use crate::usecases::transcribe_audio::TranscribeSettings;

/// What the CLI passes in.
pub struct HeadlessArgs<'a> {
    pub config_path: Option<&'a Path>,
    pub output_dir: Option<&'a str>,
    pub template_name: Option<&'a str>,
    pub language: Option<&'a str>,
    pub label: Option<&'a str>,
}

fn quoted(text: &str) -> String {
    py_repr_json(&JsonValue::String(text.into()))
}

/// A template by exact key, or an error that lists the keys.
pub fn load_template_key(templates: &YamlTemplates, key: &str) -> Result<SessionTemplate, String> {
    if templates.all_names().contains(key) {
        return templates.load_template(key).map_err(|e| e.to_string());
    }
    let available: Vec<String> = templates.all_names().into_iter().collect();
    Err(format!(
        "Unknown template {}. Available: {}",
        quoted(key),
        available.join(", ")
    ))
}

/// The default template for a locale: the full locale first (`zh-TW`), then the
/// primary subtag when only one template has it (`en-US` to `en`).
pub fn resolve_language(
    templates: &YamlTemplates,
    language: &str,
) -> Result<SessionTemplate, String> {
    let defaults: Vec<(String, String)> = templates
        .all_names()
        .into_iter()
        .filter(|name| name.starts_with("default_"))
        .filter_map(|name| {
            let template = templates.load_template(&name).ok()?;
            Some((template.metadata.locale.to_lowercase(), name))
        })
        .collect();
    let wanted = language.to_lowercase();
    let primary = |locale: &str| locale.split('-').next().unwrap_or("").to_string();
    let exact = defaults.iter().find(|(locale, _)| *locale == wanted);
    let by_subtag: Vec<_> = defaults
        .iter()
        .filter(|(locale, _)| primary(locale) == primary(&wanted))
        .collect();
    let found = exact.or((by_subtag.len() == 1).then(|| by_subtag[0]));
    let Some((_, name)) = found else {
        let mut locales: Vec<String> = defaults
            .iter()
            .filter_map(|(_, name)| templates.load_template(name).ok())
            .map(|t| t.metadata.locale)
            .collect();
        locales.sort();
        locales.dedup();
        return Err(format!(
            "Unsupported language {}. Available: {}",
            quoted(language),
            locales.join(", ")
        ));
    };
    templates.load_template(name).map_err(|e| e.to_string())
}

/// --language, then --template, then default_en.
pub(crate) fn select_template(
    templates: &YamlTemplates,
    args: &HeadlessArgs<'_>,
) -> Result<SessionTemplate, String> {
    match args.language {
        Some(language) => resolve_language(templates, language),
        None => load_template_key(templates, args.template_name.unwrap_or("default_en")),
    }
}

pub(crate) fn settings(config: &AppConfig) -> TranscribeSettings {
    let transcription = &config.transcription;
    TranscribeSettings {
        chunk_duration: transcription.chunk_duration,
        overlap: transcription.overlap,
        silence_threshold: transcription.silence_threshold,
        pause_duration: transcription.pause_duration,
        ..TranscribeSettings::default()
    }
}

pub(crate) fn language_of(template: &SessionTemplate) -> String {
    template
        .metadata
        .locale
        .split('-')
        .next()
        .unwrap_or("")
        .to_lowercase()
}

/// The config's hints, then the template's, without repeats.
pub(crate) fn hints(config: &AppConfig, template: &SessionTemplate) -> Vec<String> {
    let mut all: Vec<String> = Vec::new();
    for hint in config
        .recognition_hints
        .iter()
        .chain(&template.recognition_hints)
    {
        if !all.contains(hint) {
            all.push(hint.clone());
        }
    }
    all
}

fn eprint_lines(lines: &[String]) {
    for line in lines {
        eprintln!("{line}");
    }
}

/// A resolver that posts download progress as events.
pub(crate) fn resolver_posting(
    store: Arc<CachedModelStore>,
    poster: &Poster,
    model_name: &str,
) -> CachedModelResolver {
    let (poster, name) = (poster.clone(), model_name.to_string());
    CachedModelResolver::new(
        store,
        Some(Box::new(move |progress| {
            if let Some(percent) = download_percent(&progress) {
                poster.post(Event::ModelDownloadProgress {
                    percent,
                    model_name: name.clone(),
                });
            }
        })),
    )
}

pub fn new_runtime() -> Result<tokio::runtime::Runtime, String> {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|e| e.to_string())
}

fn config_and_template(args: &HeadlessArgs<'_>) -> Result<(Loaded, SessionTemplate), String> {
    let loaded = load_config(args.config_path, args.output_dir)?;
    let template = select_template(&YamlTemplates::new(&loaded.paths), args)?;
    Ok((loaded, template))
}

pub fn record(args: &HeadlessArgs<'_>, mute_mic: bool) -> Result<i32, String> {
    let (loaded, template) = config_and_template(args)?;
    let Loaded {
        config,
        infra,
        paths,
    } = loaded;
    check_transcription_provider(&infra)?;
    let runtime = new_runtime()?;
    let llm = build_llm_client(&infra)?;
    let store = Arc::new(CachedModelStore::new(&paths));
    let plan = PreflightPlan {
        config: config.clone(),
        template: template.clone(),
        base_dir: resolve_base_dir(&paths, args.output_dir, &config),
        llm_client: llm.clone(),
        llm_provider: infra.llm_provider.clone(),
        asset_store: store.clone(),
        runtime: runtime.handle().clone(),
        mic_muted: mute_mic,
        cancel: Arc::default(),
    };
    let mut transcriber: Box<dyn Transcriber> =
        Box::new(SubprocessWhisperTranscriber::new().map_err(|e| e.to_string())?);
    let report = run_preflight(&plan, transcriber.as_mut(), &|_| {});
    for check in &report.checks {
        eprintln!("{}", format_report_line(check));
    }
    if report.blocked() {
        transcriber.close();
        return Err("Preflight failed; fix the items marked FAIL and run again.".into());
    }
    for check in report.failures() {
        eprintln!("⚠ continuing despite: {}", check.title);
    }
    if !report.llm_ok() {
        eprintln!("⚠ AI notes off: recording the transcript only");
    }
    let out_dir = make_session_dir(&plan.base_dir, args.label).map_err(|e| e.to_string())?;
    let persistence = FilePersistence::new(out_dir.clone()).map_err(|e| e.to_string())?;
    let controller =
        SessionController::new(config.clone(), template.clone(), llm, Box::new(persistence));
    let source = build_mixed_source(&config);
    if mute_mic {
        crate::ports::audio_source::AudioSource::set_mic_muted(source.as_ref(), true);
    }
    let model_name = config
        .transcription
        .model_for_locale(&template.metadata.locale)
        .to_string();
    let worker_config = AudioWorkerConfig {
        model_path: String::new(),
        language: language_of(&template),
        settings: settings(&config),
        recognition_hints: hints(&config, &template),
        folder: Some(Arc::new(
            crate::adapters::audio::session_folder::SessionFolder::new(out_dir.clone()),
        )),
        save_audio: config.output.save_audio,
        timing: LoopTiming::default(),
    };
    let worker_source = source.clone();
    let launcher: Launcher = Box::new(move |poster, cancel| {
        let resolved = resolver_posting(store, &poster, &model_name).resolve(&model_name);
        let model_path = match resolved {
            Ok(path) => path,
            Err(e) => {
                poster.post(Event::status_error(WorkerState::Error, e.to_string()));
                return;
            }
        };
        let paused = AtomicBool::new(false);
        let config = AudioWorkerConfig {
            model_path,
            ..worker_config
        };
        run_audio_worker(
            &poster,
            &cancel,
            &paused,
            config,
            transcriber,
            worker_source,
        );
    });
    let session = HeadlessSession::new(
        controller,
        out_dir.clone(),
        "recording",
        report.llm_ok(),
        Some(source),
        StatusWriter::stderr(),
    );
    eprint_lines(&banner_lines(
        "recording",
        &out_dir,
        session.status().pid,
        true,
    ));
    let _awake = keep_awake();
    Ok(run_session(
        &runtime, &out_dir, &config, session, launcher, false,
    ))
}

pub fn transcribe(args: &HeadlessArgs<'_>, audio_path: &Path) -> Result<i32, String> {
    let (loaded, template) = config_and_template(args)?;
    let Loaded {
        config,
        infra,
        paths,
    } = loaded;
    check_transcription_provider(&infra)?;
    let base_dir = resolve_base_dir(&paths, args.output_dir, &config);
    let out_dir = make_session_dir(&base_dir, args.label).map_err(|e| e.to_string())?;
    let runtime = new_runtime()?;
    let llm = build_llm_client(&infra)?;
    runtime.block_on(warn_missing_models(llm.as_ref(), &config));
    let persistence = FilePersistence::new(out_dir.clone()).map_err(|e| e.to_string())?;
    let controller =
        SessionController::new(config.clone(), template.clone(), llm, Box::new(persistence));
    let store = Arc::new(CachedModelStore::new(&paths));
    let launcher = file_launcher(store, &config, &template, audio_path)?;
    let session = HeadlessSession::new(
        controller,
        out_dir.clone(),
        "transcribing",
        true,
        None,
        StatusWriter::stderr(),
    );
    eprint_lines(&banner_lines(
        "transcribing",
        &out_dir,
        session.status().pid,
        false,
    ));
    Ok(run_session(
        &runtime, &out_dir, &config, session, launcher, true,
    ))
}

/// The file transcription worker for `audio_path`, ready to start on its own thread.
pub fn file_launcher(
    store: Arc<CachedModelStore>,
    config: &AppConfig,
    template: &SessionTemplate,
    audio_path: &Path,
) -> Result<Launcher, String> {
    let transcriber: Box<dyn Transcriber> =
        Box::new(SubprocessWhisperTranscriber::new().map_err(|e| e.to_string())?);
    let model_name = config
        .transcription
        .model_for_locale(&template.metadata.locale)
        .to_string();
    let (language, file_settings, file_hints) = (
        language_of(template),
        settings(config),
        hints(config, template),
    );
    let audio_path: PathBuf = audio_path.to_path_buf();
    Ok(Box::new(move |poster, cancel| {
        let resolver = resolver_posting(store, &poster, &model_name);
        let job = FileJob {
            audio_path: &audio_path,
            model_name: &model_name,
            language: &language,
            settings: file_settings,
            recognition_hints: file_hints,
        };
        run_file_transcription(
            &poster,
            &cancel,
            job,
            &resolver as &dyn ModelResolver,
            transcriber,
        );
    }))
}

/// Warn up front about models the provider lacks. Headless has no screen to show it later.
async fn warn_missing_models(llm: &dyn crate::ports::llm::LlmClient, config: &AppConfig) {
    if let Err(error) = llm.check_connectivity().await {
        eprintln!("Warning: LLM provider not reachable ({error}). Digests will fail.");
        eprintln!("Transcript-only mode: audio capture will still work.");
        return;
    }
    let mut models = vec![config.digest.model.clone()];
    if config.interactive.model != config.digest.model {
        models.push(config.interactive.model.clone());
    }
    let missing = llm.check_models(&models).await;
    for model in models.iter().filter(|model| missing.contains(model)) {
        eprintln!("⚠ model not available: {model} (cycles using it will fail)");
    }
}

/// Run the session, print the summary, and return 1 when it ended in an error.
///
/// With `drop_if_empty`, a failed run whose folder holds only the status file
/// removes the folder: a script retrying a bad file would otherwise leave one
/// empty folder per try.
fn run_session(
    runtime: &tokio::runtime::Runtime,
    out_dir: &Path,
    config: &AppConfig,
    mut session: HeadlessSession,
    launcher: Launcher,
    drop_if_empty: bool,
) -> i32 {
    setup_file_logging(out_dir, config.output.save_debug_log);
    runtime.block_on(session.run(launcher));
    close_file_logging();
    eprint_lines(&summary_lines(out_dir, session.status()));
    let failed = session.status().state == st::ERROR;
    let dropped = failed
        && drop_if_empty
        && holds_only_status(out_dir)
        && std::fs::remove_file(out_dir.join(STATUS_FILE)).is_ok()
        && std::fs::remove_dir(out_dir).is_ok();
    if dropped {
        eprintln!("Removed {}: the run saved nothing.", out_dir.display());
    } else {
        println!("{}", out_dir.display());
    }
    i32::from(failed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapters::paths::Paths;

    fn templates() -> (tempfile::TempDir, YamlTemplates) {
        let home = tempfile::tempdir().unwrap();
        let templates = YamlTemplates::new(&Paths::under(home.path()));
        (home, templates)
    }

    #[test]
    fn a_language_finds_its_default_template() {
        let (_home, templates) = templates();
        assert_eq!(
            resolve_language(&templates, "en-US")
                .unwrap()
                .metadata
                .locale,
            "en"
        );
        assert_eq!(
            resolve_language(&templates, "zh-tw")
                .unwrap()
                .metadata
                .locale,
            "zh-TW"
        );
        let error = resolve_language(&templates, "fr").unwrap_err();
        assert!(
            error.starts_with("Unsupported language 'fr'. Available: "),
            "{error}"
        );
    }

    #[test]
    fn an_unknown_template_lists_the_keys() {
        let (_home, templates) = templates();
        let error = load_template_key(&templates, "nope").unwrap_err();
        assert!(
            error.starts_with("Unknown template 'nope'. Available: daily_standup_en, "),
            "{error}"
        );
    }
}
