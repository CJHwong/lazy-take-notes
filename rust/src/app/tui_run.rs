//! `record` and `transcribe` with the TUI: build the session, then run its screen.

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::Duration;

use crate::adapters::assets::store::CachedModelStore;
use crate::adapters::audio::session_folder::SessionFolder;
use crate::adapters::file_persistence::FilePersistence;
use crate::adapters::nemo_diarizer::NemoSpeechDiarizer;
use crate::adapters::paths::Paths;
use crate::adapters::session_controller::SessionController;
use crate::adapters::templates::YamlTemplates;
use crate::adapters::whisper::transcriber::SubprocessWhisperTranscriber;
use crate::app::container::build_mixed_source;
use crate::app::container::{
    Loaded, build_llm_client, check_transcription_provider, load_config, make_session_dir,
    resolve_base_dir,
};
use crate::app::events::{Event, WorkerState};
use crate::app::headless::Launcher;
use crate::app::headless_run::{
    HeadlessArgs, file_launcher, load_template_key, new_runtime, resolve_language,
};
use crate::app::headless_run::{hints, language_of, resolver_posting, settings};
use crate::app::keep_awake::keep_awake;
use crate::app::logging::{close_file_logging, setup_file_logging};
use crate::app::preflight::PreflightPlan;
use crate::app::workers::audio::{AudioWorkerConfig, LoopTiming, run_audio_worker};
use crate::app::workers::diarization::{DiarizationTiming, run_diarization_worker};
use crate::domain::config::AppConfig;
use crate::domain::template::SessionTemplate;
use crate::ports::llm::LlmClient;
use crate::ports::model_resolver::ModelResolver;
use crate::ports::transcriber::Transcriber;
use crate::tui::host::run_screen;
use crate::tui::screens::preflight::{PreflightOutcome, PreflightScreen, Reload};
use crate::tui::screens::record::{RecordScreen, RecordWorkers};
use crate::tui::screens::session::Session;
use crate::tui::screens::transcribe::TranscribeScreen;
use crate::tui::system::saved_theme;

/// Python's `preflight_llm`: (missing digest models, missing interactive models).
/// An unreachable provider only warns, through `warn`; the digests then fail in the session.
pub async fn preflight_llm(
    llm: &dyn LlmClient,
    config: &AppConfig,
    warn: impl Fn(&str),
) -> (Vec<String>, Vec<String>) {
    if let Err(error) = llm.check_connectivity().await {
        warn(&format!(
            "Warning: LLM provider not reachable ({error}). Digests will fail."
        ));
        warn("Transcript-only mode: audio capture will still work.");
        return (Vec::new(), Vec::new());
    }
    let (digest, interactive) = (&config.digest.model, &config.interactive.model);
    let mut models = vec![digest.clone()];
    if interactive != digest {
        models.push(interactive.clone());
    }
    let missing = llm.check_models(&models).await;
    let pick = |model: &String| match missing.contains(model) {
        true => vec![model.clone()],
        false => Vec::new(),
    };
    (pick(digest), pick(interactive))
}

/// --language, then --template. None means the picker has to choose.
fn template_from_flags(
    templates: &YamlTemplates,
    args: &HeadlessArgs<'_>,
) -> Option<Result<SessionTemplate, String>> {
    if let Some(language) = args.language {
        return Some(resolve_language(templates, language));
    }
    args.template_name
        .map(|name| load_template_key(templates, name))
}

/// --language, then --template, then the picker. `Err(code)` ends the command:
/// 0 when the user cancels the picker.
fn choose_template(
    paths: &Paths,
    show_builtins: bool,
    args: &HeadlessArgs<'_>,
) -> Result<Result<SessionTemplate, i32>, String> {
    if let Some(template) = template_from_flags(&YamlTemplates::new(paths), args) {
        return Ok(Ok(template?));
    }
    Ok(
        match pick_template(paths, show_builtins, args.config_path)? {
            Some(template) => Ok(template),
            None => Err(0),
        },
    )
}

/// The template picker, as Python's `pick_template`. `n` and `a` run the
/// builder, then show the picker again. None when the user cancels.
pub fn pick_template(
    paths: &Paths,
    show_builtins: bool,
    config_path: Option<&Path>,
) -> Result<Option<SessionTemplate>, String> {
    use crate::tui::screens::template_picker::{TemplateChoice, TemplatePicker};
    crate::tui::terminal::clear_normal_screen();
    loop {
        let runtime = new_runtime()?;
        let picker = TemplatePicker::new(
            YamlTemplates::new(paths),
            show_builtins,
            paths.builtin_templates_noticed(),
        );
        let result = run_screen(Box::new(picker), &saved_theme(), runtime.handle().clone())
            .map_err(|e| e.to_string())?;
        runtime.shutdown_background();
        let choice = result.and_then(|payload| payload.downcast::<TemplateChoice>().ok());
        match choice.map(|choice| *choice) {
            None => return Ok(None),
            Some(TemplateChoice::Key(key)) => {
                return load_template_key(&YamlTemplates::new(paths), &key).map(Some);
            }
            Some(TemplateChoice::Create) => run_builder(paths, config_path, None)?,
            Some(TemplateChoice::EditWithAi(key)) => {
                let base = load_template_key(&YamlTemplates::new(paths), &key)?;
                run_builder(paths, config_path, Some((base, key)))?;
            }
        }
    }
}

/// The template builder, as its own app between two pickers.
fn run_builder(
    paths: &Paths,
    config_path: Option<&Path>,
    base: Option<(SessionTemplate, String)>,
) -> Result<(), String> {
    let runtime = new_runtime()?;
    let screen = builder_screen(paths, config_path, base);
    run_screen(Box::new(screen), &saved_theme(), runtime.handle().clone())
        .map_err(|e| e.to_string())?;
    // A call still running ends with the builder.
    runtime.shutdown_background();
    Ok(())
}

/// The template builder screen. It reads config.yaml on the first send, as
/// Python does, so an edit made in Settings meanwhile counts.
pub fn builder_screen(
    paths: &Paths,
    config_path: Option<&Path>,
    base: Option<(SessionTemplate, String)>,
) -> crate::tui::screens::template_builder::TemplateBuilderScreen {
    use crate::tui::screens::template_builder::{BuilderSetup, TemplateBuilderScreen};
    let config_path = config_path.map(Path::to_path_buf);
    let example_paths = paths.clone();
    let setup = Box::new(move || {
        let loaded = load_config(config_path.as_deref(), None)?;
        let example = YamlTemplates::new(&example_paths)
            .load_template("default_en")
            .map_err(|e| e.to_string())?;
        Ok(BuilderSetup {
            llm: build_llm_client(&loaded.infra)?,
            model: loaded.config.digest.model.clone(),
            provider: loaded.infra.llm_provider.clone(),
            example,
        })
    });
    TemplateBuilderScreen::new(setup, YamlTemplates::new(paths), base)
}

/// The file picker, from the current folder. None when the user cancels.
pub fn pick_file() -> Result<Option<std::path::PathBuf>, String> {
    let start = std::env::current_dir().map_err(|e| e.to_string())?;
    let runtime = new_runtime()?;
    crate::tui::terminal::clear_normal_screen();
    let picker = crate::tui::screens::file_picker::FilePicker::new(&start);
    let result = run_screen(Box::new(picker), &saved_theme(), runtime.handle().clone())
        .map_err(|e| e.to_string())?;
    runtime.shutdown_background();
    Ok(crate::tui::screens::session_picker::picked_dir(result))
}

/// The transcription TUI for one file. Returns the exit code.
pub fn transcribe(args: &HeadlessArgs<'_>, audio_path: &Path) -> Result<i32, String> {
    transcribe_source(args, TranscribeSource::Audio(audio_path.to_path_buf())).map(|_| 0)
}

/// What a transcription session reads: an audio file, or a plugin's subtitle
/// segments replayed without whisper.
pub enum TranscribeSource {
    Audio(std::path::PathBuf),
    Subtitles(Vec<crate::domain::transcript::TranscriptSegment>),
}

/// Run a transcription session in the TUI. Returns the session folder, or None
/// when the template picker was cancelled before a session existed.
pub fn transcribe_source(
    args: &HeadlessArgs<'_>,
    source: TranscribeSource,
) -> Result<Option<std::path::PathBuf>, String> {
    let Loaded {
        config,
        infra,
        paths,
    } = load_config(args.config_path, args.output_dir)?;
    let template = match choose_template(&paths, infra.show_builtin_templates, args)? {
        Ok(template) => template,
        Err(_) => return Ok(None),
    };
    let loaded = Loaded {
        config,
        infra,
        paths,
    };
    let start = TranscribeStart::new(&loaded, &template, args.output_dir, args.label)?;
    let session_dir = start.out_dir.clone();
    let runtime = new_runtime()?;
    let missing = runtime.block_on(preflight_llm(start.llm.as_ref(), &start.config, |line| {
        eprintln!("{line}")
    }));
    let screen = start.screen(&source, missing)?;
    let outcome = run_screen(Box::new(screen), &saved_theme(), runtime.handle().clone());
    close_file_logging();
    // The file worker and any LLM call end with the process; do not wait for them.
    runtime.shutdown_background();
    outcome
        .map(|_| Some(session_dir))
        .map_err(|e| e.to_string())
}

/// A transcription session up to its LLM preflight, as Python's
/// `build_transcribe_session`. The preflight runs between `new` and `screen`.
pub struct TranscribeStart {
    pub llm: Arc<dyn LlmClient>,
    pub config: AppConfig,
    template: SessionTemplate,
    paths: Paths,
    pub out_dir: std::path::PathBuf,
    label: String,
}

impl TranscribeStart {
    pub fn new(
        loaded: &Loaded,
        template: &SessionTemplate,
        output_dir: Option<&str>,
        label: Option<&str>,
    ) -> Result<Self, String> {
        check_transcription_provider(&loaded.infra)?;
        let base_dir = resolve_base_dir(&loaded.paths, output_dir, &loaded.config);
        let out_dir = make_session_dir(&base_dir, label).map_err(|e| e.to_string())?;
        Ok(Self {
            llm: build_llm_client(&loaded.infra)?,
            config: loaded.config.clone(),
            template: template.clone(),
            paths: loaded.paths.clone(),
            out_dir,
            label: label.unwrap_or("").to_string(),
        })
    }

    /// The screen, given the missing (digest, interactive) models. Opens `debug.log`.
    pub fn screen(
        self,
        source: &TranscribeSource,
        (missing_digest, missing_interactive): (Vec<String>, Vec<String>),
    ) -> Result<TranscribeScreen, String> {
        let config = self.config;
        let persistence = FilePersistence::new(self.out_dir.clone()).map_err(|e| e.to_string())?;
        let controller = SessionController::new(
            config.clone(),
            self.template.clone(),
            self.llm,
            Box::new(persistence),
        );
        let launcher: Launcher = match source {
            TranscribeSource::Audio(audio_path) => {
                let store = Arc::new(CachedModelStore::new(&self.paths));
                file_launcher(store, &config, &self.template, audio_path)?
            }
            TranscribeSource::Subtitles(segments) => {
                let segments = segments.clone();
                Box::new(move |poster, cancel| {
                    crate::app::workers::file_transcription::run_subtitle_replay(
                        &poster, &cancel, segments,
                    );
                })
            }
        };
        setup_file_logging(&self.out_dir, config.output.save_debug_log);
        let session = Session::new(
            controller,
            self.out_dir,
            &self.label,
            missing_digest,
            missing_interactive,
        );
        Ok(TranscribeScreen::new(
            session,
            config.transcription.silence_threshold,
            launcher,
        ))
    }
}

/// The preflight plan, as Python's `build_preflight_plan`. The checks and the
/// session then share its LLM client and model store.
fn build_plan(
    loaded: &Loaded,
    template: &SessionTemplate,
    output_dir: Option<&str>,
    runtime: tokio::runtime::Handle,
    mute_mic: bool,
) -> Result<PreflightPlan, String> {
    Ok(PreflightPlan {
        config: loaded.config.clone(),
        template: template.clone(),
        base_dir: resolve_base_dir(&loaded.paths, output_dir, &loaded.config),
        llm_client: build_llm_client(&loaded.infra)?,
        llm_provider: loaded.infra.llm_provider.clone(),
        asset_store: Arc::new(CachedModelStore::new(&loaded.paths)),
        runtime,
        mic_muted: mute_mic,
        cancel: Arc::default(),
    })
}

/// The live recording TUI: template, preflight screen, then the record screen.
pub fn record(args: &HeadlessArgs<'_>, mute_mic: bool) -> Result<i32, String> {
    let loaded = load_config(args.config_path, args.output_dir)?;
    let template = match choose_template(&loaded.paths, loaded.infra.show_builtin_templates, args)?
    {
        Ok(template) => template,
        Err(code) => return Ok(code),
    };
    let runtime = new_runtime()?;
    let screen = preflight_screen(
        &loaded,
        &template,
        args.config_path,
        args.output_dir,
        runtime.handle().clone(),
        mute_mic,
    )?;
    crate::tui::terminal::clear_normal_screen();
    let result = run_screen(Box::new(screen), &saved_theme(), runtime.handle().clone())
        .map_err(|e| e.to_string())?;
    let Some(outcome) = result.and_then(|payload| payload.downcast::<PreflightOutcome>().ok())
    else {
        runtime.shutdown_background();
        return Ok(0);
    };
    let screen = record_screen(*outcome, &loaded.paths, args.label, mute_mic)?;
    let outcome = {
        let _awake = keep_awake();
        run_screen(Box::new(screen), &saved_theme(), runtime.handle().clone())
    };
    close_file_logging();
    // The audio worker flushed before the screen closed; the rest ends with the process.
    runtime.shutdown_background();
    outcome.map(|_| 0).map_err(|e| e.to_string())
}

/// The preflight screen for `template`, as Python's `PreflightScreen(build_preflight_plan(...),
/// reload=plan_reloader(...))`.
pub fn preflight_screen(
    loaded: &Loaded,
    template: &SessionTemplate,
    config_path: Option<&Path>,
    output_dir: Option<&str>,
    runtime: tokio::runtime::Handle,
    mute_mic: bool,
) -> Result<PreflightScreen, String> {
    check_transcription_provider(&loaded.infra)?;
    let plan = build_plan(loaded, template, output_dir, runtime.clone(), mute_mic)?;
    let transcriber: Box<dyn Transcriber> =
        Box::new(SubprocessWhisperTranscriber::new().map_err(|e| e.to_string())?);
    let reload: Reload = {
        let config_path = config_path.map(Path::to_path_buf);
        let output_dir = output_dir.map(str::to_string);
        let template = template.clone();
        Box::new(move || {
            let loaded = load_config(config_path.as_deref(), output_dir.as_deref())?;
            build_plan(
                &loaded,
                &template,
                output_dir.as_deref(),
                runtime.clone(),
                mute_mic,
            )
        })
    };
    Ok(PreflightScreen::new(plan, transcriber, Some(reload)))
}

/// The record screen after a passed preflight, as Python's `build_record_session`.
/// Opens `debug.log`; the caller closes it when the screen ends.
pub fn record_screen(
    outcome: PreflightOutcome,
    paths: &Paths,
    label: Option<&str>,
    mute_mic: bool,
) -> Result<RecordScreen, String> {
    let PreflightOutcome {
        report,
        plan,
        transcriber,
    } = outcome;
    let config = plan.config.clone();
    // An overridden AI check starts the session in transcript-only mode.
    let (missing_digest, missing_interactive) = match report.llm_ok() {
        true => (vec![], vec![]),
        false => (
            vec![config.digest.model.clone()],
            vec![config.interactive.model.clone()],
        ),
    };
    let out_dir = make_session_dir(&plan.base_dir, label).map_err(|e| e.to_string())?;
    let persistence = FilePersistence::new(out_dir.clone()).map_err(|e| e.to_string())?;
    let controller = SessionController::new(
        config.clone(),
        plan.template.clone(),
        plan.llm_client.clone(),
        Box::new(persistence),
    );
    let workers = record_workers(&plan, paths, &out_dir, transcriber, mute_mic);
    setup_file_logging(&out_dir, config.output.save_debug_log);
    let session = Session::new(
        controller,
        out_dir,
        label.unwrap_or(""),
        missing_digest,
        missing_interactive,
    );
    Ok(RecordScreen::new(
        session,
        config.transcription.silence_threshold,
        workers,
    ))
}

/// The audio and diarization workers for a session in `out_dir`, as Python's
/// `RecordApp._start_audio_worker` and `_start_diarization_worker`.
pub fn record_workers(
    plan: &PreflightPlan,
    paths: &Paths,
    out_dir: &Path,
    transcriber: Box<dyn Transcriber>,
    mute_mic: bool,
) -> RecordWorkers {
    let config = &plan.config;
    let template = &plan.template;
    let source = build_mixed_source(config);
    if mute_mic {
        crate::ports::audio_source::AudioSource::set_mic_muted(source.as_ref(), true);
    }
    let folder = Arc::new(SessionFolder::new(out_dir.to_path_buf()));
    let paused = Arc::new(AtomicBool::new(false));
    let model_name = config
        .transcription
        .model_for_locale(&template.metadata.locale)
        .to_string();
    let worker_config = AudioWorkerConfig {
        model_path: String::new(),
        language: language_of(template),
        settings: settings(config),
        recognition_hints: hints(config, template),
        folder: Some(folder.clone()),
        save_audio: config.output.save_audio,
        timing: LoopTiming::default(),
    };
    let (store, worker_source, worker_paused) =
        (plan.asset_store.clone(), source.clone(), paused.clone());
    let audio: Launcher = Box::new(move |poster, cancel| {
        // Resolve the model on the worker thread, so a download does not block the screen.
        let resolved = resolver_posting(store, &poster, &model_name).resolve(&model_name);
        let model_path = match resolved {
            Ok(path) => path,
            Err(e) => {
                poster.post(Event::status_error(WorkerState::Error, e.to_string()));
                return;
            }
        };
        let config = AudioWorkerConfig {
            model_path,
            ..worker_config
        };
        run_audio_worker(
            &poster,
            &cancel,
            &worker_paused,
            config,
            transcriber,
            worker_source,
        );
    });
    let diarization_config = &config.diarization;
    let mut startup_warning = None;
    let mut diarization: Option<crate::tui::screens::record::DiarizationLauncher> = None;
    if diarization_config.enabled && !config.output.save_audio {
        startup_warning = Some("Diarization needs save_audio: true in config.yaml".to_string());
    } else if diarization_config.enabled {
        let diarizer = NemoSpeechDiarizer::new(
            &diarization_config.binary,
            &diarization_config.model,
            &diarization_config.device,
            Some(plan.asset_store.clone()),
        );
        let timing = DiarizationTiming {
            interval: Duration::from_secs_f64(diarization_config.interval_s.max(0.0)),
            ..DiarizationTiming::default()
        };
        let folder = folder.clone();
        diarization = Some(Box::new(move |poster, cancel, finish| {
            let audio_path = || folder.path().join("recording.wav");
            run_diarization_worker(&poster, &cancel, &finish, &audio_path, &diarizer, &timing);
        }));
    }
    RecordWorkers {
        audio,
        diarization,
        paused,
        source,
        folder,
        mic_muted: mute_mic,
        consent_noticed: paths.consent_noticed(),
        startup_warning,
    }
}
