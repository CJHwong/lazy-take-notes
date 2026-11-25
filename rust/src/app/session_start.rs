//! A session up to its first screen: the LLM preflight, the preflight plan,
//! and the session and workers a front end then runs.

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
use crate::app::container::build_mixed_source;
use crate::app::container::{
    Loaded, build_llm_client, check_transcription_provider, make_session_dir, resolve_base_dir,
};
use crate::app::events::{Event, WorkerState};
use crate::app::headless::Launcher;
use crate::app::headless_run::{HeadlessArgs, file_launcher, load_template_key, resolve_language};
use crate::app::headless_run::{hints, language_of, resolver_posting, settings};
use crate::app::logging::setup_file_logging;
use crate::app::preflight::{PreflightOutcome, PreflightPlan};
use crate::app::session::SessionRuntime;
use crate::app::session::record::{DiarizationLauncher, RecordWorkers};
use crate::app::workers::audio::{AudioWorkerConfig, LoopTiming, run_audio_worker};
use crate::app::workers::diarization::{DiarizationTiming, run_diarization_worker};
use crate::domain::config::AppConfig;
use crate::domain::template::SessionTemplate;
use crate::ports::llm::LlmClient;
use crate::ports::model_resolver::ModelResolver;
use crate::ports::transcriber::Transcriber;

/// A file session, ready for its screen.
pub struct FileStart {
    pub session: SessionRuntime,
    pub launcher: Launcher,
    pub silence_threshold: f64,
}

/// A recording, ready for its screen.
pub struct RecordStart {
    pub session: SessionRuntime,
    pub workers: RecordWorkers,
    pub silence_threshold: f64,
}

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
pub fn template_from_flags(
    templates: &YamlTemplates,
    args: &HeadlessArgs<'_>,
) -> Option<Result<SessionTemplate, String>> {
    if let Some(language) = args.language {
        return Some(resolve_language(templates, language));
    }
    args.template_name
        .map(|name| load_template_key(templates, name))
}

/// What a transcription session reads: an audio file, or a plugin's subtitle
/// segments replayed without whisper.
pub enum TranscribeSource {
    Audio(std::path::PathBuf),
    Subtitles(Vec<crate::domain::transcript::TranscriptSegment>),
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

    /// The session, given the missing (digest, interactive) models. Opens `debug.log`.
    pub fn start(
        self,
        source: &TranscribeSource,
        (missing_digest, missing_interactive): (Vec<String>, Vec<String>),
    ) -> Result<FileStart, String> {
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
        Ok(FileStart {
            session: SessionRuntime::new(
                controller,
                self.out_dir,
                &self.label,
                missing_digest,
                missing_interactive,
            ),
            launcher,
            silence_threshold: config.transcription.silence_threshold,
        })
    }
}

/// The preflight plan, as Python's `build_preflight_plan`. The checks and the
/// session then share its LLM client and model store.
pub fn build_plan(
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

/// The recording after a passed preflight, as Python's `build_record_session`.
/// Opens `debug.log`; the caller closes it when the session ends.
pub fn record_start(
    outcome: PreflightOutcome,
    paths: &Paths,
    label: Option<&str>,
    mute_mic: bool,
) -> Result<RecordStart, String> {
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
    Ok(RecordStart {
        session: SessionRuntime::new(
            controller,
            out_dir,
            label.unwrap_or(""),
            missing_digest,
            missing_interactive,
        ),
        workers,
        silence_threshold: config.transcription.silence_threshold,
    })
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
    let mut diarization: Option<DiarizationLauncher> = None;
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
        diarization = Some(Box::new(move |poster, cancel, finish, now| {
            let audio_path = || folder.path().join("recording.wav");
            run_diarization_worker(
                &poster,
                &cancel,
                &finish,
                &now,
                &audio_path,
                &diarizer,
                &timing,
            );
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
