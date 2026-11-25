//! `record` and `transcribe` with the TUI: build the session, then run its screen.

use std::path::Path;

use crate::adapters::paths::Paths;
use crate::adapters::templates::YamlTemplates;
use crate::adapters::whisper::transcriber::SubprocessWhisperTranscriber;
use crate::app::container::{Loaded, build_llm_client, check_transcription_provider, load_config};
use crate::app::headless_run::{HeadlessArgs, load_template_key, new_runtime};
use crate::app::keep_awake::keep_awake;
use crate::app::logging::close_file_logging;
use crate::app::preflight::{PreflightOutcome, Reload};
use crate::app::session_start::{
    FileStart, RecordStart, build_plan, preflight_llm, record_start, template_from_flags,
};
pub use crate::app::session_start::{TranscribeSource, TranscribeStart};
use crate::domain::template::SessionTemplate;
use crate::ports::transcriber::Transcriber;
use crate::tui::host::run_screen;
use crate::tui::screens::preflight::PreflightScreen;
use crate::tui::screens::record::RecordScreen;
use crate::tui::screens::session::{Session, SessionView};
use crate::tui::screens::transcribe::TranscribeScreen;
use crate::tui::system::saved_theme;

/// The screen for a file session.
pub fn transcribe_screen(start: FileStart) -> TranscribeScreen {
    let session = Session {
        runtime: start.session,
        view: SessionView::default(),
    };
    TranscribeScreen::new(session, start.silence_threshold, start.launcher)
}

/// The record screen after a passed preflight. Opens `debug.log`; the caller
/// closes it when the screen ends.
pub fn record_screen(
    outcome: PreflightOutcome,
    paths: &Paths,
    label: Option<&str>,
    mute_mic: bool,
) -> Result<RecordScreen, String> {
    let RecordStart {
        session,
        workers,
        silence_threshold,
    } = record_start(outcome, paths, label, mute_mic)?;
    let session = Session {
        runtime: session,
        view: SessionView::default(),
    };
    Ok(RecordScreen::new(session, silence_threshold, workers))
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
    let screen = transcribe_screen(start.start(&source, missing)?);
    let outcome = run_screen(Box::new(screen), &saved_theme(), runtime.handle().clone());
    close_file_logging();
    // The file worker and any LLM call end with the process; do not wait for them.
    runtime.shutdown_background();
    outcome
        .map(|_| Some(session_dir))
        .map_err(|e| e.to_string())
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
