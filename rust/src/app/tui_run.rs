//! `record` and `transcribe` with the TUI: build the session, then run its screen.

use std::path::Path;
use std::sync::Arc;

use crate::adapters::assets::store::CachedModelStore;
use crate::adapters::file_persistence::FilePersistence;
use crate::adapters::paths::Paths;
use crate::adapters::session_controller::SessionController;
use crate::adapters::templates::YamlTemplates;
use crate::app::container::{
    Loaded, build_llm_client, check_transcription_provider, load_config, make_session_dir,
    resolve_base_dir,
};
use crate::app::headless_run::{
    HeadlessArgs, file_launcher, load_template_key, new_runtime, resolve_language,
};
use crate::app::logging::{close_file_logging, setup_file_logging};
use crate::domain::config::AppConfig;
use crate::domain::template::SessionTemplate;
use crate::ports::llm::LlmClient;
use crate::tui::host::run_screen;
use crate::tui::screens::session::Session;
use crate::tui::screens::transcribe::TranscribeScreen;
use crate::tui::system::saved_theme;

/// Python's `preflight_llm`: (missing digest models, missing interactive models).
/// An unreachable provider only warns; the digests then fail in the session.
pub async fn preflight_llm(llm: &dyn LlmClient, config: &AppConfig) -> (Vec<String>, Vec<String>) {
    if let Err(error) = llm.check_connectivity().await {
        eprintln!("Warning: LLM provider not reachable ({error}). Digests will fail.");
        eprintln!("Transcript-only mode: audio capture will still work.");
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

pub enum TemplatePick {
    Chosen(Box<SessionTemplate>),
    Cancelled,
    /// `n` or `a`: the builder, which this version does not have yet.
    NeedsBuilder,
}

/// The template picker, as Python's `pick_template`.
pub fn pick_template(paths: &Paths, show_builtins: bool) -> Result<TemplatePick, String> {
    use crate::tui::screens::template_picker::{TemplateChoice, TemplatePicker};
    let runtime = new_runtime()?;
    crate::tui::terminal::clear_normal_screen();
    let picker = TemplatePicker::new(
        YamlTemplates::new(paths),
        show_builtins,
        paths.builtin_templates_noticed(),
    );
    let result = run_screen(Box::new(picker), &saved_theme(), runtime.handle().clone())
        .map_err(|e| e.to_string())?;
    runtime.shutdown_background();
    let choice = result.and_then(|payload| payload.downcast::<TemplateChoice>().ok());
    Ok(match choice.map(|choice| *choice) {
        None => TemplatePick::Cancelled,
        Some(TemplateChoice::Key(key)) => TemplatePick::Chosen(Box::new(load_template_key(
            &YamlTemplates::new(paths),
            &key,
        )?)),
        Some(TemplateChoice::Create | TemplateChoice::EditWithAi(_)) => TemplatePick::NeedsBuilder,
    })
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
    let Loaded {
        config,
        infra,
        paths,
    } = load_config(args.config_path, args.output_dir)?;
    let templates = YamlTemplates::new(&paths);
    let template = match template_from_flags(&templates, args) {
        Some(template) => template?,
        None => match pick_template(&paths, infra.show_builtin_templates)? {
            TemplatePick::Chosen(template) => *template,
            TemplatePick::Cancelled => return Ok(0),
            TemplatePick::NeedsBuilder => {
                eprintln!(
                    "lazy-take-notes (Rust): the template builder is not usable yet; pass --template or --language"
                );
                return Ok(2);
            }
        },
    };
    check_transcription_provider(&infra)?;
    let base_dir = resolve_base_dir(&paths, args.output_dir, &config);
    let out_dir = make_session_dir(&base_dir, args.label).map_err(|e| e.to_string())?;
    let runtime = new_runtime()?;
    let llm = build_llm_client(&infra)?;
    let (missing_digest, missing_interactive) =
        runtime.block_on(preflight_llm(llm.as_ref(), &config));
    let persistence = FilePersistence::new(out_dir.clone()).map_err(|e| e.to_string())?;
    let controller =
        SessionController::new(config.clone(), template.clone(), llm, Box::new(persistence));
    let store = Arc::new(CachedModelStore::new(&paths));
    let launcher = file_launcher(store, &config, &template, audio_path)?;
    setup_file_logging(&out_dir, config.output.save_debug_log);
    let session = Session::new(
        controller,
        out_dir,
        args.label.unwrap_or(""),
        missing_digest,
        missing_interactive,
    );
    let screen = TranscribeScreen::new(session, config.transcription.silence_threshold, launcher);
    let outcome = run_screen(Box::new(screen), &saved_theme(), runtime.handle().clone());
    close_file_logging();
    // The file worker and any LLM call end with the process; do not wait for them.
    runtime.shutdown_background();
    outcome.map(|_| 0).map_err(|e| e.to_string())
}
