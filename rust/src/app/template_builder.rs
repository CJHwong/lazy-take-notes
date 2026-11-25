//! The template builder without any screen: the chat, the calls to the AI model,
//! the accepted template and its save (the logic of Python's `TemplateBuilderApp`).

use std::path::{Path, PathBuf};
use std::sync::{Arc, LazyLock};
use std::time::Duration;

use tracing::error;

use crate::adapters::paths::Paths;
use crate::adapters::templates::YamlTemplates;
use crate::app::container::{build_llm_client, load_config};
use crate::app::tasks::{Payload, Tasks};
use crate::domain::chat::ChatMessage;
use crate::domain::template::SessionTemplate;
use crate::ports::llm::{LlmClient, LlmError};
use crate::usecases::template_builder::{BuilderResult, TemplateBuilder};
use crate::usecases::template_diff::diff_templates;

const MAX_AUTO_FIX: usize = 2;
/// A full template takes a slow model a few minutes; past this the call is stuck.
pub const LLM_TIMEOUT: Duration = Duration::from_secs(300);

const WELCOME: &str = "Describe the template you want to create. For example:
- \"1-on-1 template, English, track action items and career goals\"
- \"Lecture notes in Chinese, focus on key concepts and exam topics\"

I'll guide you if I need more details, or generate right away if your description is complete enough.
";

const EDIT_WELCOME: &str = "Describe the change you want. For example:
- \"Add a quick action that lists risks\"
- \"Make the summary shorter and write it in Traditional Chinese\"

I'll restate the change and wait for your yes before I edit the template. The preview shows the current version.
";

/// The preview before any template exists.
pub const NO_TEMPLATE: &str = "*No template yet. Describe what you want.*";

/// What the builder needs from the config, built on the first send as in Python.
pub struct BuilderSetup {
    pub llm: Arc<dyn LlmClient>,
    pub model: String,
    pub provider: String,
    pub example: SessionTemplate,
}

pub type Setup = Box<dyn Fn() -> Result<BuilderSetup, String>>;

/// The setup from config.yaml. It reads the file on the first send, as Python
/// does, so an edit made in Settings meanwhile counts.
pub fn config_setup(paths: &Paths, config_path: Option<&Path>) -> Setup {
    let config_path = config_path.map(Path::to_path_buf);
    let example_paths = paths.clone();
    Box::new(move || {
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
    })
}

/// `_slugify`: a safe file name.
pub fn slugify(text: &str) -> String {
    static NOT_WORD: LazyLock<regex::Regex> =
        LazyLock::new(|| regex::Regex::new(r"[^\w]").expect("a valid pattern"));
    static UNDERSCORES: LazyLock<regex::Regex> =
        LazyLock::new(|| regex::Regex::new("_+").expect("a valid pattern"));
    let slug = NOT_WORD
        .replace_all(&text.trim().to_lowercase(), "_")
        .into_owned();
    let slug = UNDERSCORES.replace_all(&slug, "_");
    let slug = slug.trim_matches('_');
    match slug.is_empty() {
        true => "custom_template".into(),
        false => slug.into(),
    }
}

/// The preview pane's Markdown, as Python's `_update_preview`.
pub fn preview_markdown(template: &SessionTemplate) -> String {
    let meta = &template.metadata;
    let name = if meta.name.is_empty() {
        "Untitled"
    } else {
        &meta.name
    };
    let locale = if meta.locale.is_empty() {
        "not set"
    } else {
        &meta.locale
    };
    let mut lines = vec![
        format!("## {name}"),
        String::new(),
        match meta.description.is_empty() {
            true => String::new(),
            false => format!("> {}", meta.description),
        },
        String::new(),
        format!("**Locale:** `{locale}`"),
    ];
    if !template.quick_actions.is_empty() {
        lines.extend([String::new(), "### Quick Actions".into()]);
        for (index, action) in template.quick_actions.iter().enumerate() {
            let description = match action.description.is_empty() {
                true => String::new(),
                false => format!(" — {}", action.description),
            };
            lines.push(format!(
                "- **`{}`** {}{description}",
                index + 1,
                action.label
            ));
        }
    }
    if !template.recognition_hints.is_empty() {
        lines.extend([
            String::new(),
            format!(
                "**Recognition hints:** {}",
                template.recognition_hints.join(", ")
            ),
        ]);
    }
    for (title, body) in [
        ("System Prompt", &template.system_prompt),
        ("Digest Template", &template.digest_user_template),
    ] {
        lines.extend([
            String::new(),
            "---".into(),
            String::new(),
            format!("### {title}"),
            String::new(),
            body.clone(),
        ]);
    }
    lines.join("\n")
}

/// The answer to an LLM error, as Python's `_error_text`.
pub fn error_text(error: &LlmError, model: &str, provider: &str) -> String {
    let not_found = error.kind == "NotFoundError" || error.message.ends_with("(status code: 404)");
    if not_found {
        let fix = match provider {
            "ollama" => format!("Run: ollama pull {model}"),
            _ => "Pick an available model in Settings > Models.".into(),
        };
        return format!("The AI model {model} is not available. {fix}");
    }
    format!("Error: {}: {}", error.kind, error.message)
}

enum Turn {
    Done(Box<BuilderResult>),
    Failed(LlmError),
    TimedOut,
}

/// One send: the reply, plus the builder and history handed back.
struct TurnDone {
    builder: TemplateBuilder,
    history: Vec<ChatMessage>,
    turn: Turn,
}

async fn run_turn(
    mut builder: TemplateBuilder,
    mut history: Vec<ChatMessage>,
    llm: Arc<dyn LlmClient>,
    model: String,
    text: String,
) -> TurnDone {
    let turn = async {
        let first = builder.generate(llm.as_ref(), &text, &model, &mut history);
        let mut result = match tokio::time::timeout(LLM_TIMEOUT, first).await {
            Err(_) => return Turn::TimedOut,
            Ok(Err(e)) => return Turn::Failed(e),
            Ok(Ok(result)) => result,
        };
        if !result.error.is_empty() || result.validation_errors.is_empty() {
            return Turn::Done(Box::new(result));
        }
        for _ in 0..MAX_AUTO_FIX {
            let errors = result.validation_errors.clone();
            let fix = builder.auto_fix(llm.as_ref(), &errors, &model, &mut history);
            result = match tokio::time::timeout(LLM_TIMEOUT, fix).await {
                Err(_) => return Turn::TimedOut,
                Ok(Err(e)) => return Turn::Failed(e),
                Ok(Ok(result)) => result,
            };
            if result.validation_errors.is_empty() && result.error.is_empty() {
                break;
            }
        }
        Turn::Done(Box::new(result))
    }
    .await;
    TurnDone {
        builder,
        history,
        turn,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    User,
    Ai,
}

/// The names the save dialog starts with.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SaveSuggestion {
    pub file_name: String,
    pub display_name: String,
}

/// The model and client, once the config is read.
struct Ready {
    llm: Arc<dyn LlmClient>,
    model: String,
    provider: String,
}

pub struct TemplateBuilderRuntime {
    setup: Setup,
    ready: Option<Ready>,
    /// None while a call holds it.
    builder: Option<TemplateBuilder>,
    history: Vec<ChatMessage>,
    base_template: Option<SessionTemplate>,
    base_key: String,
    current: Option<SessionTemplate>,
    saved: bool,
    busy: bool,
    chat: Vec<(Role, String)>,
    /// Goes up each time the preview's Markdown changes.
    preview_revision: u64,
    task: Option<tokio::task::AbortHandle>,
    templates: YamlTemplates,
}

impl TemplateBuilderRuntime {
    /// Create a template, or edit `base` (saved back as `base_key` by default).
    pub fn new(
        setup: Setup,
        templates: YamlTemplates,
        base: Option<(SessionTemplate, String)>,
    ) -> Self {
        let (base_template, base_key) = match base {
            Some((template, key)) => (Some(template), key),
            None => (None, String::new()),
        };
        let welcome = match base_template {
            None => WELCOME,
            Some(_) => EDIT_WELCOME,
        };
        Self {
            setup,
            ready: None,
            builder: None,
            history: Vec::new(),
            saved: base_template.is_some(),
            current: base_template.clone(),
            base_template,
            base_key,
            busy: false,
            chat: vec![(Role::Ai, welcome.to_string())],
            preview_revision: 0,
            task: None,
            templates,
        }
    }

    /// The title: "Template Builder", or the name of the template in edit.
    pub fn title(&self) -> String {
        match &self.base_template {
            None => "Template Builder".to_string(),
            Some(template) => {
                let name = match template.metadata.name.is_empty() {
                    true => &self.base_key,
                    false => &template.metadata.name,
                };
                format!("Template Builder: Editing: {name}")
            }
        }
    }

    /// The hint in the empty message input.
    pub fn placeholder(&self) -> &'static str {
        match self.base_template {
            None => "Describe your template...",
            Some(_) => "Describe the change...",
        }
    }

    pub fn chat(&self) -> &[(Role, String)] {
        &self.chat
    }

    /// True while a call to the model runs.
    pub fn busy(&self) -> bool {
        self.busy
    }

    pub fn current(&self) -> Option<&SessionTemplate> {
        self.current.as_ref()
    }

    /// The preview's Markdown: the current template, or `NO_TEMPLATE`.
    pub fn preview(&self) -> String {
        match &self.current {
            Some(template) => preview_markdown(template),
            None => NO_TEMPLATE.to_string(),
        }
    }

    /// Changes each time `preview` changes, so a view redraws it only then.
    pub fn preview_revision(&self) -> u64 {
        self.preview_revision
    }

    fn append(&mut self, role: Role, text: &str) {
        self.chat.push((role, text.to_string()));
    }

    /// Send the user's text to the model. The reply comes back as a task result
    /// for `task_result`. False while a call runs.
    pub fn send(&mut self, text: String, tasks: &Tasks) -> bool {
        if self.busy {
            return false;
        }
        self.append(Role::User, &text);
        if self.ready.is_none() {
            match (self.setup)() {
                Ok(setup) => {
                    self.builder = Some(TemplateBuilder::new(
                        &setup.example,
                        self.base_template.clone(),
                    ));
                    self.ready = Some(Ready {
                        llm: setup.llm,
                        model: setup.model,
                        provider: setup.provider,
                    });
                }
                Err(e) => {
                    // A bad config must leave the builder usable for a retry.
                    error!(target: "ltn.template_builder", "Template generation failed: {e}");
                    self.append(Role::Ai, &format!("Error: {e}"));
                    return true;
                }
            }
        }
        self.busy = true;
        let ready = self.ready.as_ref().expect("set above");
        let builder = self.builder.take().expect("no call is running");
        let history = std::mem::take(&mut self.history);
        // A panic in the call gives the builder back as it was, ready for a retry,
        // as Python's `except Exception` does.
        let (kept_builder, kept_history) = (builder.clone(), history.clone());
        self.task = Some(tasks.spawn_or(
            run_turn(
                builder,
                history,
                ready.llm.clone(),
                ready.model.clone(),
                text,
            ),
            move |message| TurnDone {
                builder: kept_builder,
                history: kept_history,
                turn: Turn::Failed(LlmError::new("panic", message)),
            },
        ));
        true
    }

    /// A task result. Gives the payload back when it is not a reply.
    pub fn task_result(&mut self, payload: Payload) -> Result<(), Payload> {
        let done = payload.downcast::<TurnDone>()?;
        self.turn_done(*done);
        Ok(())
    }

    fn turn_done(&mut self, done: TurnDone) {
        self.task = None;
        self.builder = Some(done.builder);
        self.history = done.history;
        let (model, provider) = self
            .ready
            .as_ref()
            .map(|ready| (ready.model.clone(), ready.provider.clone()))
            .unwrap_or_default();
        match done.turn {
            Turn::TimedOut => {
                error!(
                    target: "ltn.template_builder",
                    "Template builder LLM call got no answer within {}s",
                    LLM_TIMEOUT.as_secs()
                );
                let minutes = LLM_TIMEOUT.as_secs_f64() / 60.0;
                self.append(
                    Role::Ai,
                    &format!(
                        "Error: the AI model did not answer within {} minutes. Send it again.",
                        minutes.round()
                    ),
                );
            }
            Turn::Failed(e) => {
                error!(target: "ltn.template_builder", "Template generation failed: {}: {}", e.kind, e.message);
                self.append(Role::Ai, &error_text(&e, &model, &provider));
            }
            Turn::Done(result) => self.show_result(*result),
        }
        self.busy = false;
    }

    fn show_result(&mut self, result: BuilderResult) {
        if !result.error.is_empty() {
            let text = format!(
                "{}\n\n*(Error: {})*",
                result.assistant_message, result.error
            );
            self.append(Role::Ai, &text);
            return;
        }
        let Some(template) = result.template else {
            self.append(Role::Ai, &result.assistant_message);
            return;
        };
        let previous = self.current.replace(template.clone());
        self.saved = false;
        self.preview_revision += 1;
        let message = match result.assistant_message.is_empty() {
            true => "Template generated.",
            false => &result.assistant_message,
        };
        self.append(Role::Ai, message);
        if let (Some(_), Some(previous)) = (&self.base_template, previous) {
            let changes = diff_templates(&previous, &template);
            let text = match changes.is_empty() {
                true => "No field changed.".to_string(),
                false => format!("Changed: {}", changes.join(", ")),
            };
            self.append(Role::Ai, &text);
        }
    }

    /// The names to offer for a save. None when there is no template yet.
    pub fn save_suggestion(&self) -> Option<SaveSuggestion> {
        let current = self.current.as_ref()?;
        let file_name = if !self.base_key.is_empty() {
            self.base_key.clone()
        } else if !current.metadata.key.is_empty() {
            current.metadata.key.clone()
        } else {
            slugify(&current.metadata.name)
        };
        Some(SaveSuggestion {
            file_name,
            display_name: current.metadata.name.clone(),
        })
    }

    /// Save the template as `file_name` (made safe), renamed to `display_name`
    /// when that is set. Gives the safe name and the path, or the error text.
    /// None when there is no template.
    pub fn save(
        &mut self,
        file_name: &str,
        display_name: &str,
    ) -> Option<Result<(String, PathBuf), String>> {
        let current = self.current.as_mut()?;
        if !display_name.is_empty() && display_name != current.metadata.name {
            current.metadata.name = display_name.to_string();
            self.preview_revision += 1;
        }
        let safe_name = slugify(file_name);
        Some(
            match self.templates.save_user_template(current, &safe_name) {
                Ok(path) => {
                    self.saved = true;
                    Ok((safe_name, path))
                }
                Err(e) => {
                    error!(target: "ltn.template_builder", "Cannot save template {safe_name}: {e}");
                    Err(e.to_string())
                }
            },
        )
    }

    /// True when leaving would lose a template: ask first.
    pub fn unsaved(&self) -> bool {
        self.current.is_some() && !self.saved
    }

    /// Close the builder; a running call ends with it, as Textual cancels the worker.
    pub fn leave(&mut self) {
        if let Some(task) = self.task.take() {
            task.abort();
        }
    }
}

#[cfg(test)]
#[path = "template_builder_tests.rs"]
mod tests;
