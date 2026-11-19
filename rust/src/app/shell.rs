//! The kiosk shell: `lazy-take-notes` with no command. One host runs the welcome
//! picker, the pickers and each mode, as Python's `ShellApp`.

use std::path::{Path, PathBuf};

use crossterm::event::KeyEvent;
use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::Style;
use ratatui::widgets::Paragraph;
use tracing::error;

use crate::adapters::templates::YamlTemplates;
use crate::app::container::{Loaded, load_config, resolve_base_dir};
use crate::app::headless_run::load_template_key;
use crate::app::keep_awake::{KeepAwake, keep_awake};
use crate::app::logging::close_file_logging;
use crate::app::tui_run::{
    TranscribeStart, builder_screen, preflight_llm, preflight_screen, record_screen,
};
use crate::domain::template::SessionTemplate;
use crate::tui::host::{Action, Ctx, Msg, Payload, Screen, Severity};
use crate::tui::screens::file_picker::FilePicker;
use crate::tui::screens::preflight::PreflightOutcome;
use crate::tui::screens::session_picker::{SessionPicker, picked_dir};
use crate::tui::screens::settings::SettingsScreen;
use crate::tui::screens::template_picker::{TemplateChoice, TemplatePicker};
use crate::tui::screens::view::ViewScreen;
use crate::tui::screens::welcome::{ChosenMode, WelcomePicker};

const START_ERROR_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(12);

#[derive(Clone, Copy, PartialEq, Eq)]
enum Flow {
    Record,
    /// Transcribe this file.
    Transcribe,
}

/// What the screen above the shell will hand back when it pops.
enum Waiting {
    Welcome,
    TemplatePicker(Flow),
    /// The builder, opened from the template picker of this flow.
    BuilderThenPicker(Flow),
    Preflight,
    FilePicker,
    /// The LLM preflight of a transcription runs; nothing is above the shell.
    TranscribeChecks,
    SessionPicker,
    /// A mode screen; its end goes back to the welcome picker.
    Mode,
}

/// The LLM preflight result for a transcription.
struct ChecksDone(TranscribeStart, (Vec<String>, Vec<String>));

pub struct Shell {
    config_path: Option<PathBuf>,
    output_dir: Option<String>,
    kiosk: bool,
    last_mode: Option<&'static str>,
    waiting: Waiting,
    /// The config the current flow loaded, as Python's `_record_state`.
    loaded: Option<Loaded>,
    audio_path: Option<PathBuf>,
    /// Held while a recording runs.
    awake: Option<KeepAwake>,
    /// `debug.log` is open for a session.
    logging: bool,
}

impl Shell {
    /// `kiosk` (LTN_KIOSK=1) comes back to the welcome picker after each mode;
    /// otherwise the shell ends after one.
    pub fn new(config_path: Option<&Path>, output_dir: Option<&str>, kiosk: bool) -> Self {
        Self {
            config_path: config_path.map(Path::to_path_buf),
            output_dir: output_dir.map(str::to_string),
            kiosk,
            last_mode: None,
            waiting: Waiting::Welcome,
            loaded: None,
            audio_path: None,
            awake: None,
            logging: false,
        }
    }

    fn welcome(&mut self) -> Action {
        self.waiting = Waiting::Welcome;
        Action::Push(Box::new(WelcomePicker::new(self.last_mode)))
    }

    /// A mode ended: release what it held, then the welcome picker or the exit.
    fn finished(&mut self) -> Action {
        self.awake = None;
        if std::mem::take(&mut self.logging) {
            close_file_logging();
        }
        if self.kiosk {
            self.welcome()
        } else {
            Action::Pop(None)
        }
    }

    /// Python's `_back_to_welcome_on_error`.
    fn failed(&mut self, step: &str, message: &str, ctx: &mut Ctx) -> Action {
        error!(target: "ltn.app", "{step} failed: {message}");
        ctx.notify_with(
            format!("Could not start: {message}"),
            Severity::Error,
            START_ERROR_TIMEOUT,
            "",
        );
        self.finished()
    }

    fn open_mode(&mut self, mode: &str, ctx: &mut Ctx) -> Action {
        match mode {
            "record" => self.open_picker(Flow::Record, ctx),
            "transcribe" => {
                // Python's `FilePicker()` with no start folder opens the home folder.
                self.waiting = Waiting::FilePicker;
                match dirs::home_dir() {
                    Some(home) => Action::Push(Box::new(FilePicker::new(&home))),
                    None => self.failed("open_transcribe", "no home folder", ctx),
                }
            }
            "view" => self.open_session_picker(ctx),
            "create-template" => {
                let paths = match self.load() {
                    Ok(loaded) => loaded.paths,
                    Err(e) => return self.failed("open_template_builder", &e, ctx),
                };
                self.waiting = Waiting::Mode;
                let screen = builder_screen(&paths, self.config_path.as_deref(), None);
                Action::Push(Box::new(screen))
            }
            "config" => match crate::adapters::paths::Paths::for_user() {
                Some(paths) => {
                    self.waiting = Waiting::Mode;
                    let screen = SettingsScreen::new(paths, self.config_path.clone());
                    Action::Push(Box::new(screen))
                }
                None => self.failed("open_config", "no home folder", ctx),
            },
            _ => self.finished(),
        }
    }

    fn load(&self) -> Result<Loaded, String> {
        load_config(self.config_path.as_deref(), self.output_dir.as_deref())
    }

    /// Load config.yaml again and show the template picker, as `_open_record`.
    fn open_picker(&mut self, flow: Flow, ctx: &mut Ctx) -> Action {
        let loaded = match self.load() {
            Ok(loaded) => loaded,
            Err(e) => return self.failed("open_picker", &e, ctx),
        };
        let picker = TemplatePicker::new(
            YamlTemplates::new(&loaded.paths),
            loaded.infra.show_builtin_templates,
            loaded.paths.builtin_templates_noticed(),
        );
        self.loaded = Some(loaded);
        self.waiting = Waiting::TemplatePicker(flow);
        Action::Push(Box::new(picker))
    }

    fn on_template(&mut self, flow: Flow, choice: TemplateChoice, ctx: &mut Ctx) -> Action {
        let Some(loaded) = self.loaded.as_ref() else {
            return self.failed("on_template", "no config loaded", ctx);
        };
        let templates = YamlTemplates::new(&loaded.paths);
        let key = match choice {
            TemplateChoice::Key(key) => key,
            TemplateChoice::Create => {
                let screen = builder_screen(&loaded.paths, self.config_path.as_deref(), None);
                self.waiting = Waiting::BuilderThenPicker(flow);
                return Action::Push(Box::new(screen));
            }
            TemplateChoice::EditWithAi(key) => {
                let base = match load_template_key(&templates, &key) {
                    Ok(base) => base,
                    Err(e) => return self.failed("on_template", &e, ctx),
                };
                let paths = loaded.paths.clone();
                let screen = builder_screen(&paths, self.config_path.as_deref(), Some((base, key)));
                self.waiting = Waiting::BuilderThenPicker(flow);
                return Action::Push(Box::new(screen));
            }
        };
        match load_template_key(&templates, &key) {
            Ok(template) if flow == Flow::Record => self.start_record(&template, ctx),
            Ok(template) => self.start_transcribe(&template, ctx),
            Err(e) => self.failed("on_template", &e, ctx),
        }
    }

    fn start_record(&mut self, template: &SessionTemplate, ctx: &mut Ctx) -> Action {
        let Some(loaded) = self.loaded.as_ref() else {
            return self.failed("start_record", "no config loaded", ctx);
        };
        let screen = preflight_screen(
            loaded,
            template,
            self.config_path.as_deref(),
            self.output_dir.as_deref(),
            ctx.runtime.clone(),
            false,
        );
        match screen {
            Ok(screen) => {
                self.waiting = Waiting::Preflight;
                Action::Push(Box::new(screen))
            }
            Err(e) => self.failed("start_record", &e, ctx),
        }
    }

    fn after_preflight(&mut self, outcome: PreflightOutcome, ctx: &mut Ctx) -> Action {
        let paths = match self.loaded.as_ref() {
            Some(loaded) => loaded.paths.clone(),
            None => return self.failed("after_preflight", "no config loaded", ctx),
        };
        match record_screen(outcome, &paths, None, false) {
            Ok(screen) => {
                self.logging = true;
                self.awake = Some(keep_awake());
                self.waiting = Waiting::Mode;
                Action::Push(Box::new(screen))
            }
            Err(e) => self.failed("after_preflight", &e, ctx),
        }
    }

    fn start_transcribe(&mut self, template: &SessionTemplate, ctx: &mut Ctx) -> Action {
        let Some(loaded) = self.loaded.as_ref() else {
            return self.failed("start_transcribe", "no config loaded", ctx);
        };
        let start = match TranscribeStart::new(loaded, template, self.output_dir.as_deref(), None) {
            Ok(start) => start,
            Err(e) => return self.failed("start_transcribe", &e, ctx),
        };
        self.waiting = Waiting::TranscribeChecks;
        ctx.spawn(async move {
            // The TUI owns the terminal: the warning goes to the log.
            let warn = |line: &str| tracing::warn!(target: "ltn.app", "{line}");
            let missing = preflight_llm(start.llm.as_ref(), &start.config, warn).await;
            ChecksDone(start, missing)
        });
        Action::None
    }

    fn after_checks(&mut self, done: ChecksDone, ctx: &mut Ctx) -> Action {
        let ChecksDone(start, missing) = done;
        let Some(audio_path) = self.audio_path.clone() else {
            return self.failed("start_transcribe", "no audio file", ctx);
        };
        match start.screen(
            &crate::app::tui_run::TranscribeSource::Audio(audio_path),
            missing,
        ) {
            Ok(screen) => {
                self.logging = true;
                self.waiting = Waiting::Mode;
                Action::Push(Box::new(screen))
            }
            Err(e) => self.failed("start_transcribe", &e, ctx),
        }
    }

    fn open_session_picker(&mut self, ctx: &mut Ctx) -> Action {
        match self.load() {
            Ok(loaded) => {
                let base_dir =
                    resolve_base_dir(&loaded.paths, self.output_dir.as_deref(), &loaded.config);
                self.waiting = Waiting::SessionPicker;
                Action::Push(Box::new(SessionPicker::new(&base_dir)))
            }
            Err(e) => self.failed("open_session_picker", &e, ctx),
        }
    }
}

impl Screen for Shell {
    fn mounted(&mut self, _ctx: &mut Ctx) -> Action {
        self.welcome()
    }

    fn resumed(&mut self, result: Option<Payload>, ctx: &mut Ctx) -> Action {
        let waiting = std::mem::replace(&mut self.waiting, Waiting::Mode);
        match waiting {
            Waiting::Welcome => {
                match result.and_then(|payload| payload.downcast::<ChosenMode>().ok()) {
                    Some(mode) => {
                        self.last_mode = Some(mode.0);
                        self.open_mode(mode.0, ctx)
                    }
                    None => Action::Pop(None),
                }
            }
            Waiting::TemplatePicker(flow) => {
                match result.and_then(|payload| payload.downcast::<TemplateChoice>().ok()) {
                    Some(choice) => self.on_template(flow, *choice, ctx),
                    None => self.finished(),
                }
            }
            Waiting::BuilderThenPicker(flow) => self.open_picker(flow, ctx),
            Waiting::Preflight => {
                match result.and_then(|payload| payload.downcast::<PreflightOutcome>().ok()) {
                    Some(outcome) => self.after_preflight(*outcome, ctx),
                    None => self.finished(),
                }
            }
            Waiting::FilePicker => match picked_dir(result) {
                Some(path) => {
                    self.audio_path = Some(path);
                    self.open_picker(Flow::Transcribe, ctx)
                }
                None => self.finished(),
            },
            Waiting::SessionPicker => match picked_dir(result) {
                Some(dir) => {
                    self.waiting = Waiting::Mode;
                    Action::Push(Box::new(ViewScreen::new(dir)))
                }
                None => self.finished(),
            },
            Waiting::TranscribeChecks | Waiting::Mode => self.finished(),
        }
    }

    fn message(&mut self, msg: Msg, ctx: &mut Ctx) -> (Option<Msg>, Action) {
        let Msg::Task(payload) = msg else {
            return (Some(msg), Action::None);
        };
        match payload.downcast::<ChecksDone>() {
            Ok(done) if matches!(self.waiting, Waiting::TranscribeChecks) => {
                (None, self.after_checks(*done, ctx))
            }
            Ok(_) => (None, Action::None),
            Err(payload) => (Some(Msg::Task(payload)), Action::None),
        }
    }

    fn render(&mut self, frame: &mut Frame, area: Rect, ctx: &Ctx) {
        let background = ctx.theme.color("background");
        frame.render_widget(
            Paragraph::new("").style(Style::default().bg(background)),
            area,
        );
    }

    fn key(&mut self, _key: &KeyEvent, _name: &str, _ctx: &mut Ctx) -> Action {
        Action::None
    }
}

#[cfg(test)]
mod tests {
    use crossterm::event::{KeyCode, KeyModifiers};

    use super::*;
    use crate::tui::host::Host;

    /// A shell whose config file does not exist, so every mode fails to start.
    fn broken_shell(kiosk: bool) -> (Host, tempfile::TempDir, tokio::runtime::Runtime) {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("missing.yaml");
        let mut host = Host::new("textual-dark", runtime.handle().clone());
        host.push(Box::new(Shell::new(Some(&missing), None, kiosk)));
        (host, dir, runtime)
    }

    fn press(host: &mut Host, code: KeyCode) {
        host.key(&KeyEvent::new(code, KeyModifiers::NONE));
    }

    #[test]
    fn a_mode_that_cannot_start_returns_to_the_welcome_picker_in_the_kiosk() {
        let (mut host, _dir, _runtime) = broken_shell(true);
        press(&mut host, KeyCode::Down);
        press(&mut host, KeyCode::Down);
        press(&mut host, KeyCode::Enter);
        let toasts = host.ctx().toast_texts();
        assert_eq!(toasts.len(), 1);
        assert!(toasts[0].starts_with("Could not start: "), "{toasts:?}");
        // The welcome picker is back, on the mode that failed.
        assert_eq!(host.depth(), 2);
        press(&mut host, KeyCode::Enter);
        assert_eq!(host.ctx().toast_texts().len(), 2);
        press(&mut host, KeyCode::Esc);
        assert!(host.finished());
    }

    #[test]
    fn outside_the_kiosk_the_shell_ends_after_one_mode() {
        let (mut host, _dir, _runtime) = broken_shell(false);
        press(&mut host, KeyCode::Enter);
        assert!(host.finished());
    }
}
