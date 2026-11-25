//! The window's root: the welcome menu, then one mode at a time, then back to
//! the menu, as the TUI's kiosk shell. It builds each session from `app/` and
//! shows one view at a time; the views hold no navigation.

use std::path::{Path, PathBuf};
use std::time::Duration;

use gpui_kit::*;
use tracing::error;

use crate::adapters::paths::Paths;
use crate::adapters::templates::YamlTemplates;
use crate::adapters::whisper::transcriber::SubprocessWhisperTranscriber;
use crate::app::container::{Loaded, check_transcription_provider, load_config, resolve_base_dir};
use crate::app::headless_run::load_template_key;
use crate::app::keep_awake::{KeepAwake, keep_awake};
use crate::app::logging::close_file_logging;
use crate::app::preflight::Reload;
use crate::app::preflight_runtime::PreflightRuntime;
use crate::app::session::Severity;
use crate::app::session_start::{
    TranscribeSource, TranscribeStart, build_plan, preflight_llm, record_start,
};
use crate::app::template_builder::{TemplateBuilderRuntime, config_setup};
use crate::domain::template::SessionTemplate;
use crate::gui::bridge::notify;
use crate::gui::builder::{BuilderEvent, BuilderView};
use crate::gui::preflight::{PreflightEvent, PreflightView};
use crate::gui::session::{SessionEvent, SessionView};
use crate::gui::session_picker::{SessionPickerEvent, SessionPickerView};
use crate::gui::settings::{SettingsEvent, SettingsView};
use crate::gui::template_picker::{TemplatePickerEvent, TemplatePickerView};
use crate::gui::view::{ReaderEvent, SessionReader};
use crate::gui::welcome::{Welcome, WelcomeEvent};

const START_ERROR_TIMEOUT: Duration = Duration::from_secs(12);

#[derive(Clone, Copy, PartialEq, Eq)]
enum Flow {
    Record,
    Transcribe,
}

/// The screen the window shows.
enum Screen {
    Welcome(Entity<Welcome>),
    TemplatePicker(Entity<TemplatePickerView>, Flow),
    /// The builder, opened from the template picker of `Some(flow)`, or from the menu.
    Builder(Entity<BuilderView>, Option<Flow>),
    Preflight(Entity<PreflightView>),
    /// The AI check of a transcription runs.
    Checking,
    Session(Entity<SessionView>),
    SessionPicker(Entity<SessionPickerView>),
    Reader(Entity<SessionReader>),
    Settings(Entity<SettingsView>),
}

pub struct Shell {
    screen: Screen,
    last_mode: Option<&'static str>,
    /// The config the current flow loaded.
    loaded: Option<Loaded>,
    template: Option<SessionTemplate>,
    audio_path: Option<PathBuf>,
    /// Held while a recording runs.
    awake: Option<KeepAwake>,
    /// `debug.log` is open for a session.
    logging: bool,
    _subscription: Option<Subscription>,
}

/// The shell of the one window, for the app-level Quit action.
struct ShellHandle(WeakEntity<Shell>, AnyWindowHandle);

impl Global for ShellHandle {}

/// LTN_OUTPUT_DIR, which the app sets when nothing else does.
fn output_dir() -> Option<String> {
    std::env::var("LTN_OUTPUT_DIR").ok()
}

impl Shell {
    pub fn new(window: &mut Window, cx: &mut Context<Self>) -> Self {
        crate::gui::welcome::bind_keys(cx);
        let welcome = cx.new(|cx| Welcome::new(None, window, cx));
        let mut shell = Self {
            screen: Screen::Welcome(welcome),
            last_mode: None,
            loaded: None,
            template: None,
            audio_path: None,
            awake: None,
            logging: false,
            _subscription: None,
        };
        shell.show_welcome(window, cx);
        shell
    }

    fn show(&mut self, screen: Screen, subscription: Subscription, cx: &mut Context<Self>) {
        self.screen = screen;
        self._subscription = Some(subscription);
        cx.notify();
    }

    fn show_welcome(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        // A theme saved in Settings shows from here on, as in the terminal app.
        crate::gui::theme::apply(&crate::app::theme::saved_theme(), cx);
        let last_mode = self.last_mode;
        let welcome = cx.new(|cx| Welcome::new(last_mode, window, cx));
        let subscription = cx.subscribe_in(&welcome, window, Self::on_welcome);
        self.show(Screen::Welcome(welcome), subscription, cx);
    }

    /// A mode ended: release what it held, then the menu.
    fn finished(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.awake = None;
        if std::mem::take(&mut self.logging) {
            close_file_logging();
        }
        self.show_welcome(window, cx);
    }

    fn failed(&mut self, step: &str, message: &str, window: &mut Window, cx: &mut Context<Self>) {
        error!(target: "ltn.app", "{step} failed: {message}");
        notify(
            window,
            cx,
            format!("Could not start: {message}"),
            Severity::Error,
            START_ERROR_TIMEOUT,
        );
        self.finished(window, cx);
    }

    fn load(&self) -> Result<Loaded, String> {
        load_config(None, output_dir().as_deref())
    }

    fn on_welcome(
        &mut self,
        _: &Entity<Welcome>,
        event: &WelcomeEvent,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        match event {
            WelcomeEvent::Open(mode) => {
                self.last_mode = Some(mode);
                self.open_mode(mode, window, cx);
            }
            // Nothing records on the menu, and the shell is mid-update here,
            // so the app quits without asking the shell again.
            WelcomeEvent::Quit => cx.quit(),
        }
    }

    fn open_mode(&mut self, mode: &str, window: &mut Window, cx: &mut Context<Self>) {
        match mode {
            "record" => self.open_picker(Flow::Record, window, cx),
            "transcribe" => self.choose_audio(window, cx),
            "view" => self.open_session_picker(window, cx),
            "create-template" => self.open_builder(None, None, window, cx),
            "config" => self.open_settings(window, cx),
            _ => self.finished(window, cx),
        }
    }

    /// The system's file dialog, then the template picker.
    fn choose_audio(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let paths = cx.prompt_for_paths(PathPromptOptions {
            files: true,
            directories: false,
            multiple: false,
            prompt: Some("Transcribe".into()),
        });
        cx.spawn_in(window, async move |this, cx| {
            let picked = match paths.await {
                Ok(Ok(Some(paths))) => paths.into_iter().next(),
                _ => None,
            };
            let _ = this.update_in(cx, |shell, window, cx| match picked {
                Some(path) => {
                    shell.audio_path = Some(path);
                    shell.open_picker(Flow::Transcribe, window, cx);
                }
                None => shell.finished(window, cx),
            });
        })
        .detach();
    }

    /// Load config.yaml again and show the template picker.
    fn open_picker(&mut self, flow: Flow, window: &mut Window, cx: &mut Context<Self>) {
        let loaded = match self.load() {
            Ok(loaded) => loaded,
            Err(e) => return self.failed("open_picker", &e, window, cx),
        };
        let picker = cx.new(|cx| TemplatePickerView::new(&loaded, window, cx));
        self.loaded = Some(loaded);
        let subscription = cx.subscribe_in(&picker, window, move |shell, _, event, window, cx| {
            shell.on_template(flow, event, window, cx)
        });
        self.show(Screen::TemplatePicker(picker, flow), subscription, cx);
    }

    fn on_template(
        &mut self,
        flow: Flow,
        event: &TemplatePickerEvent,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(loaded) = self.loaded.as_ref() else {
            return self.failed("on_template", "no config loaded", window, cx);
        };
        let templates = YamlTemplates::new(&loaded.paths);
        let key = match event {
            TemplatePickerEvent::Cancelled => return self.finished(window, cx),
            TemplatePickerEvent::Create => return self.open_builder(Some(flow), None, window, cx),
            TemplatePickerEvent::EditWithAi(key) => {
                return match load_template_key(&templates, key) {
                    Ok(base) => {
                        self.open_builder(Some(flow), Some((base, key.clone())), window, cx)
                    }
                    Err(e) => self.failed("on_template", &e, window, cx),
                };
            }
            TemplatePickerEvent::Picked(key) => key,
        };
        match load_template_key(&templates, key) {
            Ok(template) if flow == Flow::Record => self.start_record(template, window, cx),
            Ok(template) => self.start_transcribe(&template, window, cx),
            Err(e) => self.failed("on_template", &e, window, cx),
        }
    }

    fn open_builder(
        &mut self,
        flow: Option<Flow>,
        base: Option<(SessionTemplate, String)>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let paths = match self.load() {
            Ok(loaded) => loaded.paths,
            Err(e) => return self.failed("open_template_builder", &e, window, cx),
        };
        let runtime = TemplateBuilderRuntime::new(
            config_setup(&paths, None),
            YamlTemplates::new(&paths),
            base,
        );
        let builder = cx.new(|cx| BuilderView::new(runtime, window, cx));
        let subscription = cx.subscribe_in(&builder, window, move |shell, _, event, window, cx| {
            let BuilderEvent::Closed = event;
            match flow {
                Some(flow) => shell.open_picker(flow, window, cx),
                None => shell.finished(window, cx),
            }
        });
        self.show(Screen::Builder(builder, flow), subscription, cx);
    }

    fn start_record(
        &mut self,
        template: SessionTemplate,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(loaded) = self.loaded.as_ref() else {
            return self.failed("start_record", "no config loaded", window, cx);
        };
        let runtime = match preflight_runtime(loaded, &template) {
            Ok(runtime) => runtime,
            Err(e) => return self.failed("start_record", &e, window, cx),
        };
        self.template = Some(template);
        let preflight = cx.new(|cx| PreflightView::new(runtime, window, cx));
        let subscription = cx.subscribe_in(&preflight, window, Self::on_preflight);
        self.show(Screen::Preflight(preflight), subscription, cx);
    }

    fn on_preflight(
        &mut self,
        view: &Entity<PreflightView>,
        event: &PreflightEvent,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if let PreflightEvent::Back = event {
            return self.finished(window, cx);
        }
        let Some(outcome) = view.update(cx, |view, _| view.take_outcome()) else {
            return self.finished(window, cx);
        };
        let Some(paths) = self.loaded.as_ref().map(|loaded| loaded.paths.clone()) else {
            return self.failed("after_preflight", "no config loaded", window, cx);
        };
        match record_start(outcome, &paths, None, false) {
            Ok(start) => {
                self.logging = true;
                self.awake = Some(keep_awake());
                let session = cx.new(|cx| SessionView::record(start, window, cx));
                let subscription = cx.subscribe_in(&session, window, Self::on_session);
                self.show(Screen::Session(session), subscription, cx);
            }
            Err(e) => self.failed("after_preflight", &e, window, cx),
        }
    }

    fn start_transcribe(
        &mut self,
        template: &SessionTemplate,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(loaded) = self.loaded.as_ref() else {
            return self.failed("start_transcribe", "no config loaded", window, cx);
        };
        let start = match TranscribeStart::new(loaded, template, output_dir().as_deref(), None) {
            Ok(start) => start,
            Err(e) => return self.failed("start_transcribe", &e, window, cx),
        };
        self.screen = Screen::Checking;
        cx.notify();
        let checks = crate::gui::runtime().spawn(async move {
            // The window has no console: the warning goes to the log.
            let warn = |line: &str| tracing::warn!(target: "ltn.app", "{line}");
            let missing = preflight_llm(start.llm.as_ref(), &start.config, warn).await;
            (start, missing)
        });
        cx.spawn_in(window, async move |this, cx| {
            let done = checks.await;
            let _ = this.update_in(cx, |shell, window, cx| match done {
                Ok((start, missing)) => shell.after_checks(start, missing, window, cx),
                Err(e) => shell.failed("start_transcribe", &e.to_string(), window, cx),
            });
        })
        .detach();
    }

    fn after_checks(
        &mut self,
        start: TranscribeStart,
        missing: (Vec<String>, Vec<String>),
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(audio_path) = self.audio_path.clone() else {
            return self.failed("start_transcribe", "no audio file", window, cx);
        };
        match start.start(&TranscribeSource::Audio(audio_path), missing) {
            Ok(start) => {
                self.logging = true;
                let session = cx.new(|cx| SessionView::file(start, window, cx));
                let subscription = cx.subscribe_in(&session, window, Self::on_session);
                self.show(Screen::Session(session), subscription, cx);
            }
            Err(e) => self.failed("start_transcribe", &e, window, cx),
        }
    }

    fn on_session(
        &mut self,
        _: &Entity<SessionView>,
        event: &SessionEvent,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        match event {
            SessionEvent::Closed => self.finished(window, cx),
            SessionEvent::Exit => {
                self.finished(window, cx);
                cx.quit();
            }
        }
    }

    fn open_session_picker(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let loaded = match self.load() {
            Ok(loaded) => loaded,
            Err(e) => return self.failed("open_session_picker", &e, window, cx),
        };
        let base_dir = resolve_base_dir(&loaded.paths, output_dir().as_deref(), &loaded.config);
        let picker = cx.new(|cx| SessionPickerView::new(&base_dir, window, cx));
        let subscription =
            cx.subscribe_in(&picker, window, |shell, _, event, window, cx| match event {
                SessionPickerEvent::Picked(dir) => shell.open_reader(dir, window, cx),
                SessionPickerEvent::Cancelled => shell.finished(window, cx),
            });
        self.show(Screen::SessionPicker(picker), subscription, cx);
    }

    fn open_reader(&mut self, dir: &Path, window: &mut Window, cx: &mut Context<Self>) {
        let reader = cx.new(|cx| SessionReader::new(dir.to_path_buf(), window, cx));
        // The picker comes back after each session, as in the terminal app.
        let subscription = cx.subscribe_in(&reader, window, |shell, _, event, window, cx| {
            let ReaderEvent::Closed = event;
            shell.open_session_picker(window, cx);
        });
        self.show(Screen::Reader(reader), subscription, cx);
    }

    fn open_settings(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let Some(paths) = Paths::for_user() else {
            return self.failed("open_config", "no home folder", window, cx);
        };
        let settings = cx.new(|cx| SettingsView::new(paths, None, window, cx));
        let subscription = cx.subscribe_in(&settings, window, |shell, _, event, window, cx| {
            let SettingsEvent::Closed = event;
            shell.finished(window, cx);
        });
        self.show(Screen::Settings(settings), subscription, cx);
    }

    /// The window may close now. Otherwise the session starts its way out
    /// (stop the recording, wait for the audio flush) and closes the app after.
    fn should_close(&mut self, window: &mut Window, cx: &mut Context<Self>) -> bool {
        match &self.screen {
            Screen::Session(session) => {
                session.update(cx, |view, cx| view.should_close(window, cx))
            }
            _ => true,
        }
    }
}

/// The preflight for `template`, with a reload that reads config.yaml again on Retry.
fn preflight_runtime(
    loaded: &Loaded,
    template: &SessionTemplate,
) -> Result<PreflightRuntime, String> {
    check_transcription_provider(&loaded.infra)?;
    let runtime = crate::gui::runtime();
    let plan = build_plan(
        loaded,
        template,
        output_dir().as_deref(),
        runtime.clone(),
        false,
    )?;
    let transcriber = SubprocessWhisperTranscriber::new().map_err(|e| e.to_string())?;
    let template = template.clone();
    let reload: Reload = Box::new(move || {
        let output_dir = output_dir();
        let loaded = load_config(None, output_dir.as_deref())?;
        build_plan(
            &loaded,
            &template,
            output_dir.as_deref(),
            runtime.clone(),
            false,
        )
    });
    Ok(PreflightRuntime::new(
        plan,
        Box::new(transcriber),
        Some(reload),
    ))
}

impl Render for Shell {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let body: AnyElement = match &self.screen {
            Screen::Welcome(view) => view.clone().into_any_element(),
            Screen::TemplatePicker(view, _) => view.clone().into_any_element(),
            Screen::Builder(view, _) => view.clone().into_any_element(),
            Screen::Preflight(view) => view.clone().into_any_element(),
            Screen::Checking => div()
                .size_full()
                .flex()
                .items_center()
                .justify_center()
                .text_color(crate::gui::theme::color(cx, "text-muted"))
                .child("Checking the AI provider...")
                .into_any_element(),
            Screen::Session(view) => view.clone().into_any_element(),
            Screen::SessionPicker(view) => view.clone().into_any_element(),
            Screen::Reader(view) => view.clone().into_any_element(),
            Screen::Settings(view) => view.clone().into_any_element(),
        };
        div()
            .size_full()
            .bg(crate::gui::theme::color(cx, "background"))
            .text_color(crate::gui::theme::color(cx, "foreground"))
            .child(body)
    }
}

/// Route the window's close button through `Shell::should_close`, and quit
/// the app when the window is gone.
pub fn watch_close(handle: AnyWindowHandle, shell: Entity<Shell>, cx: &mut App) {
    cx.set_global(ShellHandle(shell.downgrade(), handle));
    let weak = shell.downgrade();
    let _ = handle.update(cx, |_, window, cx| {
        window.on_window_should_close(cx, move |window, cx| {
            weak.update(cx, |shell, cx| shell.should_close(window, cx))
                .unwrap_or(true)
        });
    });
    cx.on_window_closed(|cx, _| cx.quit()).detach();
}

/// Cmd-Q and Esc on the menu: the same way out as the close button.
pub fn request_quit(cx: &mut App) {
    let Some(ShellHandle(shell, handle)) = cx.try_global::<ShellHandle>() else {
        cx.quit();
        return;
    };
    let (shell, handle) = (shell.clone(), *handle);
    let closes = handle
        .update(cx, |_, window, cx| {
            shell
                .update(cx, |shell, cx| shell.should_close(window, cx))
                .unwrap_or(true)
        })
        .unwrap_or(true);
    if closes {
        cx.quit();
    }
}
