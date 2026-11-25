//! The window's root: a sidebar with the commands and the saved sessions, and
//! a detail pane on the right. One recording or transcription runs at a time,
//! and it keeps running while the pane shows something else. The shell builds
//! each session from `app/`; the views hold no navigation.

use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::time::Duration;

use gpui_kit::component::{ActiveTheme, Icon, IconName};
use gpui_kit::prelude::FluentBuilder as _;
use gpui_kit::*;
use tracing::error;

use crate::adapters::paths::Paths;
use crate::adapters::session_reader::{SessionInfo, list_sessions};
use crate::adapters::templates::YamlTemplates;
use crate::adapters::whisper::transcriber::SubprocessWhisperTranscriber;
use crate::app::container::{Loaded, check_transcription_provider, load_config, resolve_base_dir};
use crate::app::headless_run::load_template_key;
use crate::app::keep_awake::{KeepAwake, keep_awake};
use crate::app::logging::close_file_logging;
use crate::app::preflight::Reload;
use crate::app::preflight_runtime::PreflightRuntime;
use crate::app::session::Severity;
use crate::app::session_list::{Entry, grouped, title};
use crate::app::session_start::{
    TranscribeSource, TranscribeStart, build_plan, preflight_llm, record_start,
};
use crate::app::template_builder::{TemplateBuilderRuntime, config_setup};
use crate::domain::template::SessionTemplate;
use crate::gui::bridge::notify;
use crate::gui::builder::{BuilderEvent, BuilderView};
use crate::gui::preflight::{PreflightEvent, PreflightView};
use crate::gui::session::{SessionEvent, SessionView};
use crate::gui::settings::{SettingsEvent, SettingsView};
use crate::gui::template_picker::{TemplatePickerEvent, TemplatePickerView};
use crate::gui::view::{ReaderEvent, SessionReader};
use crate::gui::widgets::{DragArm, brand, mascot, title_strip};

const START_ERROR_TIMEOUT: Duration = Duration::from_secs(12);
const SIDEBAR_WIDTH: Pixels = px(240.);
/// The traffic lights sit in the sidebar's title strip.
pub const TRAFFIC_LIGHTS: Point<Pixels> = Point {
    x: px(16.),
    y: px(16.),
};

#[derive(Clone, Copy, PartialEq, Eq)]
enum Flow {
    Record,
    Transcribe,
}

/// The step a recording or transcription is at.
enum Stage {
    TemplatePicker(Entity<TemplatePickerView>),
    Builder(Entity<BuilderView>),
    Preflight(Entity<PreflightView>),
    /// The AI check of a transcription runs.
    Checking,
    Session(Entity<SessionView>),
}

/// The one recording or transcription, from its template choice to its end.
struct Live {
    flow: Flow,
    stage: Stage,
    _subscription: Option<Subscription>,
}

/// What the detail pane shows.
enum Detail {
    Empty,
    Live,
    Saved(Entity<SessionReader>, PathBuf),
    Templates(Entity<TemplatePickerView>),
    Builder(Entity<BuilderView>),
    Settings(Entity<SettingsView>),
}

pub struct Shell {
    live: Option<Live>,
    detail: Detail,
    sessions: Vec<SessionInfo>,
    /// Why the session list is empty, when config.yaml did not load.
    list_error: Option<String>,
    /// The live session's folder, as last seen: a rename moves it.
    live_dir: Option<PathBuf>,
    /// The config the current flow loaded.
    loaded: Option<Loaded>,
    template: Option<SessionTemplate>,
    audio_path: Option<PathBuf>,
    /// Held while a recording runs.
    awake: Option<KeepAwake>,
    /// `debug.log` is open for a session.
    logging: bool,
    sidebar_drag: DragArm,
    pane_drag: DragArm,
    _detail_subscription: Option<Subscription>,
    _live_observer: Option<Subscription>,
    _appearance: Subscription,
}

/// The shell of the one window, for the app-level actions.
struct ShellHandle(WeakEntity<Shell>, AnyWindowHandle);

impl Global for ShellHandle {}

/// LTN_OUTPUT_DIR, which the app sets when nothing else does.
fn output_dir() -> Option<String> {
    std::env::var("LTN_OUTPUT_DIR").ok()
}

impl Shell {
    pub fn new(window: &mut Window, cx: &mut Context<Self>) -> Self {
        let appearance = crate::gui::theme::follow_system(window, cx);
        let mut shell = Self {
            live: None,
            detail: Detail::Empty,
            sessions: Vec::new(),
            list_error: None,
            live_dir: None,
            loaded: None,
            template: None,
            audio_path: None,
            awake: None,
            logging: false,
            sidebar_drag: DragArm::default(),
            pane_drag: DragArm::default(),
            _detail_subscription: None,
            _live_observer: None,
            _appearance: appearance,
        };
        shell.refresh_sessions();
        shell
    }

    fn refresh_sessions(&mut self) {
        match self.load() {
            Ok(loaded) => {
                let base = resolve_base_dir(&loaded.paths, output_dir().as_deref(), &loaded.config);
                self.sessions = list_sessions(&base);
                self.list_error = None;
            }
            Err(e) => {
                self.sessions.clear();
                self.list_error = Some(e);
            }
        }
    }

    fn load(&self) -> Result<Loaded, String> {
        load_config(None, output_dir().as_deref())
    }

    fn show_detail(
        &mut self,
        detail: Detail,
        subscription: Option<Subscription>,
        cx: &mut Context<Self>,
    ) {
        self.detail = detail;
        self._detail_subscription = subscription;
        cx.notify();
    }

    fn set_stage(&mut self, stage: Stage, subscription: Subscription, cx: &mut Context<Self>) {
        if let Some(live) = self.live.as_mut() {
            live.stage = stage;
            live._subscription = Some(subscription);
        }
        self.show_detail(Detail::Live, None, cx);
    }

    /// The live flow ended: release what it held, and show `next`.
    fn finished(&mut self, next: Option<PathBuf>, window: &mut Window, cx: &mut Context<Self>) {
        self.live = None;
        self.live_dir = None;
        self._live_observer = None;
        self.awake = None;
        if std::mem::take(&mut self.logging) {
            close_file_logging();
        }
        self.refresh_sessions();
        match next {
            Some(dir) if dir.is_dir() => self.open_saved(&dir, window, cx),
            _ if matches!(self.detail, Detail::Live) => self.show_detail(Detail::Empty, None, cx),
            _ => cx.notify(),
        }
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
        self.finished(None, window, cx);
    }

    /// New Recording or Transcribe File. With a flow running, the pane shows it instead.
    fn start(&mut self, flow: Flow, window: &mut Window, cx: &mut Context<Self>) {
        if self.live.is_some() {
            return self.show_detail(Detail::Live, None, cx);
        }
        self.live = Some(Live {
            flow,
            stage: Stage::Checking,
            _subscription: None,
        });
        match flow {
            Flow::Record => self.open_picker(window, cx),
            Flow::Transcribe => self.choose_audio(window, cx),
        }
    }

    fn flow(&self) -> Option<Flow> {
        self.live.as_ref().map(|live| live.flow)
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
                    shell.open_picker(window, cx);
                }
                None => shell.finished(None, window, cx),
            });
        })
        .detach();
    }

    /// Load config.yaml again and show the template picker.
    fn open_picker(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let loaded = match self.load() {
            Ok(loaded) => loaded,
            Err(e) => return self.failed("open_picker", &e, window, cx),
        };
        let picker = cx.new(|cx| TemplatePickerView::new(&loaded, window, cx));
        self.loaded = Some(loaded);
        let subscription = cx.subscribe_in(&picker, window, Self::on_template);
        self.set_stage(Stage::TemplatePicker(picker), subscription, cx);
    }

    fn on_template(
        &mut self,
        _: &Entity<TemplatePickerView>,
        event: &TemplatePickerEvent,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(loaded) = self.loaded.as_ref() else {
            return self.failed("on_template", "no config loaded", window, cx);
        };
        let templates = YamlTemplates::new(&loaded.paths);
        let key = match event {
            TemplatePickerEvent::Cancelled => return self.finished(None, window, cx),
            TemplatePickerEvent::Create => return self.open_flow_builder(None, window, cx),
            TemplatePickerEvent::EditWithAi(key) => {
                return match load_template_key(&templates, key) {
                    Ok(base) => self.open_flow_builder(Some((base, key.clone())), window, cx),
                    Err(e) => self.failed("on_template", &e, window, cx),
                };
            }
            TemplatePickerEvent::Picked(key) => key,
        };
        match load_template_key(&templates, key) {
            Ok(template) if self.flow() == Some(Flow::Record) => {
                self.start_record(template, window, cx)
            }
            Ok(template) => self.start_transcribe(&template, window, cx),
            Err(e) => self.failed("on_template", &e, window, cx),
        }
    }

    fn builder(
        &mut self,
        base: Option<(SessionTemplate, String)>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Result<Entity<BuilderView>, String> {
        let paths = self.load()?.paths;
        let runtime = TemplateBuilderRuntime::new(
            config_setup(&paths, None),
            YamlTemplates::new(&paths),
            base,
        );
        Ok(cx.new(|cx| BuilderView::new(runtime, window, cx)))
    }

    /// The builder inside a flow: it goes back to the flow's template picker.
    fn open_flow_builder(
        &mut self,
        base: Option<(SessionTemplate, String)>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let builder = match self.builder(base, window, cx) {
            Ok(builder) => builder,
            Err(e) => return self.failed("open_template_builder", &e, window, cx),
        };
        let subscription = cx.subscribe_in(&builder, window, |shell, _, event, window, cx| {
            let BuilderEvent::Closed = event;
            shell.open_picker(window, cx);
        });
        self.set_stage(Stage::Builder(builder), subscription, cx);
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
        self.set_stage(Stage::Preflight(preflight), subscription, cx);
    }

    fn on_preflight(
        &mut self,
        view: &Entity<PreflightView>,
        event: &PreflightEvent,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if let PreflightEvent::Back = event {
            return self.finished(None, window, cx);
        }
        let Some(outcome) = view.update(cx, |view, _| view.take_outcome()) else {
            return self.finished(None, window, cx);
        };
        let Some(paths) = self.loaded.as_ref().map(|loaded| loaded.paths.clone()) else {
            return self.failed("after_preflight", "no config loaded", window, cx);
        };
        match record_start(outcome, &paths, None, false) {
            Ok(start) => {
                self.logging = true;
                self.awake = Some(keep_awake());
                let session = cx.new(|cx| SessionView::record(start, window, cx));
                self.show_session(session, window, cx);
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
        if let Some(live) = self.live.as_mut() {
            live.stage = Stage::Checking;
            live._subscription = None;
        }
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
                self.show_session(session, window, cx);
            }
            Err(e) => self.failed("start_transcribe", &e, window, cx),
        }
    }

    /// The session runs: the sidebar lists its folder, and follows a rename.
    fn show_session(
        &mut self,
        session: Entity<SessionView>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.live_dir = Some(session.read(cx).output_dir().to_path_buf());
        self.refresh_sessions();
        self._live_observer = Some(cx.observe(&session, |shell, session, cx| {
            let dir = session.read(cx).output_dir().to_path_buf();
            if shell.live_dir.as_ref() != Some(&dir) {
                shell.live_dir = Some(dir);
                shell.refresh_sessions();
                cx.notify();
            }
        }));
        let subscription = cx.subscribe_in(&session, window, Self::on_session);
        self.set_stage(Stage::Session(session), subscription, cx);
    }

    fn on_session(
        &mut self,
        _: &Entity<SessionView>,
        event: &SessionEvent,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        match event {
            SessionEvent::Closed => {
                let dir = self.live_dir.clone();
                self.finished(dir, window, cx);
            }
            SessionEvent::Exit => {
                self.finished(None, window, cx);
                cx.quit();
            }
        }
    }

    fn open_saved(&mut self, dir: &Path, window: &mut Window, cx: &mut Context<Self>) {
        if self.live_dir.as_deref() == Some(dir) {
            return self.show_detail(Detail::Live, None, cx);
        }
        let reader = cx.new(|cx| SessionReader::new(dir.to_path_buf(), window, cx));
        let subscription = cx.subscribe_in(&reader, window, |shell, _, event, _, cx| {
            let ReaderEvent::Closed = event;
            shell.show_detail(Detail::Empty, None, cx);
        });
        self.show_detail(
            Detail::Saved(reader, dir.to_path_buf()),
            Some(subscription),
            cx,
        );
    }

    fn open_templates(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let loaded = match self.load() {
            Ok(loaded) => loaded,
            Err(e) => return self.failed("open_templates", &e, window, cx),
        };
        let picker = cx.new(|cx| TemplatePickerView::new(&loaded, window, cx));
        let subscription = cx.subscribe_in(&picker, window, move |shell, _, event, window, cx| {
            let base = match event {
                TemplatePickerEvent::Cancelled => {
                    return shell.show_detail(Detail::Empty, None, cx);
                }
                // The page browses; a session starts from the sidebar.
                TemplatePickerEvent::Picked(_) => return,
                TemplatePickerEvent::Create => None,
                TemplatePickerEvent::EditWithAi(key) => {
                    match shell.load().and_then(|loaded| {
                        load_template_key(&YamlTemplates::new(&loaded.paths), key)
                    }) {
                        Ok(base) => Some((base, key.clone())),
                        Err(e) => return shell.failed("open_templates", &e, window, cx),
                    }
                }
            };
            shell.open_page_builder(base, window, cx);
        });
        self.show_detail(Detail::Templates(picker), Some(subscription), cx);
    }

    /// The builder from the Templates page: it goes back to that page.
    fn open_page_builder(
        &mut self,
        base: Option<(SessionTemplate, String)>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let builder = match self.builder(base, window, cx) {
            Ok(builder) => builder,
            Err(e) => return self.failed("open_template_builder", &e, window, cx),
        };
        let subscription = cx.subscribe_in(&builder, window, |shell, _, event, window, cx| {
            let BuilderEvent::Closed = event;
            shell.open_templates(window, cx);
        });
        self.show_detail(Detail::Builder(builder), Some(subscription), cx);
    }

    fn open_settings(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if matches!(self.detail, Detail::Settings(_)) {
            return;
        }
        let Some(paths) = Paths::for_user() else {
            return self.failed("open_config", "no home folder", window, cx);
        };
        let settings = cx.new(|cx| SettingsView::new(paths, None, window, cx));
        let subscription = cx.subscribe_in(&settings, window, |shell, _, event, _, cx| {
            let SettingsEvent::Closed = event;
            shell.show_detail(Detail::Empty, None, cx);
        });
        self.show_detail(Detail::Settings(settings), Some(subscription), cx);
    }

    /// The window may close now. Otherwise the live session starts its way
    /// out (stop the recording, wait for the audio flush) and closes the app
    /// after. What the pane shows does not matter.
    fn should_close(&mut self, window: &mut Window, cx: &mut Context<Self>) -> bool {
        match self.live.as_ref().map(|live| &live.stage) {
            Some(Stage::Session(session)) => {
                let session = session.clone();
                self.show_detail(Detail::Live, None, cx);
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

/// What a sidebar click runs.
type OnClick = Rc<dyn Fn(&mut Shell, &mut Window, &mut Context<Shell>)>;

/// How one sidebar row looks.
struct NavRow {
    id: ElementId,
    icon: Option<IconName>,
    title: SharedString,
    detail: Option<SharedString>,
    selected: bool,
    /// The live session's mark.
    dot: Option<Hsla>,
}

fn nav(id: impl Into<ElementId>, title: impl Into<SharedString>) -> NavRow {
    NavRow {
        id: id.into(),
        icon: None,
        title: title.into(),
        detail: None,
        selected: false,
        dot: None,
    }
}

impl Shell {
    /// One sidebar row: an icon or a dot, the title, and a muted detail.
    fn nav_row(&self, row: NavRow, on_click: OnClick, cx: &mut Context<Self>) -> Stateful<Div> {
        let NavRow {
            id,
            icon,
            title,
            detail,
            selected,
            dot,
        } = row;
        let theme = cx.theme();
        // As the Finder sidebar: a soft grey under the selected row, the text unchanged.
        let (text, muted) = (theme.sidebar_foreground, theme.muted_foreground);
        let hover = theme.sidebar_foreground.opacity(0.05);
        let selected_bg = theme.sidebar_foreground.opacity(0.1);
        div()
            .id(id)
            .flex()
            .items_center()
            .gap_2()
            .mx_2()
            .px_2()
            .py(px(5.))
            .rounded(px(6.))
            .text_sm()
            .text_color(text)
            .when(selected, |row| row.bg(selected_bg))
            .when(!selected, |row| row.hover(|row| row.bg(hover)))
            .when_some(icon, |row, icon| {
                row.child(Icon::new(icon).size_4().text_color(theme.primary))
            })
            .when_some(dot, |row, dot| {
                row.child(div().flex_none().size(px(8.)).rounded_full().bg(dot))
            })
            .child(
                div()
                    .flex()
                    .flex_col()
                    .min_w_0()
                    .flex_1()
                    .child(div().truncate().child(title))
                    .when_some(detail, |column, detail| {
                        column.child(div().text_xs().text_color(muted).truncate().child(detail))
                    }),
            )
            .on_click(cx.listener(move |shell, _, window, cx| on_click(shell, window, cx)))
    }

    fn session_rows(&self, cx: &mut Context<Self>) -> Vec<AnyElement> {
        let mut sessions = self.sessions.clone();
        // A session that just started may have no file the list counts yet.
        if let Some(dir) = &self.live_dir
            && !sessions.iter().any(|s| &s.dir == dir)
            && let Some(name) = dir.file_name()
        {
            sessions.insert(
                0,
                SessionInfo {
                    dir: dir.clone(),
                    name: name.to_string_lossy().into_owned(),
                    has_notes: false,
                    has_transcript: false,
                },
            );
        }
        let today = chrono::Local::now().date_naive();
        let live_dir = self.live_dir.clone();
        let recording = match self.live.as_ref().map(|live| &live.stage) {
            Some(Stage::Session(session)) => session.read(cx).is_recording(),
            _ => false,
        };
        let shown = match &self.detail {
            Detail::Saved(_, dir) => Some(dir.clone()),
            Detail::Live => live_dir.clone(),
            _ => None,
        };
        let mut rows = Vec::new();
        for (group, entries) in grouped(sessions, today) {
            rows.push(
                div()
                    .px_4()
                    .pt_3()
                    .pb_1()
                    .text_xs()
                    .font_weight(FontWeight::SEMIBOLD)
                    .text_color(cx.theme().muted_foreground)
                    .child(group.title())
                    .into_any_element(),
            );
            for Entry {
                info, title, when, ..
            } in entries
            {
                let is_live = live_dir.as_ref() == Some(&info.dir);
                let dot = is_live.then(|| {
                    if recording {
                        cx.theme().danger
                    } else {
                        cx.theme().primary
                    }
                });
                let selected = shown.as_ref() == Some(&info.dir);
                let dir = info.dir.clone();
                let on_click: OnClick =
                    Rc::new(move |shell, window, cx| shell.open_saved(&dir, window, cx));
                let id = ElementId::Name(info.dir.to_string_lossy().into_owned().into());
                let row = NavRow {
                    detail: Some(when.into()),
                    selected,
                    dot,
                    ..nav(id, title)
                };
                rows.push(self.nav_row(row, on_click, cx).into_any_element());
            }
        }
        if rows.is_empty() {
            let note = self
                .list_error
                .clone()
                .unwrap_or_else(|| "No saved sessions yet.".into());
            rows.push(
                div()
                    .px_4()
                    .pt_3()
                    .text_xs()
                    .text_color(cx.theme().muted_foreground)
                    .child(note)
                    .into_any_element(),
            );
        }
        rows
    }

    fn sidebar(&self, cx: &mut Context<Self>) -> Div {
        let flow = self.flow();
        let on_live = matches!(self.detail, Detail::Live);
        let record: OnClick = Rc::new(|shell, window, cx| shell.start(Flow::Record, window, cx));
        let transcribe: OnClick =
            Rc::new(|shell, window, cx| shell.start(Flow::Transcribe, window, cx));
        let templates: OnClick = Rc::new(|shell, window, cx| shell.open_templates(window, cx));
        let settings: OnClick = Rc::new(|shell, window, cx| shell.open_settings(window, cx));
        let on_templates = matches!(self.detail, Detail::Templates(_) | Detail::Builder(_));
        let on_settings = matches!(self.detail, Detail::Settings(_));
        let theme = cx.theme();
        let (background, border) = (theme.sidebar, theme.sidebar_border);
        div()
            .flex()
            .flex_col()
            .flex_none()
            .w(SIDEBAR_WIDTH)
            .h_full()
            .bg(background)
            .border_r_1()
            .border_color(border)
            .child(title_strip("sidebar-title", &self.sidebar_drag).w_full())
            .child(self.nav_row(
                NavRow {
                    icon: Some(IconName::Mic),
                    selected: on_live && flow == Some(Flow::Record),
                    ..nav("new-recording", "New Recording")
                },
                record,
                cx,
            ))
            .child(self.nav_row(
                NavRow {
                    icon: Some(IconName::File),
                    selected: on_live && flow == Some(Flow::Transcribe),
                    ..nav("transcribe-file", "Transcribe File…")
                },
                transcribe,
                cx,
            ))
            .child(
                div()
                    .id("sessions")
                    .flex_1()
                    .min_h_0()
                    .overflow_y_scroll()
                    .pb_2()
                    .children(self.session_rows(cx)),
            )
            .child(
                div()
                    .flex()
                    .flex_col()
                    .gap_1()
                    .py_2()
                    .border_t_1()
                    .border_color(border)
                    .child(self.nav_row(
                        NavRow {
                            icon: Some(IconName::BookOpen),
                            selected: on_templates,
                            ..nav("templates", "Templates")
                        },
                        templates,
                        cx,
                    ))
                    .child(self.nav_row(
                        NavRow {
                            icon: Some(IconName::Settings),
                            selected: on_settings,
                            ..nav("settings", "Settings")
                        },
                        settings,
                        cx,
                    )),
            )
    }

    fn empty_state(&self, cx: &Context<Self>) -> AnyElement {
        div()
            .size_full()
            .flex()
            .flex_col()
            .items_center()
            .justify_center()
            .gap_4()
            .child(mascot(px(6.)))
            .child(brand(px(22.)))
            .child(
                div()
                    .text_sm()
                    .text_color(cx.theme().muted_foreground)
                    .child("Start a recording with ⌘N, or transcribe a file with ⌘O."),
            )
            .into_any_element()
    }

    /// The pane's title and its body.
    fn pane(&self, cx: &Context<Self>) -> (SharedString, AnyElement) {
        let muted = |text: &'static str| {
            div()
                .size_full()
                .flex()
                .items_center()
                .justify_center()
                .text_color(cx.theme().muted_foreground)
                .child(text)
                .into_any_element()
        };
        match &self.detail {
            Detail::Empty => ("".into(), self.empty_state(cx)),
            Detail::Live => match self.live.as_ref().map(|live| (&live.stage, live.flow)) {
                Some((Stage::TemplatePicker(view), _)) => {
                    ("Choose a Template".into(), view.clone().into_any_element())
                }
                Some((Stage::Builder(view), _)) => {
                    ("New Template".into(), view.clone().into_any_element())
                }
                Some((Stage::Preflight(view), _)) => (
                    "Checking Your Setup".into(),
                    view.clone().into_any_element(),
                ),
                Some((Stage::Checking, Flow::Transcribe)) => {
                    ("Transcribe File".into(), muted("Checking the AI provider…"))
                }
                Some((Stage::Checking, Flow::Record)) => ("New Recording".into(), muted("")),
                Some((Stage::Session(view), flow)) => {
                    let title = match flow {
                        Flow::Record => "Recording",
                        Flow::Transcribe => "Transcription",
                    };
                    (title.into(), view.clone().into_any_element())
                }
                None => ("".into(), self.empty_state(cx)),
            },
            Detail::Saved(view, dir) => {
                let name = dir
                    .file_name()
                    .map(|name| name.to_string_lossy().into_owned())
                    .unwrap_or_default();
                (title(&name).into(), view.clone().into_any_element())
            }
            Detail::Templates(view) => ("Templates".into(), view.clone().into_any_element()),
            Detail::Builder(view) => ("Template Builder".into(), view.clone().into_any_element()),
            Detail::Settings(view) => ("Settings".into(), view.clone().into_any_element()),
        }
    }
}

impl Render for Shell {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let (title, body) = self.pane(cx);
        let theme = cx.theme();
        let (background, foreground) = (theme.background, theme.foreground);
        div()
            .size_full()
            .flex()
            .bg(background)
            .text_color(foreground)
            .child(self.sidebar(cx))
            .child(
                div()
                    .flex()
                    .flex_col()
                    .flex_1()
                    .min_w_0()
                    .h_full()
                    .child(
                        title_strip("pane-title", &self.pane_drag)
                            .px_4()
                            .font_weight(FontWeight::SEMIBOLD)
                            .child(div().truncate().child(title)),
                    )
                    .child(div().flex_1().min_h_0().child(body)),
            )
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

/// Run `f` on the shell, for a menu or key command.
fn with_shell(cx: &mut App, f: impl FnOnce(&mut Shell, &mut Window, &mut Context<Shell>)) {
    let Some(ShellHandle(shell, handle)) = cx.try_global::<ShellHandle>() else {
        return;
    };
    let (shell, handle) = (shell.clone(), *handle);
    let _ = handle.update(cx, |_, window, cx| {
        let _ = shell.update(cx, |shell, cx| f(shell, window, cx));
    });
}

pub fn new_recording(cx: &mut App) {
    with_shell(cx, |shell, window, cx| {
        shell.start(Flow::Record, window, cx)
    });
}

pub fn transcribe_file(cx: &mut App) {
    with_shell(cx, |shell, window, cx| {
        shell.start(Flow::Transcribe, window, cx)
    });
}

pub fn open_settings(cx: &mut App) {
    with_shell(cx, |shell, window, cx| shell.open_settings(window, cx));
}

/// Cmd-Q and Cmd-W: the same way out as the close button.
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
