//! Settings (Python's `ConfigApp`): a page list at the left, one page at a time.
//! The flow lives in `app::settings::SettingsRuntime`; this screen draws the
//! pages, routes keys and clicks, and shows the runtime's state.

mod dialogs;
mod form;
mod models;
mod pages;
mod preview;

pub use dialogs::DISCARD_BINDINGS;

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use crossterm::event::KeyEvent;
use indexmap::IndexMap;
use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::Line;
use ratatui::widgets::{Block, BorderType, Borders, Paragraph};
use tracing::error;

use self::dialogs::{Answer, Dialog};
use self::form::{FormEvent, Kind, Page};
use self::models::{LanguageEditor, ModelChooser, SpeakerState};
use crate::adapters::assets::store::CachedModelStore;
use crate::adapters::paths::Paths;
use crate::app::session::Severity;
use crate::app::settings::form::{FormValues, opening_values, python_str};
use crate::app::settings::{Edited, SPEAKER, SettingsIo, SettingsRuntime, Step, open_privacy};
use crate::app::tasks::Tasks;
use crate::tui::host::{Action, Ctx, MouseInput, MouseKind, Msg, Payload, Screen};
use crate::tui::keys::{Binding, SCREEN_BASE, bind};
use crate::tui::screens::template_picker::resolve_editor;
use crate::tui::theme::Theme;
use crate::tui::widgets::list::ListCursor;
use crate::tui::widgets::status_bar::rms_to_char;

pub const SETTINGS_BINDINGS: [Binding; 6] = [
    SCREEN_BASE[0],
    SCREEN_BASE[1],
    SCREEN_BASE[2],
    SCREEN_BASE[3],
    bind("escape", "quit_app", "Quit").priority(),
    bind("ctrl+s", "save_config", "Save").priority(),
];

const NAV: [&str; 5] = ["Models", "Digest", "Audio", "Sessions", "Appearance"];
const NAV_WIDTH: u16 = 20;
/// The Appearance page is a plain column; the others scroll and take the focus.
const APPEARANCE: usize = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Focus {
    Nav,
    /// The page's scroll container.
    Container,
    Page,
}

/// The pages, read by Python's widget id.
struct Fields<'a>(&'a mut [Page; 5]);

impl Fields<'_> {
    /// The page that holds `id`.
    fn page_of(&mut self, id: &str) -> &mut Page {
        let index = self
            .0
            .iter()
            .position(|page| page.item(id).is_some())
            .unwrap_or(0);
        &mut self.0[index]
    }
}

impl FormValues for Fields<'_> {
    fn input(&mut self, id: &str) -> String {
        self.page_of(id).input(id)
    }

    fn switch(&mut self, id: &str) -> bool {
        self.page_of(id).switch(id)
    }

    fn select(&mut self, id: &str) -> String {
        self.page_of(id).select(id)
    }

    fn text_area(&mut self, id: &str) -> String {
        self.page_of(id).text_area(id)
    }

    fn language_models(&mut self) -> IndexMap<String, String> {
        self.0[0]
            .custom::<LanguageEditor>("models-language")
            .map(|editor| editor.values())
            .unwrap_or_default()
    }

    fn chosen_model_uri(&mut self) -> String {
        self.0[0]
            .custom::<ModelChooser>("models-chooser")
            .map(|chooser| chooser.selected_uri())
            .unwrap_or_default()
    }
}

/// The runtime's notices become toasts; its tasks report to this screen.
impl SettingsIo for Ctx {
    fn notify(&mut self, text: String, severity: Severity, timeout: Duration) {
        self.notify_with(text, severity, timeout, "");
    }

    fn tasks(&self) -> Tasks {
        Ctx::tasks(self)
    }
}

pub struct SettingsScreen {
    runtime: SettingsRuntime,
    pages: [Page; 5],
    nav: ListCursor,
    focus: Focus,
    /// The probe state the Audio page buttons show.
    probe_shown: bool,
    discarding: bool,
}

fn build_pages(runtime: &SettingsRuntime, devices: &[String]) -> [Page; 5] {
    let values = opening_values(&runtime.loaded, devices);
    let store = runtime.store();
    let source = pages::Source {
        values: &values,
        catalog: store.catalog(),
        cache_root: store.cache_root(),
    };
    [
        pages::models_page(&source),
        pages::digest_page(&values),
        pages::audio_page(&values),
        pages::sessions_page(&values),
        pages::appearance_page(&values),
    ]
}

impl SettingsScreen {
    pub fn new(paths: Paths, config_path: Option<PathBuf>) -> Self {
        let store = Arc::new(CachedModelStore::new(&paths));
        let devices = crate::adapters::audio::microphone::list_input_devices();
        Self::with(paths, config_path, store, &devices)
    }

    /// Build with a given store and device list, for tests.
    fn with(
        paths: Paths,
        config_path: Option<PathBuf>,
        store: Arc<CachedModelStore>,
        devices: &[String],
    ) -> Self {
        let runtime = SettingsRuntime::new(paths, config_path, store);
        let pages = build_pages(&runtime, devices);
        let mut nav = ListCursor::default();
        nav.reset(NAV.len());
        nav.select(0);
        Self {
            runtime,
            pages,
            nav,
            focus: Focus::Nav,
            probe_shown: false,
            discarding: false,
        }
    }

    fn page_index(&self) -> usize {
        self.nav.index.unwrap_or(0)
    }

    fn page(&mut self) -> &mut Page {
        let index = self.page_index();
        &mut self.pages[index]
    }

    #[cfg(test)]
    fn collect(&mut self) -> yaml_serde::Value {
        self.runtime.collect(&mut Fields(&mut self.pages))
    }

    fn has_unsaved_changes(&mut self) -> bool {
        self.runtime
            .has_unsaved_changes(&mut Fields(&mut self.pages))
    }

    fn save(&mut self, ctx: &mut Ctx) {
        let fields = &mut Fields(&mut self.pages);
        if let Some(has_key) = self.runtime.save(fields, ctx) {
            self.pages[0].show("clear-key-row", has_key);
        }
    }

    // --- Leave ----------------------------------------------------------

    fn quit(&mut self) -> Action {
        let index = self.page_index();
        if self.pages[index].expanded() {
            // Esc closes an open dropdown, as it does everywhere else.
            self.pages[index].close_selects();
            return Action::None;
        }
        if self.focus != Focus::Nav {
            // Inside a page, Esc goes back to the page list; from the list, it leaves.
            self.pages[index].set_focus(None);
            self.focus = Focus::Nav;
            return Action::None;
        }
        if self.has_unsaved_changes() {
            self.discarding = true;
            return Action::Push(Box::new(Dialog::discard()));
        }
        self.leave()
    }

    fn leave(&mut self) -> Action {
        let saved = self.runtime.leave();
        Action::Pop(Some(Box::new(SettingsClosed { saved })))
    }

    // --- Focus ----------------------------------------------------------

    fn focus_next(&mut self, forward: bool) {
        let index = self.page_index();
        let has_container = index != APPEARANCE;
        match (self.focus, forward) {
            (Focus::Nav, true) if has_container => self.focus = Focus::Container,
            (Focus::Nav, true) | (Focus::Container, true) => self.enter_page(true),
            (Focus::Nav, false) => self.enter_page(false),
            (Focus::Container, false) => self.focus = Focus::Nav,
            (Focus::Page, _) => {
                if self.pages[index].step_focus(forward).is_none() {
                    self.focus = match (forward, has_container) {
                        (false, true) => Focus::Container,
                        _ => Focus::Nav,
                    };
                }
            }
        }
    }

    fn enter_page(&mut self, first: bool) {
        let page = self.page();
        let stop = if first {
            page.first_stop()
        } else {
            page.last_stop()
        };
        page.set_focus(stop);
        self.focus = match stop {
            Some(_) => Focus::Page,
            None => Focus::Nav,
        };
    }

    // --- Page events ----------------------------------------------------

    fn on_event(&mut self, event: FormEvent, ctx: &mut Ctx) -> Action {
        match event {
            FormEvent::Changed("cfg-theme", value) => ctx.set_theme(&value),
            FormEvent::Changed(..) => {}
            FormEvent::Custom("model-chosen", _) => {
                // A custom path wins over the chosen row, so choosing a row clears it.
                self.pages[0].set_input("cfg-trans-custom-path", "");
            }
            FormEvent::Custom(..) => {}
            FormEvent::Pressed(id) => return self.pressed(&id, ctx),
        }
        Action::None
    }

    fn pressed(&mut self, id: &str, ctx: &mut Ctx) -> Action {
        let runtime = &mut self.runtime;
        let fields = &mut Fields(&mut self.pages);
        match id {
            "cfg-openai-clear-key" => {
                fields.0[0].set_input("cfg-openai-api-key", "");
                runtime.clear_key(ctx);
            }
            "cfg-ai-test" => runtime.test_ai(fields, ctx),
            "cfg-ollama-refresh" => runtime.refresh_models(fields, ctx),
            "cfg-ollama-pull-digest" => runtime.pull_model("cfg-digest-model", fields, ctx),
            "cfg-ollama-pull-interactive" => {
                runtime.pull_model("cfg-interactive-model", fields, ctx)
            }
            "models-download-all" => runtime.download_all_missing(ctx),
            "models-open-cache" => {
                if let Err(e) = crate::tui::system::open_folder(runtime.store().cache_root()) {
                    error!(target: "ltn.config", "Cannot open the cache folder: {e}");
                }
            }
            "speaker-install" => runtime.install_speaker(ctx),
            "speaker-remove" => return self.confirm_remove(SPEAKER, ctx),
            "speaker-cancel" => runtime.cancel_speaker(),
            "audio-start" => runtime.start_probe(fields, ctx),
            "audio-stop" => runtime.stop_probe(),
            "audio-grant-system" => open_privacy("Privacy_ScreenCapture"),
            "audio-grant-mic" => open_privacy("Privacy_Microphone"),
            other => {
                if let Some(asset_id) = other.strip_prefix("download-") {
                    runtime.start_download(asset_id, ctx);
                } else if let Some(asset_id) = other.strip_prefix("remove-") {
                    return self.confirm_remove(asset_id, ctx);
                } else if let Some(asset_id) = other.strip_prefix("cancel-") {
                    runtime.cancel_download(asset_id);
                }
            }
        }
        self.show_runtime();
        Action::None
    }

    fn confirm_remove(&mut self, asset_id: &str, ctx: &mut Ctx) -> Action {
        match self.runtime.ask_remove(asset_id, ctx) {
            Some((label, estimate)) => Action::Push(Box::new(Dialog::remove(&label, &estimate))),
            None => Action::None,
        }
    }

    fn test_ai(&mut self, ctx: &mut Ctx) {
        self.runtime.test_ai(&mut Fields(&mut self.pages), ctx);
    }

    // --- Raw editor -----------------------------------------------------

    fn open_raw_editor(&mut self, ctx: &mut Ctx) {
        let Some(editor) = resolve_editor() else {
            ctx.error(
                "No editor found ($VISUAL / $EDITOR)",
                crate::tui::host::NOTIFICATION_TIMEOUT,
            );
            return;
        };
        let Some(path) = self.runtime.prepare_raw_edit(ctx) else {
            return;
        };
        ctx.suspend(move || {
            let status = std::process::Command::new(&editor[0])
                .args(&editor[1..])
                .arg(&path)
                .status();
            Edited(status.map(|_| ()).map_err(|e| {
                // $EDITOR can name a program that is not installed.
                error!(target: "ltn.config", "Cannot open editor {editor:?}: {e}");
                e.to_string()
            }))
        });
    }

    /// Push the reloaded values back into the fields, as `_repopulate_fields`.
    /// The custom model path, the buttons and the status lines stay as they are.
    fn repopulate(&mut self, ctx: &mut Ctx) {
        let devices = crate::adapters::audio::microphone::list_input_devices();
        let fresh = build_pages(&self.runtime, &devices);
        for (page, new) in self.pages.iter_mut().zip(fresh) {
            for item in new.items {
                let replace = match &item.kind {
                    Kind::Field { .. }
                    | Kind::Switch { .. }
                    | Kind::Select { .. }
                    | Kind::TextArea(_) => true,
                    Kind::Custom(_) => item.id == "models-language",
                    _ => false,
                };
                if !replace || item.id.is_empty() {
                    continue;
                }
                if let Some(old) = page.item_mut(item.id) {
                    old.kind = item.kind;
                }
            }
        }
        let model = python_str(
            self.runtime
                .loaded
                .raw
                .get("transcription")
                .and_then(|t| t.get("model")),
        );
        if let Some(chooser) = self.pages[0].custom::<ModelChooser>("models-chooser") {
            chooser.select_by_uri(&model);
        }
        // Setting the theme select applies the theme, as its Changed event does in Python.
        let theme = self.runtime.theme().to_string();
        if ctx.theme.name != theme {
            ctx.set_theme(&theme);
        }
    }

    // --- The runtime's state --------------------------------------------

    /// Copy the runtime's model rows, speaker row, AI model line and probe
    /// state into the pages.
    fn show_runtime(&mut self) {
        let runtime = &self.runtime;
        if let Some(chooser) = self.pages[0].custom::<ModelChooser>("models-chooser") {
            chooser.show_rows(&runtime.rows);
        }
        if let Some(row) = self.pages[0].custom::<SpeakerState>("speaker-state") {
            row.show(&runtime.speaker);
        }
        if let Some(text) = &runtime.models_text {
            self.pages[0].set_text("cfg-ollama-models", text);
        }
        let silence = runtime.silence_threshold();
        let ids = [
            ("audio-meter-mic", "audio-perm-mic"),
            ("audio-meter-system", "audio-perm-system"),
        ];
        for (meter, (meter_id, permission_id)) in runtime.meters.iter().zip(ids) {
            if meter.live {
                let text: String = meter
                    .levels
                    .iter()
                    .map(|level| rms_to_char(*level, silence))
                    .collect();
                self.pages[2].set_text(meter_id, &text);
            }
            if let Some(permission) = meter.permission
                && let Some(item) = self.pages[2].item_mut(permission_id)
            {
                item.kind = Kind::Text(permission.text.into(), permission.color);
            }
        }
        let running = runtime.probe_running();
        if running != self.probe_shown {
            self.probe_shown = running;
            self.set_probe_running(running);
        }
    }

    fn set_probe_running(&mut self, running: bool) {
        let page = &mut self.pages[2];
        let focused_button = page
            .focus
            .and_then(|(index, _)| (page.items[index].id == "probe-buttons").then_some(()));
        if let Some(button) = page.button_mut("audio-start") {
            button.disabled = running;
        }
        if let Some(button) = page.button_mut("audio-stop") {
            button.disabled = !running;
        }
        // A disabled button drops focus. The other one takes it, so Enter works again.
        if focused_button.is_some() {
            let index = page
                .items
                .iter()
                .position(|item| item.id == "probe-buttons");
            page.focus = index.map(|index| (index, 0));
        }
    }

    fn on_result(&mut self, payload: crate::app::tasks::Payload, ctx: &mut Ctx) -> Option<Msg> {
        let fields = &mut Fields(&mut self.pages);
        match self.runtime.task_result(payload, fields, ctx) {
            Ok(Step::None) => {}
            Ok(Step::Reloaded) => {
                self.repopulate(ctx);
                self.runtime.mark_saved_state(&mut Fields(&mut self.pages));
            }
            Err(payload) => return Some(Msg::Task(payload)),
        }
        self.show_runtime();
        None
    }

    // --- Drawing --------------------------------------------------------

    /// A click inside the page box.
    fn click_page(&mut self, input: &MouseInput, ctx: &mut Ctx) -> Action {
        let index = self.page_index();
        let page = &mut self.pages[index];
        if !page.takes_click(input.column, input.row) {
            // Empty page space: the scroll container takes the focus.
            if index != APPEARANCE {
                page.set_focus(None);
                self.focus = Focus::Container;
            }
            return Action::None;
        }
        let event = page.click(input.column, input.row);
        if page.focus.is_some() {
            self.focus = Focus::Page;
        }
        match event {
            Some(event) => self.on_event(event, ctx),
            None => Action::None,
        }
    }

    /// The page list in a rounded box titled "Settings".
    fn draw_nav(&mut self, frame: &mut Frame, area: Rect, theme: &Theme, ctx: &Ctx) {
        let focused = self.focus == Focus::Nav;
        let background = theme.color("background");
        let inner = crate::tui::widgets::rounded(
            frame,
            area,
            "Settings",
            crate::tui::widgets::border_color(focused, theme),
            background,
            theme,
        );
        let rows: Vec<Line> = NAV.iter().map(|label| Line::from(*label)).collect();
        self.nav
            .render(frame, inner, &rows, focused, background, theme);
        ctx.hit(inner, "nav", 0);
    }

    /// The page in a rounded box: its name at the left of the top border, the
    /// config file at the right. Returns the inside.
    fn draw_page_box(&self, frame: &mut Frame, area: Rect, theme: &Theme) -> Rect {
        let focused = matches!(self.focus, Focus::Container | Focus::Page);
        let border = crate::tui::widgets::border_color(focused, theme);
        let background = theme.color("background");
        let name = NAV[self.page_index()];
        // The path gives way to the name: it loses its start first.
        let room = usize::from(area.width).saturating_sub(name.len() + 10);
        let path = tail(&self.runtime.config_file().display().to_string(), room);
        let block = Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .style(Style::default().bg(background))
            .border_style(Style::default().fg(border).bg(background))
            .title(Line::from(vec![
                ratatui::text::Span::styled("─", Style::default().fg(border)),
                ratatui::text::Span::styled(
                    format!(" {name} "),
                    Style::default()
                        .fg(theme.color("foreground"))
                        .add_modifier(Modifier::BOLD),
                ),
            ]))
            .title(
                Line::styled(
                    format!(" {path} "),
                    Style::default().fg(theme.on("text-muted", "background")),
                )
                .right_aligned(),
            );
        let inner = block.inner(area);
        frame.render_widget(block, area);
        inner
    }
}

/// The end of `text` in `width` columns, with `…` in front when it is cut.
fn tail(text: &str, width: usize) -> String {
    if crate::tui::text::str_width(text) <= width {
        return text.to_string();
    }
    let mut kept: Vec<char> = Vec::new();
    let mut used = 1;
    for character in text.chars().rev() {
        let cell = crate::tui::text::str_width(&character.to_string());
        if used + cell > width {
            break;
        }
        used += cell;
        kept.push(character);
    }
    let end: String = kept.into_iter().rev().collect();
    match width {
        0 => String::new(),
        _ => format!("…{end}"),
    }
}

/// A one-row bar with `padding: 0 1`, the text at the left.
fn left_bar(
    frame: &mut Frame,
    area: Rect,
    text: &str,
    background: &str,
    foreground: &str,
    bold: bool,
    theme: &Theme,
) {
    let mut style = Style::default()
        .bg(theme.color(background))
        .fg(theme.on(foreground, background));
    if bold {
        style = style.add_modifier(Modifier::BOLD);
    }
    frame.render_widget(Paragraph::new(format!(" {text}")).style(style), area);
}

/// What Settings popped with: true when it saved at least once.
pub struct SettingsClosed {
    pub saved: bool,
}

impl Screen for SettingsScreen {
    fn mounted(&mut self, ctx: &mut Ctx) -> Action {
        ctx.set_theme(self.runtime.theme());
        self.runtime.mounted(
            &mut Fields(&mut self.pages),
            ctx,
            "Press [e] to fix the file.",
        );
        Action::None
    }

    fn render(&mut self, frame: &mut Frame, area: Rect, ctx: &Ctx) {
        let theme = &ctx.theme;
        if area.height < 4 {
            return;
        }
        left_bar(
            frame,
            Rect::new(area.x, area.bottom() - 1, area.width, 1),
            "[Ctrl+S] Save  [t] Test AI  [e] Edit raw file  [Esc] Back",
            "surface",
            "text-muted",
            false,
            theme,
        );
        let shell = Rect::new(area.x, area.y, area.width, area.height - 1);
        let nav = Rect::new(shell.x, shell.y, NAV_WIDTH.min(shell.width), shell.height);
        self.draw_nav(frame, nav, theme, ctx);
        let page_box = Rect::new(
            shell.x + nav.width,
            shell.y,
            shell.width - nav.width,
            shell.height,
        );
        ctx.hit(page_box, "page", 0);
        let page_area = self.draw_page_box(frame, page_box, theme);
        let index = self.page_index();
        self.pages[index].render(frame, page_area, theme);
    }

    /// A click on a page row opens it; on the page it focuses and presses
    /// what is under the pointer. The wheel scrolls the page.
    fn mouse(&mut self, input: &MouseInput, ctx: &mut Ctx) -> Action {
        let index = self.page_index();
        let open = self.pages[index].expanded();
        match input.kind {
            MouseKind::WheelUp | MouseKind::WheelDown => {
                if open || input.on("page").is_some() {
                    self.pages[index].wheel(input.kind == MouseKind::WheelDown);
                }
                Action::None
            }
            MouseKind::Click if open => self.click_page(input, ctx),
            MouseKind::Click if input.on("nav").is_some() => {
                if let Some(row) = self.nav.item_at(input.row) {
                    self.pages[index].close_selects();
                    self.nav.select(row);
                    self.focus = Focus::Nav;
                }
                Action::None
            }
            MouseKind::Click if input.on("page").is_some() => self.click_page(input, ctx),
            MouseKind::Click => Action::None,
        }
    }

    fn key(&mut self, key: &KeyEvent, name: &str, ctx: &mut Ctx) -> Action {
        match name {
            "escape" => return self.quit(),
            "ctrl+s" => {
                self.save(ctx);
                return Action::None;
            }
            _ => {}
        }
        let index = self.page_index();
        match self.focus {
            Focus::Page => {
                if let Some(event) = self.pages[index].key(key, name) {
                    return match event {
                        Some(event) => self.on_event(event, ctx),
                        None => Action::None,
                    };
                }
            }
            Focus::Nav if matches!(name, "up" | "down" | "home" | "end") => {
                self.nav.key(name);
                return Action::None;
            }
            Focus::Container
                if matches!(name, "up" | "down" | "pageup" | "pagedown" | "home" | "end") =>
            {
                self.pages[index].scroll_key(name);
                return Action::None;
            }
            _ => {}
        }
        match name {
            "tab" => self.focus_next(true),
            "shift+tab" => self.focus_next(false),
            "t" => self.test_ai(ctx),
            "e" => self.open_raw_editor(ctx),
            _ => {}
        }
        Action::None
    }

    fn paste(&mut self, text: &str, _ctx: &mut Ctx) -> Action {
        if self.focus == Focus::Page {
            self.page().paste(text);
        }
        Action::None
    }

    fn message(&mut self, msg: Msg, ctx: &mut Ctx) -> (Option<Msg>, Action) {
        match msg {
            Msg::Task(payload) => (self.on_result(payload, ctx), Action::None),
            other => (Some(other), Action::None),
        }
    }

    fn resumed(&mut self, result: Option<Payload>, ctx: &mut Ctx) -> Action {
        let answer = result
            .and_then(|payload| payload.downcast::<Answer>().ok())
            .map(|answer| *answer);
        let confirmed = answer.as_ref().is_some_and(|answer| answer.confirmed);
        if std::mem::take(&mut self.discarding) {
            return match confirmed {
                true => self.leave(),
                false => Action::None,
            };
        }
        let delete_legacy = answer.is_some_and(|answer| answer.delete_legacy);
        self.runtime.answer_remove(confirmed, delete_legacy, ctx);
        Action::None
    }
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
