//! The template list with its preview: a filter box, the templates by
//! locale and their buttons on the left, the preview on the right. In a flow
//! it is the start panel: a start button under the preview, and the setup
//! checks in place of the preview once they run.

use std::rc::Rc;
use std::time::Duration;

use gpui_kit::component::button::{Button, ButtonVariants as _};
use gpui_kit::component::input::{Input, InputEvent, InputState};
use gpui_kit::component::text::{TextView, TextViewState};
use gpui_kit::component::{ActiveTheme as _, Disableable as _, IconName, Sizable as _};
use gpui_kit::prelude::FluentBuilder as _;
use gpui_kit::*;
use tracing::warn;

use crate::adapters::templates::YamlTemplates;
use crate::app::container::Loaded;
use crate::app::notices::BUILTIN_NOTICE_BODY;
use crate::app::session::Severity;
use crate::app::template_catalog::{NO_TEMPLATES, Row, TemplateCatalog};
use crate::gui::bridge::notify;
use crate::gui::dialogs::{self, Extra};
use crate::gui::icons;

const NOTICE_TIMEOUT: Duration = Duration::from_secs(5);
/// The bars under the list and the preview line up.
pub const BAR_HEIGHT: Pixels = px(52.);

pub enum TemplatePickerEvent {
    Picked(String),
    Create,
    EditWithAi(String),
    Cancelled,
}

pub struct TemplatePickerView {
    catalog: TemplateCatalog,
    rows: Vec<Row>,
    /// The highlighted row; always a template row when one shows.
    selected: Option<usize>,
    search: Entity<InputState>,
    preview: Entity<TextViewState>,
    scroll: ScrollHandle,
    /// The list has the keys; otherwise the filter box has them.
    list_focused: bool,
    /// The button that starts a session from the chosen template, in a flow.
    action: Option<(&'static str, IconName)>,
    /// The setup checks of the chosen template, in place of its preview.
    checks: Option<AnyView>,
    focus: FocusHandle,
    _subscription: Subscription,
}

impl EventEmitter<TemplatePickerEvent> for TemplatePickerView {}

impl TemplatePickerView {
    pub fn new(
        loaded: &Loaded,
        action: Option<(&'static str, IconName)>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Self {
        let catalog = TemplateCatalog::new(
            YamlTemplates::new(&loaded.paths),
            loaded.infra.show_builtin_templates,
        );
        let preview = cx.new(|cx| TextViewState::markdown("", cx));
        let list_focused = catalog.loaded.is_empty();
        let placeholder = format!("Filter {} templates", catalog.loaded.len());
        let search = cx.new(|cx| InputState::new(window, cx).placeholder(placeholder));
        let subscription = cx.subscribe_in(&search, window, Self::on_search);
        let mut view = Self {
            catalog,
            rows: Vec::new(),
            selected: None,
            search,
            preview,
            scroll: ScrollHandle::new(),
            list_focused,
            action,
            checks: None,
            focus: cx.focus_handle(),
            _subscription: subscription,
        };
        view.rebuild("", cx);
        view.focus_part(list_focused, window, cx);
        let noticed = loaded.paths.builtin_templates_noticed();
        if !noticed.exists() {
            cx.defer_in(window, move |_, window, cx| {
                builtin_notice(noticed, window, cx)
            });
        }
        view
    }

    fn focus_part(&mut self, list: bool, window: &mut Window, cx: &mut Context<Self>) {
        self.list_focused = list;
        if list {
            self.focus.focus(window, cx);
        } else {
            let handle = self.search.read(cx).focus_handle(cx);
            handle.focus(window, cx);
        }
        cx.notify();
    }

    /// The checks run for the chosen template; a click on another one ends them.
    pub fn show_checks(&mut self, checks: AnyView, cx: &mut Context<Self>) {
        self.checks = Some(checks);
        cx.notify();
    }

    /// Back to the preview. Dropping the checks view cancels its checks.
    pub fn close_checks(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if self.checks.take().is_some() {
            self.focus_part(true, window, cx);
        }
    }

    fn query(&self, cx: &App) -> String {
        self.search.read(cx).value().trim().to_lowercase()
    }

    fn on_search(
        &mut self,
        _: &Entity<InputState>,
        event: &InputEvent,
        _: &mut Window,
        cx: &mut Context<Self>,
    ) {
        match event {
            InputEvent::Change => {
                let query = self.query(cx);
                self.rebuild(&query, cx);
            }
            InputEvent::Focus => self.list_focused = false,
            InputEvent::Blur | InputEvent::PressEnter { .. } => {}
        }
        cx.notify();
    }

    fn rebuild(&mut self, query: &str, cx: &mut Context<Self>) {
        self.checks = None;
        self.rows = self.catalog.rows(query);
        self.selected = self
            .rows
            .iter()
            .position(|row| matches!(row, Row::Template(_)));
        self.show_preview(cx);
    }

    fn current(&self) -> Option<String> {
        match self.rows.get(self.selected?)? {
            Row::Template(name) => Some(name.clone()),
            Row::Locale(_) => None,
        }
    }

    fn show_preview(&mut self, cx: &mut Context<Self>) {
        let markdown = match self.current() {
            Some(name) => self.catalog.preview_markdown(&name).unwrap_or_default(),
            None if self.catalog.loaded.is_empty() => NO_TEMPLATES.to_string(),
            None => "*No matching templates*".to_string(),
        };
        self.preview
            .update(cx, |preview, cx| preview.set_text(&markdown, cx));
    }

    /// The next template row up or down, wrapping around as the TUI list does.
    fn step(&mut self, down: bool, cx: &mut Context<Self>) {
        let templates: Vec<usize> = (0..self.rows.len())
            .filter(|at| matches!(self.rows[*at], Row::Template(_)))
            .collect();
        let Some(at) = self
            .selected
            .and_then(|selected| templates.iter().position(|at| *at == selected))
        else {
            return;
        };
        let next = match down {
            true => (at + 1) % templates.len(),
            false => (at + templates.len() - 1) % templates.len(),
        };
        self.select(templates[next], cx);
    }

    fn select(&mut self, row: usize, cx: &mut Context<Self>) {
        if self.selected != Some(row) {
            self.checks = None;
        }
        self.selected = Some(row);
        self.scroll.scroll_to_item(row);
        self.show_preview(cx);
        cx.notify();
    }

    /// A key or a button. While the checks show, the keys belong to them.
    fn press(&mut self, name: &str, window: &mut Window, cx: &mut Context<Self>) {
        match name {
            "up" | "down" => self.step(name == "down", cx),
            "enter" => self.pick(cx),
            "escape" => cx.emit(TemplatePickerEvent::Cancelled),
            "tab" => {
                let list = !self.list_focused;
                self.focus_part(list, window, cx);
            }
            _ => {}
        }
    }

    fn pick(&mut self, cx: &mut Context<Self>) {
        if let Some(key) = self.current() {
            cx.emit(TemplatePickerEvent::Picked(key));
        }
    }

    fn edit_with_ai(&mut self, cx: &mut Context<Self>) {
        if let Some(key) = self.current() {
            cx.emit(TemplatePickerEvent::EditWithAi(key));
        }
    }

    /// Keys the picker takes before the filter box: the box would keep the arrows.
    fn on_capture_key(
        &mut self,
        event: &KeyDownEvent,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let keystroke = &event.keystroke;
        let modifiers = keystroke.modifiers;
        if self.checks.is_some() || modifiers.control || modifiers.platform || modifiers.alt {
            return;
        }
        let key = keystroke.key.as_str();
        if matches!(key, "up" | "down" | "enter" | "escape" | "tab") {
            cx.stop_propagation();
            self.press(key, window, cx);
        }
    }

    /// `e`: the user copy of the template in the system's text editor. The
    /// list reads it again when the editor closes it.
    fn edit(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let Some(name) = self.current() else {
            return;
        };
        let path = match self.catalog.templates.ensure_user_copy(&name) {
            Ok(path) => path,
            Err(e) => {
                let text = format!("Cannot copy the template: {e}");
                notify(window, cx, text, Severity::Error, NOTICE_TIMEOUT);
                return;
            }
        };
        let edited = cx.background_executor().spawn(async move {
            // -W waits until the editor closes the file; -t opens the text editor.
            std::process::Command::new("open")
                .args(["-W", "-t"])
                .arg(&path)
                .status()
        });
        cx.spawn_in(window, async move |this, cx| {
            let status = edited.await;
            let _ = this.update_in(cx, |view, window, cx| {
                if let Err(e) = status {
                    warn!(target: "ltn.app", "Cannot run the text editor: {e}");
                }
                view.reload(&name, window, cx);
            });
        })
        .detach();
    }

    fn reload(&mut self, name: &str, window: &mut Window, cx: &mut Context<Self>) {
        if self.catalog.reload(name).is_err() {
            let text = format!("Template \"{name}\" has invalid YAML; changes ignored");
            notify(window, cx, text, Severity::Warning, NOTICE_TIMEOUT);
            return;
        }
        let query = self.query(cx);
        self.rebuild(&query, cx);
        let at = self
            .rows
            .iter()
            .position(|row| matches!(row, Row::Template(key) if key == name));
        if let Some(at) = at {
            self.select(at, cx);
        }
    }

    fn ask_delete(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let Some(name) = self.current() else {
            return;
        };
        if !self.catalog.user_names.contains(&name) {
            let text = "Only user templates can be deleted".to_string();
            notify(window, cx, text, Severity::Warning, NOTICE_TIMEOUT);
            return;
        }
        let view = cx.entity().downgrade();
        let message = format!("Delete user template \"{name}\"?");
        dialogs::confirm(window, cx, message, "Yes", true, move |window, cx| {
            let _ = view.update(cx, |view, cx| view.delete(&name, window, cx));
        });
    }

    fn delete(&mut self, name: &str, window: &mut Window, cx: &mut Context<Self>) {
        if self.catalog.delete(name).is_err() {
            let text = format!("\"{name}\" is not a user template");
            notify(window, cx, text, Severity::Error, NOTICE_TIMEOUT);
            return;
        }
        let query = self.query(cx);
        self.rebuild(&query, cx);
        let text = format!("Deleted user template \"{name}\"");
        notify(window, cx, text, Severity::Information, NOTICE_TIMEOUT);
        cx.notify();
    }

    fn render_row(&self, index: usize, row: &Row, cx: &mut Context<Self>) -> AnyElement {
        let theme = cx.theme();
        let name = match row {
            Row::Locale(locale) => {
                return div()
                    .px_2()
                    .pt_3()
                    .pb_1()
                    .font_weight(FontWeight::SEMIBOLD)
                    .text_xs()
                    .text_color(theme.muted_foreground)
                    .child(locale.to_uppercase())
                    .into_any_element();
            }
            Row::Template(name) => name.clone(),
        };
        let selected = self.selected == Some(index);
        let (background, text, muted) = match (selected, self.list_focused) {
            (true, true) => (
                theme.primary,
                theme.primary_foreground,
                theme.primary_foreground.opacity(0.75),
            ),
            (true, false) => (theme.secondary, theme.foreground, theme.muted_foreground),
            (false, _) => (theme.transparent, theme.foreground, theme.muted_foreground),
        };
        let hover = theme.foreground.opacity(0.06);
        let locale = self
            .catalog
            .loaded
            .get(&name)
            .map(|template| template.metadata.locale.clone())
            .unwrap_or_default();
        let user = self.catalog.user_names.contains(&name);
        div()
            .id(("template", index))
            .flex()
            .items_center()
            .gap_2()
            .px_2()
            .py_1()
            .rounded(px(6.))
            .bg(background)
            .text_color(text)
            .when(!selected, |row| row.hover(move |row| row.bg(hover)))
            .child(
                div()
                    .flex_1()
                    .overflow_hidden()
                    .whitespace_nowrap()
                    .text_ellipsis()
                    .child(self.catalog.label(&name)),
            )
            .when(user, |row| {
                row.child(div().text_xs().text_color(muted).child("User"))
            })
            .child(div().text_xs().text_color(muted).child(locale))
            .on_click(cx.listener(move |view, event: &ClickEvent, window, cx| {
                view.focus_part(true, window, cx);
                view.select(index, cx);
                if event.click_count() > 1 {
                    view.pick(cx);
                }
            }))
            .into_any_element()
    }

    /// New, Edit, Edit with AI and Delete, under the list.
    fn render_tools(&self, cx: &mut Context<Self>) -> Div {
        let current = self.current();
        let none = current.is_none();
        let user = current
            .as_ref()
            .is_some_and(|name| self.catalog.user_names.contains(name));
        div()
            .flex()
            .flex_none()
            .items_center()
            .gap_1()
            .h(BAR_HEIGHT)
            .px_2()
            .border_t_1()
            .border_color(cx.theme().border)
            .child(
                Button::new("new-template")
                    .ghost()
                    .small()
                    .icon(IconName::Plus)
                    .label("New")
                    .tooltip("Create a template with AI")
                    .on_click(cx.listener(|_, _, _, cx| cx.emit(TemplatePickerEvent::Create))),
            )
            .child(
                Button::new("edit-template")
                    .ghost()
                    .small()
                    .icon(icons::extra("pencil"))
                    .label("Edit")
                    .tooltip("Open the template in your text editor")
                    .disabled(none)
                    .on_click(cx.listener(|view, _, window, cx| view.edit(window, cx))),
            )
            .child(
                Button::new("edit-template-ai")
                    .ghost()
                    .small()
                    .icon(icons::extra("sparkles"))
                    .label("Edit with AI")
                    .disabled(none)
                    .on_click(cx.listener(|view, _, _, cx| view.edit_with_ai(cx))),
            )
            .child(div().flex_1())
            .child(
                Button::new("delete-template")
                    .ghost()
                    .small()
                    .icon(icons::extra("trash"))
                    .tooltip(if user {
                        "Delete this template"
                    } else {
                        "Only your own templates can be deleted"
                    })
                    .disabled(!user)
                    .on_click(cx.listener(|view, _, window, cx| view.ask_delete(window, cx))),
            )
    }

    /// Cancel and the start button, under the preview in a flow.
    fn render_action(&self, label: &'static str, icon: IconName, cx: &mut Context<Self>) -> Div {
        div()
            .flex()
            .flex_none()
            .items_center()
            .justify_end()
            .gap_2()
            .h(BAR_HEIGHT)
            .px_4()
            .border_t_1()
            .border_color(cx.theme().border)
            .child(
                Button::new("cancel-start")
                    .label("Cancel")
                    .on_click(cx.listener(|_, _, _, cx| cx.emit(TemplatePickerEvent::Cancelled))),
            )
            .child(
                Button::new("start")
                    .primary()
                    .icon(icon)
                    .label(label)
                    .disabled(self.current().is_none())
                    .on_click(cx.listener(|view, _, _, cx| view.pick(cx))),
            )
    }
}

/// The note about built-in templates, until the user stops it for good.
fn builtin_notice(noticed: std::path::PathBuf, window: &mut Window, cx: &mut App) {
    let extra = Extra {
        label: "Don't show again",
        run: Rc::new(move |window, cx| {
            let written = noticed
                .parent()
                .map_or(Ok(()), std::fs::create_dir_all)
                .and_then(|()| std::fs::write(&noticed, ""));
            if let Err(e) = written {
                let text = format!("Cannot save the setting: {e}");
                notify(window, cx, text, Severity::Error, NOTICE_TIMEOUT);
            }
        }),
    };
    dialogs::markdown(
        window,
        cx,
        "Built-in Templates",
        BUILTIN_NOTICE_BODY,
        "accent",
        Some(extra),
        None,
    );
}

impl Focusable for TemplatePickerView {
    fn focus_handle(&self, _: &App) -> FocusHandle {
        self.focus.clone()
    }
}

impl Render for TemplatePickerView {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let rows: Vec<AnyElement> = self
            .rows
            .clone()
            .iter()
            .enumerate()
            .map(|(index, row)| self.render_row(index, row, cx))
            .collect();
        let border = cx.theme().border;
        let list = div()
            .w(relative(0.32))
            .min_w(px(260.))
            .max_w(px(380.))
            .flex()
            .flex_col()
            .border_r_1()
            .border_color(border)
            .child(
                div()
                    .flex_none()
                    .p_2()
                    .child(Input::new(&self.search).cleanable(true)),
            )
            .child(
                div()
                    .id("templates")
                    .flex_1()
                    .min_h_0()
                    .px_2()
                    .pb_2()
                    .overflow_y_scroll()
                    .track_scroll(&self.scroll)
                    .on_click(cx.listener(|view, _, window, cx| view.focus_part(true, window, cx)))
                    .children(rows),
            )
            .child(self.render_tools(cx));
        let detail = match (&self.checks, self.action.clone()) {
            (Some(checks), _) => checks.clone().into_any_element(),
            (None, action) => div()
                .size_full()
                .flex()
                .flex_col()
                .child(
                    div()
                        .id("preview")
                        .flex_1()
                        .min_h_0()
                        .px_6()
                        .py_4()
                        .overflow_y_scroll()
                        .child(TextView::new(&self.preview).selectable(true)),
                )
                .when_some(action, |detail, (label, icon)| {
                    detail.child(self.render_action(label, icon, cx))
                })
                .into_any_element(),
        };
        div()
            .key_context("TemplatePicker")
            .track_focus(&self.focus)
            .capture_key_down(cx.listener(Self::on_capture_key))
            .size_full()
            .flex()
            .child(list)
            .child(div().flex_1().min_w_0().h_full().child(detail))
    }
}
