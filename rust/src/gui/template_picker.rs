//! The template list with its preview: a filter box, the templates by
//! locale and their buttons on the left, the preview on the right. In a flow
//! it is the start panel: a start button under the preview, and the setup
//! checks in place of the preview once they run.

use std::rc::Rc;
use std::time::Duration;

use gpui_kit::component::button::{Button, ButtonVariants as _};
use gpui_kit::component::input::{Input, InputEvent, InputState};
use gpui_kit::component::text::TextViewState;
use gpui_kit::component::{ActiveTheme as _, Disableable as _, Icon, IconName, Sizable as _};
use gpui_kit::prelude::FluentBuilder as _;
use gpui_kit::*;
use tracing::warn;

use crate::adapters::templates::YamlTemplates;
use crate::app::container::Loaded;
use crate::app::notices::BUILTIN_NOTICE_BODY;
use crate::app::session::Severity;
use crate::app::template_catalog::{NO_TEMPLATES, Row, TemplateCatalog};
use crate::gui::bridge::notify;
use crate::gui::design::{control, radius, size, space, text, tint};
use crate::gui::dialogs::{self, Extra};
use crate::gui::icons;
use crate::gui::preflight::PreflightView;
use crate::gui::widgets::{
    DragArm, action_bar, caption, card, markdown, page_title, pane_header, pane_title,
    section_heading, tag,
};

const NOTICE_TIMEOUT: Duration = Duration::from_secs(5);
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
    /// The pane title and the start button, in a flow.
    action: Option<StartAction>,
    /// The setup checks of the chosen template, in place of its preview.
    checks: Option<Entity<PreflightView>>,
    focus: FocusHandle,
    drag: DragArm,
    /// The lines a digest waits for, so a card can say how often notes update.
    digest_lines: usize,
    /// The instructions are open under the role cards.
    instructions_open: bool,
    _subscription: Subscription,
}

impl EventEmitter<TemplatePickerEvent> for TemplatePickerView {}

/// What a flow starts from the picker: the pane title, and the start button
/// at the title bar's right end. A recording starts from the red button.
#[derive(Clone)]
pub struct StartAction {
    pub title: SharedString,
    pub label: &'static str,
    pub icon: IconName,
    pub record: bool,
}

impl TemplatePickerView {
    pub fn new(
        loaded: &Loaded,
        action: Option<StartAction>,
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
            drag: DragArm::default(),
            digest_lines: loaded.config.digest.min_lines,
            instructions_open: false,
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
    pub fn show_checks(&mut self, checks: Entity<PreflightView>, cx: &mut Context<Self>) {
        cx.observe(&checks, |_, _, cx| cx.notify()).detach();
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

    /// The preview's markdown: the chosen template's instructions, which the
    /// role cards above it explain.
    fn show_preview(&mut self, cx: &mut Context<Self>) {
        let markdown = match self.current() {
            Some(name) => self
                .catalog
                .loaded
                .get(&name)
                .map(|template| template.system_prompt.clone())
                .unwrap_or_default(),
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
                    .px(space::SM)
                    .pt(space::MD)
                    .pb(space::XS)
                    .font_weight(text::STRONG)
                    .text_size(text::CAPTION)
                    .text_color(theme.muted_foreground)
                    .child(locale.to_uppercase())
                    .into_any_element();
            }
            Row::Template(name) => name.clone(),
        };
        let selected = self.selected == Some(index);
        // As a macOS list: the accent under the selected row while the list has
        // focus, a grey without it.
        let (background, foreground) = match (selected, self.list_focused) {
            (true, true) => (theme.primary, theme.primary_foreground),
            (true, false) => (theme.foreground.opacity(tint::STRONG), theme.foreground),
            (false, _) => (theme.transparent, theme.foreground),
        };
        let hover = theme.foreground.opacity(tint::SUBTLE);
        let user = self.catalog.user_names.contains(&name);
        let label = self.catalog.label(&name);
        let user_tag = tag("User", cx).when(selected && self.list_focused, |tag| {
            tag.text_color(foreground)
                .bg(foreground.opacity(tint::STRONG))
        });
        div()
            .id(("template", index))
            .flex()
            .items_center()
            .gap(space::SM)
            .h(size::ROW)
            .px(space::SM)
            .rounded(radius::MD)
            .text_size(text::BODY)
            .bg(background)
            .text_color(foreground)
            .when(!selected, |row| row.hover(move |row| row.bg(hover)))
            .child(
                div()
                    .flex_1()
                    .overflow_hidden()
                    .whitespace_nowrap()
                    .text_ellipsis()
                    .child(label),
            )
            .when(user, |row| row.child(user_tag))
            .on_click(cx.listener(move |view, event: &ClickEvent, window, cx| {
                view.focus_part(true, window, cx);
                view.select(index, cx);
                if event.click_count() > 1 {
                    view.pick(cx);
                }
            }))
            .into_any_element()
    }

    /// The selected template's name, source, locale and description.
    fn render_heading(&self, cx: &App) -> Option<Div> {
        let name = self.current()?;
        let template = self.catalog.loaded.get(&name)?;
        let meta = &template.metadata;
        let source = if self.catalog.user_names.contains(&name) {
            "User"
        } else {
            "Built-in"
        };
        Some(
            div()
                .flex()
                .flex_col()
                .gap(space::SM)
                .pt(space::XL)
                .pb(space::MD)
                .child(page_title(meta.name.clone()))
                .child(
                    div()
                        .flex()
                        .gap(space::XS)
                        .child(tag(source, cx))
                        .child(tag(meta.locale.clone(), cx)),
                )
                .when(!meta.description.is_empty(), |heading| {
                    heading.child(
                        div()
                            .text_color(cx.theme().muted_foreground)
                            .child(meta.description.clone()),
                    )
                }),
        )
    }

    /// What the template does, in the order the user meets it: the notes it
    /// keeps during the meeting, the questions to ask any time, the words it
    /// listens for. Then its instructions, folded.
    fn render_roles(&self, cx: &mut Context<Self>) -> Option<Div> {
        let name = self.current()?;
        let template = self.catalog.loaded.get(&name)?;
        let muted = cx.theme().muted_foreground;
        let role = |icon: Icon, title: &'static str, cx: &App| {
            card(cx).gap(space::SM).p(space::MD).child(
                div()
                    .flex()
                    .items_center()
                    .gap(space::SM)
                    .child(icon.size(size::ICON).text_color(cx.theme().primary))
                    .child(section_heading(title)),
            )
        };
        let during = role(icons::extra("sparkles"), "During the meeting", cx).child(
            div().text_color(muted).child(format!(
                "The Notes panel follows the instructions below and updates about every {} \
                 lines while people talk. When the session ends, one last pass writes the \
                 full notes.",
                self.digest_lines
            )),
        );
        let asks = template.quick_actions.iter().map(|action| {
            div()
                .flex()
                .gap(space::MD)
                .child(
                    div()
                        .flex_none()
                        .w(size::ACTION_LABEL)
                        .font_weight(text::STRONG)
                        .child(action.label.clone()),
                )
                .child(
                    div()
                        .flex_1()
                        .text_color(muted)
                        .child(action.description.clone()),
                )
        });
        let ask = role(icons::extra("zap"), "Ask any time", cx)
            .when(template.quick_actions.is_empty(), |card| {
                card.child(
                    div()
                        .text_color(muted)
                        .child("This template has no quick actions."),
                )
            })
            .when(!template.quick_actions.is_empty(), |card| {
                card.child(
                    div()
                        .text_color(muted)
                        .child("Quick Actions in the notes header answer from the meeting so far."),
                )
                .children(asks)
            });
        let hints = template.recognition_hints.clone();
        let listens = (!hints.is_empty()).then(|| {
            role(Icon::new(IconName::Mic), "Listens for", cx).child(
                div()
                    .flex()
                    .flex_wrap()
                    .gap(space::XS)
                    .children(hints.into_iter().map(|hint| tag(hint, cx))),
            )
        });
        let open = self.instructions_open;
        let fold = div()
            .id("instructions")
            .flex()
            .items_center()
            .gap(space::SM)
            .pt(space::SM)
            .cursor_pointer()
            .child(
                Icon::new(match open {
                    true => IconName::ChevronDown,
                    false => IconName::ChevronRight,
                })
                .size(size::ICON_SM),
            )
            .child(section_heading("Instructions"))
            .child(caption("What the AI is told to do", cx))
            .on_click(cx.listener(|view, _, _, cx| {
                view.instructions_open = !view.instructions_open;
                cx.notify();
            }));
        Some(
            div()
                .flex()
                .flex_col()
                .gap(space::MD)
                .child(during)
                .child(ask)
                .children(listens)
                .child(fold)
                .when(open, |roles| roles.child(markdown(&self.preview))),
        )
    }

    /// New, Edit, Edit with AI and Delete, under the list.
    fn render_tools(&self, cx: &mut Context<Self>) -> Div {
        let current = self.current();
        let none = current.is_none();
        let user = current
            .as_ref()
            .is_some_and(|name| self.catalog.user_names.contains(name));
        action_bar(cx)
            .gap(space::XS)
            .px(space::SM)
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

    /// The flow's title bar: the title, then Cancel and the start button at
    /// the right end. While the checks run, the start button waits.
    fn render_header(&self, action: StartAction, cx: &mut Context<Self>) -> Stateful<Div> {
        let checking = self
            .checks
            .as_ref()
            .is_some_and(|checks| checks.read(cx).running());
        let start = Button::new("start")
            .with_size(control::CONTROL)
            .icon(action.icon)
            .label(if checking {
                "Checking…"
            } else {
                action.label
            })
            .loading(checking)
            .disabled(self.current().is_none() || checking)
            .on_click(cx.listener(|view, _, _, cx| view.pick(cx)));
        let start = match action.record {
            true => start.danger(),
            false => start.primary(),
        };
        pane_header("picker-title", &self.drag, cx)
            .child(pane_title(action.title))
            .child(div().flex_1())
            .child(
                Button::new("cancel-start")
                    .ghost()
                    .with_size(control::CONTROL)
                    .label("Cancel")
                    .on_click(cx.listener(|_, _, _, cx| cx.emit(TemplatePickerEvent::Cancelled))),
            )
            .child(start)
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
            .w(size::LIST_COLUMN)
            .flex_none()
            .flex()
            .flex_col()
            .border_r(size::HAIRLINE)
            .border_color(border)
            .child(
                div()
                    .flex_none()
                    .px(space::SM)
                    .pb(space::SM)
                    .child(Input::new(&self.search).cleanable(true)),
            )
            .child(
                div()
                    .id("templates")
                    .flex_1()
                    .min_h_0()
                    .px(space::SM)
                    .pb(space::SM)
                    .overflow_y_scroll()
                    .track_scroll(&self.scroll)
                    .on_click(cx.listener(|view, _, window, cx| view.focus_part(true, window, cx)))
                    .children(rows),
            )
            .child(self.render_tools(cx));
        let detail = match &self.checks {
            Some(checks) => checks.clone().into_any_element(),
            None => div()
                .id("preview")
                .size_full()
                .px(space::XL)
                .pb(space::LG)
                .overflow_y_scroll()
                .children(self.render_heading(cx))
                .map(|preview| match self.render_roles(cx) {
                    Some(roles) => preview.child(roles),
                    None => preview.child(markdown(&self.preview)),
                })
                .into_any_element(),
        };
        let header = self
            .action
            .clone()
            .map(|action| self.render_header(action, cx));
        div()
            .key_context("TemplatePicker")
            .track_focus(&self.focus)
            .capture_key_down(cx.listener(Self::on_capture_key))
            .size_full()
            .flex()
            .flex_col()
            .children(header)
            .child(
                div()
                    .flex()
                    .flex_1()
                    .min_h_0()
                    .child(list)
                    .child(div().flex_1().min_w_0().h_full().child(detail)),
            )
    }
}
