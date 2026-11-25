//! The template list with its preview, as the TUI's template picker: a filter
//! box and the templates by locale on the left, the preview on the right.

use std::rc::Rc;
use std::time::Duration;

use gpui_kit::component::input::{Input, InputEvent, InputState};
use gpui_kit::component::text::{TextView, TextViewState};
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
use crate::gui::theme::{color, on};
use crate::gui::widgets::{footer, header, hint, panel};

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
    focus: FocusHandle,
    _subscription: Subscription,
}

impl EventEmitter<TemplatePickerEvent> for TemplatePickerView {}

impl TemplatePickerView {
    pub fn new(loaded: &Loaded, window: &mut Window, cx: &mut Context<Self>) -> Self {
        let catalog = TemplateCatalog::new(
            YamlTemplates::new(&loaded.paths),
            loaded.infra.show_builtin_templates,
        );
        let search = cx.new(|cx| InputState::new(window, cx).placeholder("Filter templates..."));
        let subscription = cx.subscribe_in(&search, window, Self::on_search);
        let preview = cx.new(|cx| TextViewState::markdown("", cx));
        let list_focused = catalog.loaded.is_empty();
        let mut view = Self {
            catalog,
            rows: Vec::new(),
            selected: None,
            search,
            preview,
            scroll: ScrollHandle::new(),
            list_focused,
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
        self.selected = Some(row);
        self.scroll.scroll_to_item(row);
        self.show_preview(cx);
        cx.notify();
    }

    /// A key, or a click on its hint. The letter commands act only while the list has the keys.
    fn press(&mut self, name: &str, window: &mut Window, cx: &mut Context<Self>) {
        match name {
            "up" | "down" => self.step(name == "down", cx),
            "enter" => {
                if let Some(key) = self.current() {
                    cx.emit(TemplatePickerEvent::Picked(key));
                }
            }
            "escape" => cx.emit(TemplatePickerEvent::Cancelled),
            "tab" => {
                let list = !self.list_focused;
                self.focus_part(list, window, cx);
            }
            "n" => cx.emit(TemplatePickerEvent::Create),
            "a" => {
                if let Some(key) = self.current() {
                    cx.emit(TemplatePickerEvent::EditWithAi(key));
                }
            }
            "e" => self.edit(window, cx),
            "x" => self.ask_delete(window, cx),
            _ => {}
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
        if modifiers.control || modifiers.platform || modifiers.alt {
            return;
        }
        let key = keystroke.key.as_str();
        let navigation = matches!(key, "up" | "down" | "enter" | "escape" | "tab");
        let command = self.list_focused && matches!(key, "n" | "a" | "e" | "x");
        if navigation || command {
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
        let name = match row {
            Row::Locale(locale) => {
                return div()
                    .px_2()
                    .pt_2()
                    .font_weight(FontWeight::BOLD)
                    .text_color(color(cx, "text-muted"))
                    .text_xs()
                    .child(locale.to_uppercase())
                    .into_any_element();
            }
            Row::Template(name) => name.clone(),
        };
        let selected = self.selected == Some(index);
        let (background, text, muted) = match (selected, self.list_focused) {
            (true, true) => {
                let text = on(cx, "block-cursor-foreground", "block-cursor-background");
                (color(cx, "block-cursor-background"), text, text)
            }
            (true, false) => (
                color(cx, "block-cursor-blurred-background"),
                color(cx, "foreground"),
                color(cx, "text-muted"),
            ),
            (false, _) => (
                color(cx, "background"),
                color(cx, "foreground"),
                color(cx, "text-muted"),
            ),
        };
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
            .gap_2()
            .px_2()
            .py_1()
            .rounded(px(4.))
            .bg(background)
            .text_color(text)
            .cursor_pointer()
            .child(self.catalog.label(&name))
            .child(div().text_color(muted).child(format!("({locale})")))
            .when(user, |row| {
                row.child(div().text_color(muted).child("[user]"))
            })
            .on_click(cx.listener(move |view, event: &ClickEvent, window, cx| {
                view.focus_part(true, window, cx);
                view.select(index, cx);
                if event.click_count() > 1 {
                    view.press("enter", window, cx);
                }
            }))
            .into_any_element()
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
        let title = format!(
            "Select a template ({} available)",
            self.catalog.loaded.len()
        );
        let list = panel("", self.list_focused, cx).flex_1().child(
            div()
                .id("templates")
                .flex_1()
                .min_h_0()
                .overflow_y_scroll()
                .track_scroll(&self.scroll)
                .on_click(cx.listener(|view, _, window, cx| view.focus_part(true, window, cx)))
                .children(rows),
        );
        let entity = cx.entity().downgrade();
        let on_press = Rc::new(
            move |name: &'static str, window: &mut Window, cx: &mut App| {
                let _ = entity.update(cx, |view, cx| view.press(name, window, cx));
            },
        );
        div()
            .key_context("TemplatePicker")
            .track_focus(&self.focus)
            .capture_key_down(cx.listener(Self::on_capture_key))
            .size_full()
            .flex()
            .flex_col()
            .child(header(title, cx))
            .child(
                div()
                    .flex_1()
                    .min_h_0()
                    .flex()
                    .gap_3()
                    .p_3()
                    .pt_4()
                    .child(
                        div()
                            .w(relative(0.35))
                            .min_w(px(260.))
                            .max_w(px(420.))
                            .flex()
                            .flex_col()
                            .gap_3()
                            .child(Input::new(&self.search).cleanable(true))
                            .child(list),
                    )
                    .child(
                        panel("Preview", false, cx).flex_1().child(
                            div()
                                .id("preview")
                                .flex_1()
                                .min_h_0()
                                .overflow_y_scroll()
                                .child(TextView::new(&self.preview).selectable(true)),
                        ),
                    ),
            )
            .child(footer(
                vec![
                    hint("⏎", "Select", "enter"),
                    hint("↑↓", "Navigate", "down"),
                    hint("tab", "List/Filter", "tab"),
                    hint("n", "New", "n"),
                    hint("e", "Edit", "e"),
                    hint("a", "Edit with AI", "a"),
                    hint("x", "Delete", "x"),
                    hint("esc", "Cancel", "escape"),
                ],
                on_press,
                cx,
            ))
    }
}
