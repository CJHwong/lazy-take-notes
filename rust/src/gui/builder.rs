//! The template builder: a chat with the AI on the left, the template it
//! proposes on the right, and Save and Close in the title bar row.

use std::time::Duration;

use gpui_kit::component::WindowExt as _;
use gpui_kit::component::button::{Button, ButtonVariants as _};
use gpui_kit::component::dialog::{DialogAction, DialogClose, DialogFooter};
use gpui_kit::component::input::{Input, InputEvent, InputState};
use gpui_kit::component::text::{TextView, TextViewState};
use gpui_kit::component::{ActiveTheme as _, Disableable as _, IconName, Sizable as _};
use gpui_kit::prelude::FluentBuilder as _;
use gpui_kit::*;

use crate::app::session::Severity;
use crate::app::template_builder::{Role, TemplateBuilderRuntime};
use crate::gui::bridge::{Bridge, Incoming, notify};
use crate::gui::dialogs;
use crate::gui::template_picker::BAR_HEIGHT;
use crate::gui::widgets::{DragArm, title_strip};

/// As long as the TUI's notices stay.
const NOTICE: Duration = Duration::from_secs(5);

pub enum BuilderEvent {
    Closed,
}

pub struct BuilderView {
    runtime: TemplateBuilderRuntime,
    bridge: Bridge,
    message: Entity<InputState>,
    /// A Markdown view per AI entry, None for a user entry. The chat only
    /// grows, so the views keep their text selection across redraws.
    entries: Vec<Option<Entity<TextViewState>>>,
    preview: Entity<TextViewState>,
    preview_revision: u64,
    chat_scroll: ScrollHandle,
    focus: FocusHandle,
    drag: DragArm,
    _subscriptions: Vec<Subscription>,
    _pump: Task<()>,
}

impl EventEmitter<BuilderEvent> for BuilderView {}

impl BuilderView {
    pub fn new(
        runtime: TemplateBuilderRuntime,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Self {
        let mut bridge = Bridge::new();
        let pump = bridge.pump(window, cx, Self::incoming);
        let placeholder = runtime.placeholder();
        let message = cx.new(|cx| InputState::new(window, cx).placeholder(placeholder));
        let preview_text = runtime.preview();
        let preview = cx.new(|cx| TextViewState::markdown(&preview_text, cx));
        let subscriptions = vec![cx.subscribe_in(&message, window, Self::on_message)];
        let mut view = Self {
            preview_revision: runtime.preview_revision(),
            runtime,
            bridge,
            message,
            entries: Vec::new(),
            preview,
            chat_scroll: ScrollHandle::new(),
            focus: cx.focus_handle(),
            drag: DragArm::default(),
            _subscriptions: subscriptions,
            _pump: pump,
        };
        view.sync(window, cx);
        view
    }

    fn incoming(&mut self, incoming: Incoming, window: &mut Window, cx: &mut Context<Self>) {
        let Incoming::Task(payload) = incoming else {
            return;
        };
        if self.runtime.task_result(payload).is_ok() {
            self.sync(window, cx);
        }
    }

    fn on_message(
        &mut self,
        _: &Entity<InputState>,
        event: &InputEvent,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if let InputEvent::PressEnter { .. } = event {
            self.send(window, cx);
        }
    }

    fn send(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if self.runtime.busy() {
            return;
        }
        let text = self.message.read(cx).value().trim().to_string();
        if text.is_empty() {
            return;
        }
        self.message
            .update(cx, |message, cx| message.set_value("", window, cx));
        self.runtime.send(text, &self.bridge.tasks);
        self.sync(window, cx);
    }

    /// Show the runtime after a send or a reply.
    fn sync(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let chat = self.runtime.chat();
        for (role, text) in &chat[self.entries.len()..] {
            // The user's words are shown as typed; only the AI writes Markdown.
            let entry = match role {
                Role::User => None,
                Role::Ai => Some(cx.new(|cx| TextViewState::markdown(text, cx))),
            };
            self.entries.push(entry);
        }
        if self.preview_revision != self.runtime.preview_revision() {
            self.preview_revision = self.runtime.preview_revision();
            let text = self.runtime.preview();
            self.preview
                .update(cx, |preview, cx| preview.set_text(&text, cx));
        }
        let busy = self.runtime.busy();
        self.message.update(cx, |message, cx| {
            message.set_disabled(busy, cx);
            if !busy {
                message.focus(window, cx);
            }
        });
        self.chat_scroll.scroll_to_bottom();
        cx.notify();
    }

    fn save(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let Some(suggestion) = self.runtime.save_suggestion() else {
            let text = "No template to save. Generate one first.".to_string();
            notify(window, cx, text, Severity::Warning, NOTICE);
            return;
        };
        let file_name = cx.new(|cx| {
            InputState::new(window, cx)
                .placeholder("template_name (no spaces)")
                .default_value(suggestion.file_name.clone())
        });
        let display_name = cx.new(|cx| {
            InputState::new(window, cx)
                .placeholder("Name shown in the picker")
                .default_value(suggestion.display_name.clone())
        });
        let view = cx.entity().downgrade();
        // Return in either field saves, as the dialog's Save button does.
        // The subscription runs inside this view's update, so it saves on `this`.
        for field in [&file_name, &display_name] {
            let (file_name, display_name) = (file_name.clone(), display_name.clone());
            cx.subscribe_in(
                field,
                window,
                move |this, _, event: &InputEvent, window, cx| {
                    let InputEvent::PressEnter { .. } = event else {
                        return;
                    };
                    if let Some((file, display)) = save_names(&file_name, &display_name, cx) {
                        window.close_dialog(cx);
                        this.saved_as(&file, &display, window, cx);
                    }
                },
            )
            .detach();
        }
        let first = std::cell::Cell::new(true);
        window.open_dialog(cx, move |dialog, window, cx| {
            if first.replace(false) {
                file_name.update(cx, |field, cx| {
                    field.focus(window, cx);
                    field.select_all(window, cx);
                });
            }
            let (view, file, display) = (view.clone(), file_name.clone(), display_name.clone());
            let muted = cx.theme().muted_foreground;
            let label = move |text: &'static str| div().text_sm().text_color(muted).child(text);
            dialog
                .w(px(460.))
                .title(
                    div()
                        .font_weight(FontWeight::SEMIBOLD)
                        .child("Save Template"),
                )
                .child(
                    div()
                        .flex()
                        .flex_col()
                        .gap_2()
                        .child(label("File name"))
                        .child(Input::new(&file_name))
                        .child(label("Display name"))
                        .child(Input::new(&display_name)),
                )
                .footer(
                    DialogFooter::new()
                        .justify_end()
                        .child(DialogClose::new().trigger(|button| button.label("Cancel")))
                        .child(
                            DialogAction::new()
                                .child(Button::new("save-template").primary().label("Save")),
                        ),
                )
                .on_ok(move |_, window, cx| {
                    let Some((file, display)) = save_names(&file, &display, cx) else {
                        return false;
                    };
                    let _ = view.update(cx, |view, cx| view.saved_as(&file, &display, window, cx));
                    true
                })
        });
    }

    fn saved_as(
        &mut self,
        file_name: &str,
        display_name: &str,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let saved = self.runtime.save(file_name, display_name);
        self.sync(window, cx);
        match saved {
            None => {}
            Some(Ok((safe_name, path))) => {
                let text = format!("Saved as {safe_name} → {}", path.display());
                notify(window, cx, text, Severity::Information, NOTICE);
            }
            Some(Err(e)) => {
                let text = format!("Cannot save template: {e}");
                notify(window, cx, text, Severity::Error, NOTICE);
            }
        }
    }

    fn close(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if !self.runtime.unsaved() {
            return self.leave(cx);
        }
        let view = cx.entity().downgrade();
        dialogs::confirm(
            window,
            cx,
            "You have an unsaved template. Quit anyway?",
            "Quit",
            true,
            move |_, cx| {
                let _ = view.update(cx, |view, cx| view.leave(cx));
            },
        );
    }

    fn leave(&mut self, cx: &mut Context<Self>) {
        self.runtime.leave();
        cx.emit(BuilderEvent::Closed);
    }

    fn render_title_row(&self, cx: &mut Context<Self>) -> Stateful<Div> {
        let has_template = self.runtime.current().is_some();
        title_strip("builder-title", &self.drag)
            .gap_2()
            .px_4()
            .child(
                div()
                    .flex_1()
                    .min_w_0()
                    .truncate()
                    .font_weight(FontWeight::SEMIBOLD)
                    .child(self.runtime.title()),
            )
            .child(
                Button::new("builder-save")
                    .primary()
                    .small()
                    .label("Save")
                    .disabled(!has_template)
                    .tooltip(match has_template {
                        true => "Save the template",
                        false => "Generate a template first",
                    })
                    .on_click(cx.listener(|view, _, window, cx| view.save(window, cx))),
            )
            .child(
                Button::new("builder-close")
                    .ghost()
                    .small()
                    .icon(IconName::Close)
                    .label("Close")
                    .on_click(cx.listener(|view, _, window, cx| view.close(window, cx))),
            )
    }

    fn render_entry(&self, index: usize, text: &str, cx: &App) -> Div {
        let theme = cx.theme();
        let bubble = div().max_w(relative(0.85)).px_3().py_2().rounded(px(10.));
        match &self.entries[index] {
            None => div().flex().justify_end().child(
                bubble
                    .bg(theme.primary)
                    .text_color(theme.primary_foreground)
                    .child(text.to_string()),
            ),
            Some(markdown) => div().flex().child(
                bubble
                    .bg(theme.secondary)
                    .text_color(theme.foreground)
                    .child(TextView::new(markdown).selectable(true)),
            ),
        }
    }

    fn render_chat(&self, cx: &mut Context<Self>) -> Div {
        let busy = self.runtime.busy();
        let rows: Vec<Div> = self
            .runtime
            .chat()
            .iter()
            .enumerate()
            .take(self.entries.len())
            .map(|(index, (_, text))| self.render_entry(index, text, cx))
            .collect();
        let theme = cx.theme();
        let (border, muted) = (theme.border, theme.muted_foreground);
        div()
            .flex()
            .flex_col()
            .flex_1()
            .min_w_0()
            .border_r_1()
            .border_color(border)
            .child(
                div()
                    .id("builder-chat")
                    .flex_1()
                    .min_h_0()
                    .px_4()
                    .py_3()
                    .flex()
                    .flex_col()
                    .gap_3()
                    .overflow_y_scroll()
                    .track_scroll(&self.chat_scroll)
                    .children(rows)
                    .when(busy, |chat| {
                        chat.child(div().italic().text_color(muted).child("Thinking…"))
                    }),
            )
            .child(
                div()
                    .flex()
                    .flex_none()
                    .items_center()
                    .gap_2()
                    .h(BAR_HEIGHT)
                    .px_4()
                    .border_t_1()
                    .border_color(border)
                    .child(div().flex_1().min_w_0().child(Input::new(&self.message)))
                    .child(
                        Button::new("builder-send")
                            .primary()
                            .icon(IconName::ArrowUp)
                            .label(if busy { "Thinking…" } else { "Send" })
                            .loading(busy)
                            .disabled(busy)
                            .on_click(cx.listener(|view, _, window, cx| view.send(window, cx))),
                    ),
            )
    }

    fn render_preview(&self, cx: &App) -> Div {
        let muted = cx.theme().muted_foreground;
        div()
            .flex()
            .flex_col()
            .flex_1()
            .min_w_0()
            .child(
                div()
                    .flex_none()
                    .px_4()
                    .pt_3()
                    .pb_1()
                    .text_xs()
                    .font_weight(FontWeight::SEMIBOLD)
                    .text_color(muted)
                    .child("Preview"),
            )
            .child(
                div()
                    .id("builder-preview")
                    .flex_1()
                    .min_h_0()
                    .px_4()
                    .pb_3()
                    .overflow_y_scroll()
                    .child(TextView::new(&self.preview).selectable(true)),
            )
    }
}

/// The names the save dialog holds. None keeps the dialog open, as an empty
/// file name does in the TUI.
fn save_names(
    file_name: &Entity<InputState>,
    display_name: &Entity<InputState>,
    cx: &App,
) -> Option<(String, String)> {
    let file = file_name.read(cx).value().trim().to_string();
    if file.is_empty() {
        return None;
    }
    let display = display_name.read(cx).value().trim().to_string();
    Some((file, display))
}

impl Drop for BuilderView {
    /// A click elsewhere in the sidebar drops the view: its call to the AI ends.
    fn drop(&mut self) {
        self.runtime.leave();
    }
}

impl Focusable for BuilderView {
    fn focus_handle(&self, _: &App) -> FocusHandle {
        self.focus.clone()
    }
}

impl Render for BuilderView {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        div()
            .key_context("TemplateBuilder")
            .track_focus(&self.focus)
            .on_key_down(cx.listener(|view, event: &KeyDownEvent, window, cx| {
                if event.keystroke.key == "escape" {
                    view.close(window, cx);
                }
            }))
            .size_full()
            .flex()
            .flex_col()
            .child(self.render_title_row(cx))
            .child(
                div()
                    .flex()
                    .flex_1()
                    .min_h_0()
                    .border_t_1()
                    .border_color(cx.theme().border)
                    .child(self.render_chat(cx))
                    .child(self.render_preview(cx)),
            )
    }
}
