//! The template builder: a chat with the AI on the left, the template it
//! proposes on the right, and Save and Close in the title bar row.

use std::time::Duration;

use crate::gui::widgets::scroll_area;
use gpui_kit::component::WindowExt as _;
use gpui_kit::component::button::{Button, ButtonVariants as _};
use gpui_kit::component::input::{Input, InputEvent, InputState};
use gpui_kit::component::scroll::ScrollableElement as _;
use gpui_kit::component::text::TextViewState;
use gpui_kit::component::{ActiveTheme as _, Disableable as _, Icon, IconName, Sizable as _};
use gpui_kit::prelude::FluentBuilder as _;
use gpui_kit::*;

use crate::app::session::Severity;
use crate::app::template_builder::{Role, TemplateBuilderRuntime};
use crate::gui::bridge::{Bridge, Incoming, notify};
use crate::gui::design::{control, radius, size, space, text};
use crate::gui::dialogs;
use crate::gui::icons;
use crate::gui::widgets::{
    DragArm, action_bar, column_header, empty_state, markdown, pane_header, pane_title, tool,
};

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
            let text = "No note type to save. Generate one first.".to_string();
            notify(window, cx, text, Severity::Warning, NOTICE);
            return;
        };
        let file_name = cx.new(|cx| {
            InputState::new(window, cx)
                .placeholder("file_name (no spaces)")
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
            let label = move |line: &'static str| {
                div().text_size(text::CAPTION).text_color(muted).child(line)
            };
            dialog
                .w(size::DIALOG)
                .title(dialogs::title_text("Save Note Type"))
                .child(
                    div()
                        .flex()
                        .flex_col()
                        .gap(space::XS)
                        .child(label("File name"))
                        .child(Input::new(&file_name).with_size(control::CONTROL))
                        .child(div().h(space::XS))
                        .child(label("Display name"))
                        .child(Input::new(&display_name).with_size(control::CONTROL)),
                )
                .footer(dialogs::buttons("Save", false, {
                    let (view, file, display) = (view.clone(), file.clone(), display.clone());
                    move |window, cx| save_as(&view, &file, &display, window, cx)
                }))
                .on_ok(move |_, window, cx| save_as(&view, &file, &display, window, cx))
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
                let text = format!("Cannot save the note type: {e}");
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
            "You have an unsaved note type. Discard it?",
            "Discard",
            true,
            move |_, cx| {
                let _ = view.update(cx, |view, cx| view.leave(cx));
            },
        );
    }

    /// True while the template differs from the last save.
    pub fn unsaved(&self) -> bool {
        self.runtime.unsaved()
    }

    /// Drop the template without the Closed event: the shell moves on itself.
    pub fn discard(&mut self) {
        self.runtime.leave();
    }

    fn leave(&mut self, cx: &mut Context<Self>) {
        self.runtime.leave();
        cx.emit(BuilderEvent::Closed);
    }

    /// The title, then Close and Save at the right end: Save is the pane's
    /// one primary action.
    fn render_title_row(&self, cx: &mut Context<Self>) -> Stateful<Div> {
        let has_template = self.runtime.current().is_some();
        pane_header("builder-title", &self.drag, cx)
            .child(pane_title(self.runtime.title()).flex_1().min_w_0())
            .child(
                tool("builder-close", IconName::Close, "Close")
                    .on_click(cx.listener(|view, _, window, cx| view.close(window, cx))),
            )
            .child(
                Button::new("builder-save")
                    .primary()
                    .with_size(control::CONTROL)
                    .label("Save")
                    .disabled(!has_template)
                    .tooltip(match has_template {
                        true => "Save the note type",
                        false => "Generate a note type first",
                    })
                    .on_click(cx.listener(|view, _, window, cx| view.save(window, cx))),
            )
    }

    /// The user's message in an accent bubble on the right; the AI's reply as
    /// plain text, as a reader reads it.
    fn render_entry(&self, index: usize, line: &str, cx: &App) -> Div {
        let theme = cx.theme();
        match &self.entries[index] {
            None => div().flex().justify_end().child(
                div()
                    .max_w(size::BUBBLE_MAX)
                    .px(space::MD)
                    .py(space::SM)
                    .rounded(radius::LG)
                    .bg(theme.primary)
                    .text_color(theme.primary_foreground)
                    .child(line.to_string()),
            ),
            Some(reply) => div().child(markdown(reply)),
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
            .map(|(index, (_, line))| self.render_entry(index, line, cx))
            .collect();
        let empty = rows.is_empty() && !busy;
        let theme = cx.theme();
        let (border, muted) = (theme.border, theme.muted_foreground);
        let thinking = div()
            .flex()
            .items_center()
            .gap(space::SM)
            .text_color(muted)
            .child(Icon::new(IconName::LoaderCircle).size(size::ICON_SM))
            .child("Thinking…");
        div()
            .flex()
            .flex_col()
            .flex_none()
            .w(relative(0.4))
            .min_w_0()
            .border_r(size::HAIRLINE)
            .border_color(border)
            .child(column_header("Conversation", cx))
            .child(scroll_area(
                div()
                    .id("builder-chat")
                    .flex_1()
                    .min_h_0()
                    .px(space::LG)
                    .pb(space::LG)
                    .flex()
                    .flex_col()
                    .gap(space::LG)
                    .text_size(text::READING)
                    .overflow_y_scroll()
                    .track_scroll(&self.chat_scroll)
                    .when(empty, |chat| {
                        chat.child(empty_state(
                            icons::extra("sparkles"),
                            "Describe the meeting you want notes for.",
                            cx,
                        ))
                    })
                    .children(rows)
                    .when(busy, |chat| chat.child(thinking)),
                &self.chat_scroll,
            ))
            .child(
                action_bar(cx)
                    .child(
                        div()
                            .flex_1()
                            .min_w_0()
                            .child(Input::new(&self.message).with_size(control::CONTROL)),
                    )
                    .child(
                        Button::new("builder-send")
                            .primary()
                            .with_size(control::CONTROL)
                            .icon(IconName::ArrowUp)
                            .label("Send")
                            .disabled(busy)
                            .on_click(cx.listener(|view, _, window, cx| view.send(window, cx))),
                    ),
            )
    }

    fn render_preview(&self, cx: &App) -> Div {
        div()
            .flex()
            .flex_col()
            .flex_1()
            .min_w_0()
            .child(column_header("Preview", cx))
            .child(
                div()
                    .flex_1()
                    .min_h_0()
                    .px(space::LG)
                    .pb(space::LG)
                    .text_size(text::READING)
                    .overflow_y_scrollbar()
                    .id("builder-preview")
                    .child(markdown(&self.preview)),
            )
    }
}

/// Save under the names the dialog holds. False keeps the dialog open.
fn save_as(
    view: &WeakEntity<BuilderView>,
    file_name: &Entity<InputState>,
    display_name: &Entity<InputState>,
    window: &mut Window,
    cx: &mut App,
) -> bool {
    let Some((file, display)) = save_names(file_name, display_name, cx) else {
        return false;
    };
    let _ = view.update(cx, |view, cx| view.saved_as(&file, &display, window, cx));
    true
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
                    .child(self.render_chat(cx))
                    .child(self.render_preview(cx)),
            )
    }
}
