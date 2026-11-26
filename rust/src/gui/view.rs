//! One saved session, read only: the title and the toolbar in the title bar
//! row, the transcript and the speakers on the left, the notes on the right.
//! It looks like the live session pane. The files are `app::saved_session`.

use std::path::{Path, PathBuf};
use std::time::Duration;

use gpui_kit::component::button::{Button, ButtonVariants as _};
use gpui_kit::component::input::{Input, InputEvent, InputState};
use gpui_kit::component::text::{TextView, TextViewState};
use gpui_kit::component::tooltip::Tooltip;
use gpui_kit::component::{ActiveTheme, IconName, Sizable as _};
use gpui_kit::prelude::FluentBuilder as _;
use gpui_kit::*;
use tracing::error;

use crate::adapters::saved_speakers;
use crate::app::saved_session::{self, SavedSession, split_line, spoken_by};
use crate::app::session::{Severity, format_spoken_time};
use crate::app::session_list::title;
use crate::gui::bridge::notify;
use crate::gui::dialogs;
use crate::gui::widgets::{DragArm, title_strip};
use crate::gui::{CopyDigest, CopyTranscript, RenameSession, SESSION_CONTEXT, ShowInFinder};

const COPY_TIMEOUT: Duration = Duration::from_secs(2);
const ERROR_TIMEOUT: Duration = Duration::from_secs(8);

pub enum ReaderEvent {
    Closed,
    /// The session folder moved to this path after a rename.
    Renamed(PathBuf),
}

#[derive(Clone, Copy)]
enum Copied {
    Transcript,
    Digest,
}

pub struct SessionReader {
    dir: PathBuf,
    session: SavedSession,
    transcript: ListState,
    notes: Entity<TextViewState>,
    title: Entity<InputState>,
    drag: DragArm,
    /// The speaker chip under the pointer; its lines are marked.
    hovered_speaker: Option<u32>,
    focus_handle: FocusHandle,
    _subscriptions: Vec<Subscription>,
}

impl EventEmitter<ReaderEvent> for SessionReader {}

impl SessionReader {
    pub fn new(dir: PathBuf, window: &mut Window, cx: &mut Context<Self>) -> Self {
        let mut session = SavedSession::load(&dir);
        let errors = std::mem::take(&mut session.errors);
        let notes = cx.new(|cx| TextViewState::markdown(&session.notes, cx));
        let title = cx.new(|cx| {
            InputState::new(window, cx)
                .placeholder("Untitled")
                .default_value(shown_title(&dir))
        });
        let subscriptions = vec![cx.subscribe_in(&title, window, Self::on_title)];
        let transcript = ListState::new(session.transcript.len(), ListAlignment::Top, px(400.));
        let focus_handle = cx.focus_handle();
        // The toasts need the window, and the keys start on the view.
        cx.defer_in(window, move |view, window, cx| {
            view.focus_handle.focus(window, cx);
            view.report(errors, window, cx);
        });
        Self {
            dir,
            session,
            transcript,
            notes,
            title,
            drag: DragArm::default(),
            hovered_speaker: None,
            focus_handle,
            _subscriptions: subscriptions,
        }
    }

    fn report(&self, errors: Vec<String>, window: &mut Window, cx: &mut Context<Self>) {
        for message in errors {
            error!(target: "ltn.view", "{message} ({})", self.dir.display());
            notify(window, cx, message, Severity::Error, ERROR_TIMEOUT);
        }
    }

    /// Return keeps the new title; leaving the field keeps it too.
    fn on_title(
        &mut self,
        _: &Entity<InputState>,
        event: &InputEvent,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        match event {
            InputEvent::PressEnter { .. } => self.focus_handle.focus(window, cx),
            InputEvent::Blur => self.commit_title(window, cx),
            InputEvent::Change | InputEvent::Focus => {}
        }
    }

    fn commit_title(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let typed = self.title.read(cx).value().trim().to_string();
        if typed == shown_title(&self.dir) {
            return;
        }
        match saved_session::rename_folder(&self.dir, &typed) {
            Ok(Some(moved)) => {
                self.dir = moved.clone();
                let text = format!("Session: {}", shown_title(&self.dir));
                notify(window, cx, text, Severity::Information, COPY_TIMEOUT);
                cx.emit(ReaderEvent::Renamed(moved));
            }
            Ok(None) => {}
            Err(message) => {
                error!(target: "ltn.view", "{message} ({})", self.dir.display());
                notify(window, cx, message, Severity::Error, ERROR_TIMEOUT);
            }
        }
        // An empty or refused name leaves the title as it was: show that again.
        let shown = shown_title(&self.dir);
        self.title
            .update(cx, |title, cx| title.set_value(shown, window, cx));
    }

    fn title_focused(&self, window: &Window, cx: &App) -> bool {
        self.title.read(cx).focus_handle(cx).is_focused(window)
    }

    fn rename(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.title.update(cx, |title, cx| title.focus(window, cx));
    }

    fn show_in_finder(&mut self, cx: &mut Context<Self>) {
        cx.reveal_path(&self.dir);
    }

    fn speaker_label(&self, speaker: u32) -> String {
        self.session
            .speakers
            .as_ref()
            .map_or_else(|| format!("S{speaker}"), |saved| saved.label(speaker))
    }

    fn ask_speaker_name(&mut self, speaker: u32, window: &mut Window, cx: &mut Context<Self>) {
        let current = self
            .session
            .speakers
            .as_ref()
            .and_then(|saved| saved.names.get(&speaker).cloned())
            .unwrap_or_default();
        let view = cx.entity().downgrade();
        dialogs::prompt(
            window,
            cx,
            format!("Name for S{speaker}"),
            &current,
            "Speaker name...",
            move |name, window, cx| {
                let _ = view.update(cx, |view, cx| {
                    view.rename_speaker(speaker, &name, window, cx)
                });
            },
        );
    }

    /// Write the name into the saved files, then show the relabeled transcript.
    fn rename_speaker(
        &mut self,
        speaker: u32,
        name: &str,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        match saved_speakers::rename(&self.dir, speaker, name) {
            Ok(speakers) => {
                self.session.speakers = Some(speakers);
                self.session.reload_transcript(&self.dir);
                self.transcript.reset(self.session.transcript.len());
                let errors = std::mem::take(&mut self.session.errors);
                self.report(errors, window, cx);
            }
            Err(message) => {
                error!(target: "ltn.view", "Cannot rename S{speaker}: {message}");
                let text = format!("Cannot rename S{speaker}: {message}");
                notify(window, cx, text, Severity::Error, ERROR_TIMEOUT);
            }
        }
        cx.notify();
    }

    fn copy(&mut self, copied: Copied, window: &mut Window, cx: &mut Context<Self>) {
        let (text, what, empty) = match copied {
            Copied::Digest => (
                self.session.notes.clone(),
                "Digest copied",
                "No digest to copy",
            ),
            Copied::Transcript => (
                self.session.transcript.join("\n"),
                "Transcript copied",
                "No transcript to copy",
            ),
        };
        if text.is_empty() {
            notify(window, cx, empty.into(), Severity::Warning, COPY_TIMEOUT);
            return;
        }
        cx.write_to_clipboard(ClipboardItem::new_string(text));
        notify(window, cx, what.into(), Severity::Information, COPY_TIMEOUT);
    }

    /// Escape in the title drops what was typed; elsewhere it closes the reader.
    fn on_key(&mut self, event: &KeyDownEvent, window: &mut Window, cx: &mut Context<Self>) {
        if event.keystroke.key != "escape" {
            return;
        }
        if self.title_focused(window, cx) {
            let shown = shown_title(&self.dir);
            self.title
                .update(cx, |title, cx| title.set_value(shown, window, cx));
            self.focus_handle.focus(window, cx);
        } else {
            cx.emit(ReaderEvent::Closed);
        }
        cx.stop_propagation();
    }
}

/// The label as the sidebar shows it.
fn shown_title(dir: &Path) -> String {
    let name = dir
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_default();
    title(&name)
}

/// A small heading over a column, as the live pane's.
fn heading(text: &'static str, cx: &App) -> Div {
    div()
        .flex_none()
        .px_4()
        .pt_3()
        .pb_1()
        .text_xs()
        .font_weight(FontWeight::SEMIBOLD)
        .text_color(cx.theme().muted_foreground)
        .child(text)
}

/// A centered, muted line where a column has nothing to show.
fn empty_state(text: &'static str, cx: &App) -> Div {
    div()
        .flex_1()
        .flex()
        .flex_col()
        .gap_2()
        .items_center()
        .justify_center()
        .px_4()
        .text_sm()
        .text_color(cx.theme().muted_foreground)
        .child(text)
}

impl SessionReader {
    fn render_title_row(&self, cx: &mut Context<Self>) -> Stateful<Div> {
        let hover = cx.theme().foreground.opacity(0.06);
        title_strip("reader-title", &self.drag)
            .gap_2()
            .px_4()
            .child(
                // A press in the field selects text; it must not move the window.
                div()
                    .id("reader-title-field")
                    .w(px(360.))
                    .min_w(px(120.))
                    .flex_shrink(1.)
                    .px_1()
                    .rounded(px(6.))
                    .hover(|field| field.bg(hover))
                    .font_weight(FontWeight::SEMIBOLD)
                    .tooltip(|window, cx| Tooltip::new("Rename").build(window, cx))
                    .on_mouse_down(MouseButton::Left, |_, _, cx| cx.stop_propagation())
                    .child(Input::new(&self.title).appearance(false)),
            )
            .child(div().flex_1())
            .child(
                Button::new("reader-finder")
                    .ghost()
                    .small()
                    .icon(IconName::FolderOpen)
                    .label("Show in Finder")
                    .on_click(cx.listener(|view, _, _, cx| view.show_in_finder(cx))),
            )
    }

    fn heading_with_copy(&self, text: &'static str, copied: Copied, cx: &mut Context<Self>) -> Div {
        heading(text, cx)
            .flex()
            .items_center()
            .justify_between()
            .child(
                Button::new(text)
                    .ghost()
                    .xsmall()
                    .icon(IconName::Copy)
                    .label("Copy")
                    .on_click(
                        cx.listener(move |view, _, window, cx| view.copy(copied, window, cx)),
                    ),
            )
    }

    fn render_transcript(&self, cx: &mut Context<Self>) -> Div {
        let column = div()
            .flex()
            .flex_col()
            .flex_1()
            .min_w_0()
            .child(self.heading_with_copy("Transcript", Copied::Transcript, cx));
        if self.session.transcript.is_empty() {
            let empty = match self.session.audio_only {
                true => empty_state("No transcript. The audio is saved in this folder.", cx).child(
                    Button::new("reader-audio")
                        .small()
                        .icon(IconName::FolderOpen)
                        .label("Show in Finder")
                        .on_click(cx.listener(|view, _, _, cx| view.show_in_finder(cx))),
                ),
                false => empty_state("No transcript.", cx),
            };
            return column.child(empty);
        }
        let highlight = self
            .hovered_speaker
            .map(|speaker| self.speaker_label(speaker));
        let marked = cx.theme().primary.opacity(0.12);
        let muted = cx.theme().muted_foreground;
        let lines = list(
            self.transcript.clone(),
            cx.processor(move |view: &mut Self, index: usize, _, _| {
                let Some(line) = view.session.transcript.get(index) else {
                    return div().into_any_element();
                };
                let mine = highlight
                    .as_deref()
                    .is_some_and(|label| spoken_by(line, label));
                let (stamp, text) = match split_line(line) {
                    (Some(stamp), rest) => (stamp.trim_matches(['[', ']']), rest.trim_start()),
                    (None, _) => ("", line.as_str()),
                };
                div()
                    .w_full()
                    .flex()
                    .gap_3()
                    .px_2()
                    .py(px(3.))
                    .rounded(px(4.))
                    .when(mine, |row| row.bg(marked))
                    .child(
                        div()
                            .flex_none()
                            .text_xs()
                            .pt(px(1.))
                            .text_color(muted)
                            .child(stamp.to_string()),
                    )
                    .child(div().flex_1().min_w_0().child(text.to_string()))
                    .into_any_element()
            }),
        )
        .size_full();
        column
            .child(div().flex_1().min_h_0().px_2().child(lines))
            .children(self.render_speakers(cx))
    }

    fn render_speakers(&self, cx: &mut Context<Self>) -> Option<Div> {
        let chips = self
            .session
            .speakers
            .as_ref()
            .map(|saved| saved.chips())
            .unwrap_or_default();
        if chips.is_empty() {
            return None;
        }
        let theme = cx.theme();
        let (plain, plain_text, hover) = (
            theme.secondary,
            theme.secondary_foreground,
            theme.secondary_hover,
        );
        let items = chips.into_iter().map(|(speaker, _, seconds)| {
            div()
                .id(("reader-speaker", speaker as usize))
                .px_2()
                .py(px(2.))
                .rounded_full()
                .bg(plain)
                .text_color(plain_text)
                .text_xs()
                .cursor_pointer()
                .hover(|chip| chip.bg(hover))
                .tooltip(|window, cx| Tooltip::new("Name this speaker").build(window, cx))
                .child(format!(
                    "{}  {}",
                    self.speaker_label(speaker),
                    format_spoken_time(seconds)
                ))
                .on_hover(cx.listener(move |view, hovered: &bool, _, cx| {
                    if *hovered || view.hovered_speaker == Some(speaker) {
                        view.hovered_speaker = hovered.then_some(speaker);
                        cx.notify();
                    }
                }))
                .on_click(cx.listener(move |view, _, window, cx| {
                    view.ask_speaker_name(speaker, window, cx)
                }))
        });
        Some(
            div()
                .flex()
                .flex_wrap()
                .flex_none()
                .gap_2()
                .px_4()
                .py_2()
                .border_t_1()
                .border_color(cx.theme().border)
                .children(items),
        )
    }

    fn render_notes(&self, cx: &mut Context<Self>) -> Div {
        let column = div()
            .flex()
            .flex_col()
            .flex_1()
            .min_w_0()
            .child(self.heading_with_copy("Digest", Copied::Digest, cx));
        if self.session.notes.is_empty() {
            return column.child(empty_state("No notes for this session.", cx));
        }
        column.child(
            div()
                .id("reader-notes")
                .flex_1()
                .min_h_0()
                .px_4()
                .pb_2()
                .overflow_y_scroll()
                .child(TextView::new(&self.notes).selectable(true)),
        )
    }
}

impl Focusable for SessionReader {
    fn focus_handle(&self, _: &App) -> FocusHandle {
        self.focus_handle.clone()
    }
}

impl Render for SessionReader {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let border = cx.theme().border;
        // The Session menu's copy, rename and Finder items apply here too.
        div()
            .key_context(SESSION_CONTEXT)
            .track_focus(&self.focus_handle)
            .on_key_down(cx.listener(Self::on_key))
            .on_action(cx.listener(|view, _: &CopyTranscript, window, cx| {
                view.copy(Copied::Transcript, window, cx)
            }))
            .on_action(
                cx.listener(|view, _: &CopyDigest, window, cx| {
                    view.copy(Copied::Digest, window, cx)
                }),
            )
            .on_action(cx.listener(|view, _: &RenameSession, window, cx| view.rename(window, cx)))
            .on_action(cx.listener(|view, _: &ShowInFinder, _, cx| view.show_in_finder(cx)))
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
                    .border_color(border)
                    .child(self.render_transcript(cx))
                    .child(div().w(px(1.)).h_full().bg(border))
                    .child(self.render_notes(cx)),
            )
    }
}
