//! One saved session, read only: the title and the toolbar in the title bar
//! row, the transcript and the speakers on the left, the notes on the right.
//! It looks like the live session pane. The files are `app::saved_session`.

use std::path::{Path, PathBuf};
use std::time::Duration;

use crate::gui::widgets::scroll_area;
use gpui_kit::component::button::Button;
use gpui_kit::component::input::{InputEvent, InputState};
use gpui_kit::component::scroll::ScrollableElement as _;
use gpui_kit::component::text::TextViewState;
use gpui_kit::component::{ActiveTheme, IconName};
use gpui_kit::*;
use tracing::error;

use crate::adapters::saved_speakers;
use crate::app::saved_session::{self, SavedSession, split_line, spoken_by};
use crate::app::session::{Severity, format_spoken_time};
use crate::app::session_list::title;
use crate::gui::bridge::notify;
use crate::gui::design::{size, space, text, tint};
use crate::gui::dialogs;
use crate::gui::icons;
use crate::gui::widgets::{
    DragArm, TurnColors, TurnEdge, column_header, empty_state, inline_tool, markdown, pane_header,
    speaker_chip, status_strip, tag, title_field, tool, transcript_line,
};
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
        let transcript = ListState::new(
            session.transcript.len(),
            ListAlignment::Top,
            size::LIST_OVERDRAW,
        );
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
                "Notes copied",
                "No notes to copy",
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

impl SessionReader {
    fn render_title_row(&self, cx: &mut Context<Self>) -> Stateful<Div> {
        pane_header("reader-title", &self.drag, cx)
            .child(title_field("reader-title-field", &self.title, cx))
            .child(div().flex_1())
            .child(
                tool("reader-finder", IconName::FolderOpen, "Show in Finder")
                    .on_click(cx.listener(|view, _, _, cx| view.show_in_finder(cx))),
            )
    }

    fn copy_button(&self, id: &'static str, copied: Copied, cx: &mut Context<Self>) -> Button {
        inline_tool(id, IconName::Copy, "Copy")
            .on_click(cx.listener(move |view, _, window, cx| view.copy(copied, window, cx)))
    }

    fn render_transcript(&self, cx: &mut Context<Self>) -> Div {
        let column = div()
            .flex()
            .flex_col()
            .flex_none()
            .w(relative(0.4))
            .min_w_0()
            .child(column_header("Transcript", cx).child(self.copy_button(
                "reader-copy-transcript",
                Copied::Transcript,
                cx,
            )));
        if self.session.transcript.is_empty() {
            let line = match self.session.audio_only {
                true => "No transcript. The audio is saved in this folder.",
                false => "No transcript.",
            };
            return column.child(empty_state(IconName::FileText, line, cx));
        }
        let highlight = self
            .hovered_speaker
            .map(|speaker| self.speaker_label(speaker));
        let marked = cx.theme().primary.opacity(tint::STRONG);
        let muted = cx.theme().muted_foreground;
        let fill = cx.theme().foreground.opacity(tint::BLOCK);
        let labels: Vec<String> = self
            .session
            .speakers
            .as_ref()
            .map(|saved| {
                saved
                    .chips()
                    .into_iter()
                    .map(|(speaker, _, _)| saved.label(speaker))
                    .collect()
            })
            .unwrap_or_default();
        let lines = list(
            self.transcript.clone(),
            cx.processor(move |view: &mut Self, index: usize, _, _| {
                let Some(line) = view.session.transcript.get(index) else {
                    return div().into_any_element();
                };
                let mine = highlight
                    .as_deref()
                    .is_some_and(|label| spoken_by(line, label));
                let (stamp, rest) = match split_line(line) {
                    (Some(stamp), rest) => (stamp.trim_matches(['[', ']']), rest.trim_start()),
                    (None, _) => ("", line.as_str()),
                };
                let (speaker, words) = split_speaker(rest, &labels);
                let speaker_of = |at: usize| {
                    view.session
                        .transcript
                        .get(at)
                        .map(|other| split_speaker(split_line(other).1.trim_start(), &labels).0)
                };
                let same = |other: Option<Option<&str>>| {
                    speaker.is_some() && other.is_some_and(|other| other == speaker)
                };
                let edge = TurnEdge {
                    first: !same(index.checked_sub(1).and_then(speaker_of)),
                    last: !same(speaker_of(index + 1)),
                };
                let colors = TurnColors {
                    fill,
                    marked: mine.then_some(marked),
                    muted,
                };
                transcript_line(
                    stamp.to_string(),
                    speaker.map(|name| name.to_string().into()),
                    words.to_string(),
                    edge,
                    colors,
                )
                .into_any_element()
            }),
        )
        .size_full();
        column.child(
            div()
                .flex_1()
                .min_h_0()
                .flex()
                .flex_col()
                .text_size(text::READING)
                .child(scroll_area(lines, &self.transcript)),
        )
    }

    /// The note type and the speakers, in the strip under both columns, as in
    /// a live session.
    fn render_speakers(&self, cx: &mut Context<Self>) -> Option<Div> {
        let chips = self
            .session
            .speakers
            .as_ref()
            .map(|saved| saved.chips())
            .unwrap_or_default();
        let template = self.session.template.clone();
        if chips.is_empty() && template.is_none() {
            return None;
        }
        let items = chips.into_iter().map(|(speaker, _, seconds)| {
            let label = format!(
                "{}  {}",
                self.speaker_label(speaker),
                format_spoken_time(seconds)
            );
            speaker_chip(("reader-speaker", speaker as usize), label, false, cx)
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
            status_strip(cx)
                .gap(space::XS)
                .overflow_hidden()
                .children(template.map(|name| tag(name, cx)))
                .children(items),
        )
    }

    fn render_notes(&self, cx: &mut Context<Self>) -> Div {
        let column = div().flex().flex_col().flex_1().min_w_0().child(
            column_header("Notes", cx).child(self.copy_button(
                "reader-copy-notes",
                Copied::Digest,
                cx,
            )),
        );
        if self.session.notes.is_empty() {
            return column.child(empty_state(
                icons::extra("sparkles"),
                "No notes for this session.",
                cx,
            ));
        }
        column.child(
            div()
                .flex_1()
                .min_h_0()
                .px(space::LG)
                .pb(space::SM)
                .text_size(text::READING)
                .overflow_y_scrollbar()
                .id("reader-notes")
                .child(markdown(&self.notes)),
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
                    .child(self.render_transcript(cx))
                    .child(div().w(size::HAIRLINE).h_full().bg(border))
                    .child(self.render_notes(cx)),
            )
            .children(self.render_speakers(cx))
    }
}

/// The speaker label a saved line starts with, and the words after it.
fn split_speaker<'a>(rest: &'a str, labels: &[String]) -> (Option<&'a str>, &'a str) {
    labels
        .iter()
        .find_map(|label| {
            let words = rest.strip_prefix(label.as_str())?.strip_prefix(": ")?;
            Some((Some(&rest[..label.len()]), words))
        })
        .unwrap_or((None, rest))
}
