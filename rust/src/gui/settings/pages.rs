//! What the settings pane draws: the title bar with Revert and Save, the
//! page list, and each page as titled cards of rows, as System Settings
//! draws them.

use gpui_kit::component::button::{Button, ButtonVariants as _};
use gpui_kit::component::input::{Input, Textarea};
use gpui_kit::component::progress::Progress;
use gpui_kit::component::radio::Radio;
use gpui_kit::component::select::Select;
use gpui_kit::component::switch::Switch;
use gpui_kit::component::tooltip::Tooltip;
use gpui_kit::component::{ActiveTheme as _, Disableable as _, Icon, IconName, Sizable as _};
use gpui_kit::prelude::FluentBuilder as _;
use gpui_kit::*;

use super::{API_KEY, CUSTOM_PATH, DEVICE, PAGES, Page, SettingsView};
use crate::app::download_progress::megabytes;
use crate::app::session::status_line::{LEVEL_TOP, level_step};
use crate::app::settings::rows::{RowAction, row_actions, row_detail, row_status, status_color};
use crate::app::settings::{Meter, SPEAKER, open_privacy};
use crate::domain::model_asset::AssetStatus;
use crate::gui::design::{control, size, space, tint};
use crate::gui::widgets::{
    caption, card, pane_header, pane_title, section_heading, source_row, tool,
};

/// A theme color by the name the runtime gives it.
fn tint_of(name: Option<&str>, cx: &App) -> Hsla {
    let theme = cx.theme();
    match name {
        Some("success") => theme.success,
        Some("warning") => theme.warning,
        Some("error") => theme.danger,
        Some("accent") => theme.primary,
        _ => theme.foreground,
    }
}

fn note(content: impl Into<SharedString>, cx: &App) -> Div {
    caption(content, cx)
}

/// A card row: the label and its help at the left, the control at the right
/// in a column of one width, so every control lines up.
fn row(label: &'static str, control: impl IntoElement, help: &str, cx: &App) -> Div {
    div()
        .flex()
        .items_center()
        .gap(space::LG)
        .min_h(size::FORM_ROW)
        .px(space::MD)
        .py(space::SM)
        .child(
            div()
                .flex_1()
                .min_w_0()
                .flex()
                .flex_col()
                .gap(space::XXS)
                .child(label)
                .when(!help.is_empty(), |column| {
                    column.child(note(help.to_string(), cx))
                }),
        )
        .child(
            div()
                .flex_none()
                .w(size::FORM_FIELD)
                .flex()
                .justify_end()
                .child(control),
        )
}

/// A row whose control needs the card's whole width: a text area, a list.
fn wide_row(content: impl IntoElement) -> Div {
    div().px(space::MD).py(space::SM).child(content)
}

/// The buttons of a card, right-aligned in its last row.
fn buttons() -> Div {
    div()
        .flex()
        .flex_wrap()
        .items_center()
        .justify_end()
        .gap(space::SM)
        .px(space::MD)
        .py(space::SM)
}

/// A titled card of rows with inset dividers between them.
fn section(title: &'static str, rows: Vec<AnyElement>, cx: &App) -> AnyElement {
    let line = cx.theme().border;
    let mut card = card(cx);
    for (index, item) in rows.into_iter().enumerate() {
        if index > 0 {
            card = card.child(div().mx(space::MD).h(size::HAIRLINE).bg(line));
        }
        card = card.child(item);
    }
    div()
        .flex()
        .flex_col()
        .gap(space::SM)
        .child(section_heading(title).px(space::XS))
        .child(card)
        .into_any_element()
}

impl SettingsView {
    fn input(&self, id: &'static str) -> Input {
        Input::new(&self.inputs[id]).with_size(control::CONTROL)
    }

    fn field(&self, id: &'static str, label: &'static str, help: &str, cx: &App) -> Div {
        row(label, div().w_full().child(self.input(id)), help, cx)
    }

    fn select(&self, id: &'static str, label: &'static str, help: &str, cx: &App) -> Div {
        let select = Select::new(&self.selects[id]).with_size(control::CONTROL);
        row(label, div().w_full().child(select), help, cx)
    }

    fn switch(
        &self,
        id: &'static str,
        label: &'static str,
        help: &str,
        cx: &mut Context<Self>,
    ) -> Div {
        let on = self.switches.get(id).copied().unwrap_or(false);
        let view = cx.entity().downgrade();
        let control = Switch::new(id)
            .with_size(control::CONTROL)
            .checked(on)
            .on_click(move |on, _, cx| {
                let on = *on;
                let _ = view.update(cx, |view, cx| {
                    view.switches.insert(id, on);
                    cx.notify();
                });
            });
        row(label, control, help, cx)
    }

    fn button(
        id: &'static str,
        label: &'static str,
        cx: &mut Context<Self>,
        run: impl Fn(&mut Self, &mut Window, &mut Context<Self>) + 'static,
    ) -> Button {
        Button::new(id)
            .with_size(control::CONTROL)
            .label(label)
            .on_click(cx.listener(move |view, _, window, cx| run(view, window, cx)))
    }

    // --- Models ---------------------------------------------------------

    fn models_page(&self, cx: &mut Context<Self>) -> Vec<AnyElement> {
        let models_text = self
            .runtime
            .models_text
            .clone()
            .unwrap_or_else(|| "Use Refresh to list the models on the server.".into());
        let key = div()
            .w_full()
            .flex()
            .flex_col()
            .items_end()
            .gap(space::SM)
            .child(div().w_full().child(self.input(API_KEY)))
            .when(self.key_stored, |column| {
                column.child(Self::button(
                    "clear-key",
                    "Clear Stored Key",
                    cx,
                    |view, window, cx| view.clear_key(window, cx),
                ))
            });
        let tools = buttons()
            .child(Self::button("ai-test", "Test", cx, |view, window, cx| {
                view.call(window, cx, |runtime, form, io| runtime.test_ai(form, io));
            }))
            .child(Self::button(
                "ai-refresh",
                "Refresh",
                cx,
                |view, window, cx| {
                    view.call(window, cx, |runtime, form, io| {
                        runtime.refresh_models(form, io)
                    });
                },
            ))
            .child(Self::button(
                "pull-digest",
                "Pull Summary Model",
                cx,
                |view, window, cx| {
                    view.call(window, cx, |runtime, form, io| {
                        runtime.pull_model("cfg-digest-model", form, io)
                    });
                },
            ))
            .child(Self::button(
                "pull-interactive",
                "Pull Ask AI Model",
                cx,
                |view, window, cx| {
                    view.call(window, cx, |runtime, form, io| {
                        runtime.pull_model("cfg-interactive-model", form, io)
                    });
                },
            ));
        let text_generation = vec![
            self.select(
                "cfg-llm-provider",
                "Provider",
                "Ollama runs a model here, or a cloud model after ollama signin. OpenAI needs an API key.",
                cx,
            )
            .into_any_element(),
            self.field("cfg-ollama-host", "Server", "", cx).into_any_element(),
            self.field("cfg-openai-base-url", "API base", "", cx)
                .into_any_element(),
            row("API key", key, "", cx).into_any_element(),
            self.field("cfg-digest-model", "Summary model", "", cx)
                .into_any_element(),
            self.field("cfg-interactive-model", "Ask AI model", "", cx)
                .into_any_element(),
            row("Installed models", note(models_text, cx), "", cx).into_any_element(),
            tools.into_any_element(),
        ];
        let mut speech = vec![
            self.select(
                "cfg-trans-provider",
                "Engine",
                "whisper-cpp runs locally. Plugin engines appear when installed.",
                cx,
            )
            .into_any_element(),
        ];
        for index in 0..self.assets.len() {
            speech.push(self.model_row(index, cx).into_any_element());
        }
        speech.push(
            buttons()
                .child(Self::button(
                    "download-all",
                    "Download All Missing",
                    cx,
                    |view, window, cx| {
                        view.call(window, cx, |runtime, _, io| {
                            runtime.download_all_missing(io)
                        });
                    },
                ))
                .into_any_element(),
        );
        let mut languages = vec![
            wide_row(note(
                "Use a different model for a locale. The default applies otherwise.",
                cx,
            ))
            .into_any_element(),
        ];
        for index in 0..self.languages.len() {
            languages.push(self.language_line(index, cx).into_any_element());
        }
        languages.push(
            buttons()
                .child(
                    Self::button("add-language", "Add Language", cx, |view, window, cx| {
                        view.add_language(window, cx)
                    })
                    .icon(IconName::Plus),
                )
                .into_any_element(),
        );
        let cache = self.runtime.store().cache_root().display().to_string();
        let open = Self::button("open-cache", "Open Cache Folder", cx, |view, _, cx| {
            view.open_cache(cx)
        })
        .icon(IconName::FolderOpen)
        .tooltip(cache);
        let advanced = vec![
            self.field(
                CUSTOM_PATH,
                "Custom model file",
                "A custom file wins over the marked model.",
                cx,
            )
            .into_any_element(),
            row("Cache", open, "", cx).into_any_element(),
        ];
        vec![
            section("Text generation", text_generation, cx),
            section("Speech-to-Text", speech, cx),
            section("Language models", languages, cx),
            section(
                "Speaker assets",
                vec![self.speaker_row(cx).into_any_element()],
                cx,
            ),
            section("Advanced", advanced, cx),
        ]
    }

    /// One speech model: the mark and its name, then size, status and
    /// buttons in columns, then the progress or the reason it waits.
    fn model_row(&self, index: usize, cx: &mut Context<Self>) -> Div {
        let asset = &self.assets[index];
        let id = asset.id;
        let Some(state) = self.runtime.rows.get(id).cloned() else {
            return div();
        };
        let muted = cx.theme().muted_foreground;
        let view = cx.entity().downgrade();
        let mark = Radio::new(("model", index))
            .label(asset.label)
            .checked(self.chosen == Some(id))
            .on_click(move |_, window, cx| {
                let _ = view.update(cx, |view, cx| view.choose_model(id, window, cx));
            });
        let actions = row_actions(state.status).into_iter().map(|action| {
            Button::new((action.id(), index))
                .with_size(control::CONTROL)
                .when(action == RowAction::Remove, |button| button.ghost())
                .label(action.label())
                .on_click(cx.listener(move |view, _, window, cx| match action {
                    RowAction::Download => {
                        view.call(window, cx, |runtime, _, io| runtime.start_download(id, io));
                    }
                    RowAction::Cancel => {
                        view.runtime.cancel_download(id);
                        cx.notify();
                    }
                    RowAction::Remove => view.ask_remove(id, window, cx),
                }))
        });
        let detail = row_detail(&state);
        let downloading = state.status == AssetStatus::Downloading;
        let percent = state
            .progress
            .map(|p| p.downloaded as f32 * 100. / p.total.max(1) as f32)
            .unwrap_or(0.);
        div()
            .flex()
            .flex_col()
            .gap(space::XS)
            .px(space::MD)
            .py(space::SM)
            .child(
                div()
                    .flex()
                    .items_center()
                    .min_h(size::ROW)
                    .gap(space::MD)
                    .child(div().flex_1().min_w_0().child(mark))
                    .child(
                        div()
                            .w(size::STAT_COLUMN)
                            .flex()
                            .justify_end()
                            .text_color(muted)
                            .child(megabytes(asset.size)),
                    )
                    .child(
                        div()
                            .w(size::STAT_COLUMN)
                            .text_color(tint_of(status_color(state.status), cx))
                            .child(row_status(&state)),
                    )
                    .child(
                        div()
                            .flex()
                            .gap(space::XS)
                            .w(size::ACTION_COLUMN)
                            .justify_end()
                            .children(actions),
                    ),
            )
            .when(downloading, |column| {
                column.child(
                    div()
                        .pl(space::XL)
                        .child(Progress::new(("model-progress", index)).value(percent)),
                )
            })
            .when(!detail.is_empty(), |column| {
                column.child(note(detail, cx).pl(space::XL))
            })
    }

    fn language_line(&self, index: usize, cx: &mut Context<Self>) -> Div {
        let line = &self.languages[index];
        div()
            .flex()
            .items_center()
            .gap(space::SM)
            .px(space::MD)
            .py(space::SM)
            .child(
                div()
                    .w(size::LOCALE_FIELD)
                    .child(Input::new(&line.locale).with_size(control::CONTROL)),
            )
            .child(
                div()
                    .flex_1()
                    .child(Select::new(&line.model).with_size(control::CONTROL)),
            )
            .child(
                Button::new(("remove-language", index))
                    .with_size(control::CONTROL)
                    .ghost()
                    .label("Remove")
                    .on_click(cx.listener(move |view, _, _, cx| view.remove_language(index, cx))),
            )
    }

    fn speaker_row(&self, cx: &mut Context<Self>) -> Div {
        let speaker = self.runtime.speaker.clone();
        let status = match speaker.status.is_empty() {
            true => "Checking...".to_string(),
            false => speaker.status.clone(),
        };
        let controls = div()
            .flex()
            .items_center()
            .gap(space::SM)
            .child(note(status, cx))
            .when(speaker.install, |bar| {
                bar.child(Self::button(
                    "speaker-install",
                    "Install",
                    cx,
                    |view, window, cx| {
                        view.call(window, cx, |runtime, _, io| runtime.install_speaker(io));
                    },
                ))
            })
            .when(speaker.cancel, |bar| {
                bar.child(Self::button(
                    "speaker-cancel",
                    "Cancel",
                    cx,
                    |view, _, cx| {
                        view.runtime.cancel_speaker();
                        cx.notify();
                    },
                ))
            })
            .when(speaker.remove, |bar| {
                bar.child(
                    Self::button("speaker-remove", "Remove", cx, |view, window, cx| {
                        view.ask_remove(SPEAKER, window, cx)
                    })
                    .ghost(),
                )
            });
        row(
            "Runtime and model",
            controls,
            "Who's Speaking needs these. They download once.",
            cx,
        )
    }

    // --- Digest ---------------------------------------------------------

    fn digest_page(&self, cx: &mut Context<Self>) -> Vec<AnyElement> {
        let when = vec![
            self.field("cfg-digest-min-lines", "After lines", "", cx)
                .into_any_element(),
            self.field("cfg-digest-min-interval", "Wait seconds", "", cx)
                .into_any_element(),
            self.field(
                "cfg-digest-max-lines",
                "Force after",
                "Optional. Blank means twice After lines.",
                cx,
            )
            .into_any_element(),
        ];
        let advanced = vec![
            self.field(
                "cfg-digest-compact",
                "Compaction threshold",
                "Compress history when it exceeds this token count. Change it only if you hit context-length errors.",
                cx,
            )
            .into_any_element(),
        ];
        vec![
            section("When to summarize", when, cx),
            section("Advanced", advanced, cx),
        ]
    }

    // --- Audio ----------------------------------------------------------

    /// A source's level bars against the silence threshold, and its permission.
    fn meter(&self, label: &'static str, meter: &Meter, cx: &App) -> Div {
        let theme = cx.theme();
        let (quiet, loud) = (theme.muted_foreground.opacity(tint::FAINT), theme.success);
        let silence = self.runtime.silence_threshold();
        let rise = size::METER_HEIGHT - size::METER_BAR;
        let bars = meter.levels.iter().map(|level| {
            let step = level_step(*level, silence);
            let height = size::METER_BAR + rise * (step as f32 / LEVEL_TOP as f32);
            div()
                .w(size::METER_BAR)
                .h(height)
                .rounded_full()
                .bg(if step == 0 { quiet } else { loud })
        });
        let (status, color) = match meter.permission {
            Some(permission) => (permission.text, permission.color),
            None => ("Not tested", None),
        };
        let color = match color {
            Some(_) => tint_of(color, cx),
            None => theme.muted_foreground,
        };
        row(
            label,
            div()
                .flex()
                .items_center()
                .gap(space::MD)
                .child(
                    div()
                        .flex()
                        .items_end()
                        .gap(space::XXS)
                        .h(size::METER_HEIGHT)
                        .children(bars),
                )
                .child(div().text_color(color).child(status)),
            "",
            cx,
        )
    }

    fn audio_page(&self, cx: &mut Context<Self>) -> Vec<AnyElement> {
        let running = self.runtime.probe_running();
        let check = buttons()
            .child(Self::button(
                "grant-system",
                "Grant System Audio",
                cx,
                |_, _, _| open_privacy("Privacy_ScreenCapture"),
            ))
            .child(Self::button(
                "grant-mic",
                "Open Microphone Settings",
                cx,
                |_, _, _| open_privacy("Privacy_Microphone"),
            ))
            .child(
                Self::button("audio-stop", "Stop", cx, |view, _, cx| {
                    view.runtime.stop_probe();
                    cx.notify();
                })
                .icon(IconName::Square)
                .disabled(!running),
            )
            .child(
                Self::button("audio-start", "Start Test", cx, |view, window, cx| {
                    view.call(window, cx, |runtime, form, io| {
                        runtime.start_probe(form, io)
                    });
                })
                .primary()
                .icon(IconName::Play)
                .disabled(running),
            );
        let meters = &self.runtime.meters;
        let input_check = vec![
            wide_row(note(
                "Start a 15 second test and make a sound. Nothing is saved.",
                cx,
            ))
            .into_any_element(),
            self.meter("Microphone", &meters[0], cx).into_any_element(),
            self.meter("System audio", &meters[1], cx)
                .into_any_element(),
            check.into_any_element(),
        ];
        let microphone = vec![
            self.select(
                DEVICE,
                "Input device",
                "A removed device falls back to the system default with a warning.",
                cx,
            )
            .into_any_element(),
        ];
        let processing = vec![
            self.field("cfg-trans-chunk", "Chunk duration", "", cx)
                .into_any_element(),
            self.field("cfg-trans-overlap", "Overlap", "", cx)
                .into_any_element(),
            self.field("cfg-trans-silence", "Silence threshold", "", cx)
                .into_any_element(),
            self.field("cfg-trans-pause", "Pause duration", "", cx)
                .into_any_element(),
        ];
        let speakers = vec![
            self.switch("cfg-diarization-enabled", "Label speakers", "", cx)
                .into_any_element(),
            self.select("cfg-diarization-device", "Device", "", cx)
                .into_any_element(),
            self.field("cfg-diarization-interval", "Update every", "", cx)
                .into_any_element(),
        ];
        vec![
            section("Input check", input_check, cx),
            section("Microphone", microphone, cx),
            section("Audio processing", processing, cx),
            section("Who's Speaking", speakers, cx),
        ]
    }

    // --- Sessions -------------------------------------------------------

    fn sessions_page(&self, cx: &mut Context<Self>) -> Vec<AnyElement> {
        let storage = vec![
            self.field("cfg-output-dir", "Output folder", "", cx)
                .into_any_element(),
            self.switch("cfg-output-save-audio", "Save audio", "", cx)
                .into_any_element(),
            self.switch(
                "cfg-output-save-notes-history",
                "Keep notes history",
                "",
                cx,
            )
            .into_any_element(),
            self.switch("cfg-output-save-context", "Save background for AI", "", cx)
                .into_any_element(),
            self.switch("cfg-output-save-debug-log", "Save debug log", "", cx)
                .into_any_element(),
            self.switch("cfg-output-auto-label", "Auto-label sessions", "", cx)
                .into_any_element(),
        ];
        let vocabulary = vec![
            wide_row(
                div()
                    .flex()
                    .flex_col()
                    .gap(space::SM)
                    .child(note(
                        "Names, acronyms, or jargon the recognizer should watch for. One per line.",
                        cx,
                    ))
                    .child(Textarea::new(&self.hints)),
            )
            .into_any_element(),
        ];
        let templates = vec![
            self.switch(
                "cfg-show-builtin-templates",
                "Show built-in note types",
                "Bundled examples appear in the picker. Turn it off to see only your own.",
                cx,
            )
            .into_any_element(),
        ];
        vec![
            section("Session storage", storage, cx),
            section("Custom vocabulary", vocabulary, cx),
            section("Note Types", templates, cx),
        ]
    }

    // --- The frame ------------------------------------------------------

    /// The page list, and Open Config File under it: the file every page
    /// edits.
    fn nav(&self, cx: &mut Context<Self>) -> Div {
        let rows = PAGES.iter().map(|(page, label)| {
            let (page, current) = (*page, *page == self.page);
            source_row(*label, current, cx)
                .child(
                    Icon::new(page_icon(page))
                        .size(size::ICON)
                        .text_color(cx.theme().primary),
                )
                .child(*label)
                .on_click(cx.listener(move |view, _, _, cx| {
                    view.page = page;
                    cx.notify();
                }))
        });
        let path = self.runtime.config_file().display().to_string();
        div()
            .flex_none()
            .w(size::SETTINGS_NAV)
            .h_full()
            .flex()
            .flex_col()
            .gap(space::XXS)
            .p(space::SM)
            .border_r(size::HAIRLINE)
            .border_color(cx.theme().border)
            .children(rows)
            .child(div().flex_1())
            .child(
                source_row("open-config", false, cx)
                    .child(Icon::new(IconName::FileText).size(size::ICON))
                    .child("Open Config File")
                    .tooltip(move |window, cx| Tooltip::new(path.clone()).build(window, cx))
                    .on_click(cx.listener(|view, _, window, cx| view.open_config(window, cx))),
            )
    }

    /// The title, then Revert and Save at the right end. Save waits for a
    /// change.
    fn header(&self, cx: &mut Context<Self>) -> Stateful<Div> {
        let changed = self.has_unsaved_changes(cx);
        pane_header("settings-title", &self.drag, cx)
            .child(pane_title("Settings").flex_1())
            .child(
                tool("revert", IconName::Undo2, "Revert")
                    .tooltip("Read config.yaml again")
                    .on_click(cx.listener(|view, _, window, cx| view.revert(window, cx))),
            )
            .child(
                Button::new("save")
                    .primary()
                    .with_size(control::CONTROL)
                    .label("Save")
                    .disabled(!changed)
                    .on_click(cx.listener(|view, _, window, cx| view.save(window, cx))),
            )
    }
}

fn page_icon(page: Page) -> IconName {
    match page {
        Page::Models => IconName::Bot,
        Page::Digest => IconName::BookOpen,
        Page::Audio => IconName::Mic,
        Page::Sessions => IconName::Folder,
    }
}

impl Render for SettingsView {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let items = match self.page {
            Page::Models => self.models_page(cx),
            Page::Digest => self.digest_page(cx),
            Page::Audio => self.audio_page(cx),
            Page::Sessions => self.sessions_page(cx),
        };
        div()
            .key_context("Settings")
            .track_focus(&self.focus)
            .on_key_down(cx.listener(|view, event: &KeyDownEvent, window, cx| {
                if event.keystroke.key == "escape" {
                    view.back(window, cx);
                }
            }))
            .size_full()
            .flex()
            .flex_col()
            .child(self.header(cx))
            .child(
                div().flex().flex_1().min_h_0().child(self.nav(cx)).child(
                    div()
                        .id("settings-page")
                        .flex_1()
                        .min_w_0()
                        .overflow_y_scroll()
                        .child(
                            div()
                                .max_w(size::FORM_PAGE)
                                .flex()
                                .flex_col()
                                .gap(space::XL)
                                .px(space::XL)
                                .py(space::XL)
                                .children(items),
                        ),
                ),
            )
    }
}
